import cv2
import numpy as np
import time
import csv
import os
import threading
import winsound
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────
CAMERA_INDEX      = 0
LOG_FILE          = "level_log.csv"
ROI_COLOR         = (0, 255, 255)
LEVEL_COLOR       = (0, 255, 0)
EMPTY_THRESHOLD   = 5       # % below which = empty
WARNING_THRESHOLD = 90      # % above which = warning
# ──────────────────────────────────────────────────────────

roi = {"x": 0, "y": 0, "w": 0, "h": 0, "drawing": False, "set": False}
drawing_start      = None
roi_real_height_mm = None

# ─── BUZZER STATE ─────────────────────────────────────────
buzzer_state     = "none"
buzzer_thread    = None
buzzer_stop_flag = threading.Event()

def _continuous_warning_beep(stop_event):
    """Fast high beep — 90%+ warning"""
    while not stop_event.is_set():
        winsound.Beep(1200, 200)
        time.sleep(0.1)

def _continuous_empty_beep(stop_event):
    """Slow low beep — empty alert"""
    while not stop_event.is_set():
        winsound.Beep(500, 600)
        time.sleep(0.3)

def start_buzzer(mode):
    global buzzer_thread, buzzer_stop_flag, buzzer_state
    if buzzer_state == mode:
        return  # already running this mode
    stop_buzzer()
    buzzer_stop_flag = threading.Event()
    target = _continuous_warning_beep if mode == "warning" else _continuous_empty_beep
    buzzer_thread = threading.Thread(target=target, args=(buzzer_stop_flag,), daemon=True)
    buzzer_thread.start()
    buzzer_state = mode
    print(f"  [BUZZER] Started — {mode.upper()}")

def stop_buzzer():
    global buzzer_state
    buzzer_stop_flag.set()
    buzzer_state = "none"

def check_and_buzz(pct):
    global buzzer_state
    if pct >= WARNING_THRESHOLD:
        start_buzzer("warning")
    elif pct <= EMPTY_THRESHOLD:
        start_buzzer("empty")
    else:
        if buzzer_state != "none":
            stop_buzzer()
            print("  [BUZZER] Stopped — level normal")

# ─── MOUSE ────────────────────────────────────────────────
def mouse_callback(event, x, y, flags, param):
    global drawing_start
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_start  = (x, y)
        roi["drawing"] = True
        roi["set"]     = False
    elif event == cv2.EVENT_MOUSEMOVE and roi["drawing"]:
        roi["x"] = min(drawing_start[0], x)
        roi["y"] = min(drawing_start[1], y)
        roi["w"] = abs(x - drawing_start[0])
        roi["h"] = abs(y - drawing_start[1])
    elif event == cv2.EVENT_LBUTTONUP:
        roi["x"] = min(drawing_start[0], x)
        roi["y"] = min(drawing_start[1], y)
        roi["w"] = abs(x - drawing_start[0])
        roi["h"] = abs(y - drawing_start[1])
        roi["drawing"] = False
        roi["set"]     = True

# ─── DETECTION ────────────────────────────────────────────
def detect_level(frame, roi_box):
    x, y, w, h = roi_box["x"], roi_box["y"], roi_box["w"], roi_box["h"]
    if w < 10 or h < 10:
        return None, None

    crop = frame[y:y+h, x:x+w]
    if crop.size == 0:
        return None, None

    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges   = cv2.Canny(blurred, 20, 60)

    kernel = np.ones((3, 3), np.uint8)
    edges  = cv2.dilate(edges, kernel, iterations=1)

    # Mask out side walls (left 15% and right 15%)
    # Real water surface spans horizontally — walls are vertical on sides
    wall_margin  = int(w * 0.15)
    masked_edges = edges.copy()
    masked_edges[:, :wall_margin]     = 0
    masked_edges[:, w - wall_margin:] = 0

    row_sums        = np.sum(masked_edges, axis=1)
    row_sums_smooth = np.convolve(row_sums, np.ones(5)/5, mode='same')

    margin_top    = int(h * 0.05)
    margin_bottom = int(h * 0.90)
    search_sums   = row_sums_smooth[margin_top:margin_bottom]

    best_local = int(np.argmax(search_sums))
    best_score = search_sums[best_local]

    # Check horizontal spread of best row
    best_row_idx    = best_local + margin_top
    best_row_pixels = masked_edges[best_row_idx, :]
    nonzero_cols    = np.count_nonzero(best_row_pixels)
    spread_ratio    = nonzero_cols / max((w - 2 * wall_margin), 1)

    # EMPTY: edge score too low OR spread too narrow
    if best_score < 30 or spread_ratio < 0.20:
        return "EMPTY", edges

    top_row_local  = best_row_idx
    top_row_global = y + top_row_local

    if roi_real_height_mm is not None:
        fraction_from_bottom = 1.0 - (top_row_local / h)
        level_mm = fraction_from_bottom * roi_real_height_mm
    else:
        level_mm = None

    return (top_row_global, top_row_local, h, level_mm), edges

# ─── OVERLAY ──────────────────────────────────────────────
def draw_overlay(frame, roi_box, result, fps, alert_state):
    x, y, w, h = roi_box["x"], roi_box["y"], roi_box["w"], roi_box["h"]

    if roi_box["set"] or roi_box["drawing"]:
        cv2.rectangle(frame, (x, y), (x+w, y+h), ROI_COLOR, 1)
        cv2.line(frame, (x, y+h), (x+w, y+h), (0, 165, 255), 1)
        cv2.putText(frame, "0 mm", (x+4, y+h-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        if roi_real_height_mm is not None:
            cv2.putText(frame, f"{roi_real_height_mm:.0f} mm", (x+4, y+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    # ── EMPTY ─────────────────────────────────────────────
    if result == "EMPTY":
        cv2.rectangle(frame, (8, 8), (300, 115), (0, 0, 0), -1)
        cv2.rectangle(frame, (8, 8), (300, 115), (0, 0, 255), 2)
        cv2.putText(frame, "CONTAINER EMPTY",
                    (14, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 255), 2)
        cv2.putText(frame, "Level : 0.0 mm  |  Fill : 0.0%",
                    (14, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 100, 255), 1)
        cv2.putText(frame, f"FPS   : {fps:.1f}",
                    (14, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # ── VALID RESULT ──────────────────────────────────────
    elif result is not None:
        top_row_global, top_row_local, roi_h, level_mm = result
        cv2.line(frame, (x, top_row_global), (x+w, top_row_global), LEVEL_COLOR, 2)

        box_color = (0, 0, 255) if alert_state == "warning" else (0, 255, 0)
        cv2.rectangle(frame, (8, 8), (300, 135), (0, 0, 0), -1)
        cv2.rectangle(frame, (8, 8), (300, 135), box_color, 1)

        if level_mm is not None:
            cv2.putText(frame, f"Level : {level_mm:.1f} mm",
                        (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 0), 2)

            if roi_real_height_mm:
                pct = min(100.0, (level_mm / roi_real_height_mm) * 100)

                if pct >= WARNING_THRESHOLD:
                    fill_color = (0, 0, 255)
                    label      = f"Fill  : {pct:.1f}%  WARNING"
                elif pct <= EMPTY_THRESHOLD:
                    fill_color = (0, 100, 255)
                    label      = f"Fill  : {pct:.1f}%  LOW"
                else:
                    fill_color = (180, 255, 180)
                    label      = f"Fill  : {pct:.1f}%"

                cv2.putText(frame, label,
                            (14, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fill_color, 1)

                # Fill bar
                bar_x, bar_y, bar_w, bar_h = 14, 78, 200, 12
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                filled_w = int(bar_w * pct / 100)
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + filled_w, bar_y + bar_h), fill_color, -1)
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)

                check_and_buzz(pct)

        else:
            cv2.putText(frame, "Set ROI height (press H)",
                        (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        cv2.putText(frame, f"Pixel Y: {top_row_global} px",
                    (14, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1)
        cv2.putText(frame, f"FPS    : {fps:.1f}",
                    (14, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # ── NO DETECTION ──────────────────────────────────────
    else:
        cv2.rectangle(frame, (8, 8), (280, 55), (0, 0, 0), -1)
        cv2.putText(frame, "No surface detected",
                    (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)

    h_frame = frame.shape[0]
    cv2.putText(frame,
                "Drag=ROI | H=SetHeight | L=Log | R=Reset | E=Edges | Q=Quit",
                (8, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (200, 200, 200), 1)

# ─── LOGGING ──────────────────────────────────────────────
def log_reading(level_mm):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "level_mm"])
        writer.writerow([datetime.now().strftime("%H:%M:%S"), round(level_mm, 2)])
    print(f"  [LOG] {datetime.now().strftime('%H:%M:%S')}  →  {level_mm:.1f} mm")

# ─── MAIN ─────────────────────────────────────────────────
def main():
    global roi_real_height_mm

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Liquid Level Monitor")
    cv2.setMouseCallback("Liquid Level Monitor", mouse_callback)

    print("=" * 55)
    print("  LIQUID LEVEL MONITOR — Proportional ROI Mode")
    print("=" * 55)
    print("  WORKFLOW:")
    print("  1. Drag mouse to draw ROI around container")
    print("  2. Press H → type the real height in mm")
    print("  3. Green line = detected water surface")
    print(f"  ALERTS: Continuous beep at >={WARNING_THRESHOLD}% and when EMPTY")
    print("  L=Log | R=Reset | E=EdgeView | Q=Quit")
    print("=" * 55)

    prev_time   = time.time()
    last_result = None
    show_edges  = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        edge_vis = None
        if roi["set"]:
            last_result, edge_vis = detect_level(frame, roi)

            # Buzz check for EMPTY state
            if last_result == "EMPTY":
                check_and_buzz(0.0)

        display = frame.copy()
        if show_edges and edge_vis is not None:
            ex, ey, ew, eh = roi["x"], roi["y"], roi["w"], roi["h"]
            edge_bgr = cv2.cvtColor(edge_vis, cv2.COLOR_GRAY2BGR)
            display[ey:ey+eh, ex:ex+ew] = edge_bgr

        draw_overlay(display, roi, last_result, fps, buzzer_state)
        cv2.imshow("Liquid Level Monitor", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            stop_buzzer()
            break
        elif key == ord('r') or key == ord('R'):
            roi["set"]         = False
            last_result        = None
            roi_real_height_mm = None
            stop_buzzer()
            print("  ROI reset.")
        elif key == ord('h') or key == ord('H'):
            try:
                val = float(input("  Enter real height of ROI in mm: "))
                roi_real_height_mm = val
                print(f"  ROI height set to {val} mm")
            except ValueError:
                print("  Invalid input.")
        elif key == ord('l') or key == ord('L'):
            if last_result and last_result != "EMPTY" and last_result[3] is not None:
                log_reading(last_result[3])
            elif last_result == "EMPTY":
                print("  Container is empty — nothing to log.")
            else:
                print("  No valid reading. Draw ROI and press H first.")
        elif key == ord('e') or key == ord('E'):
            show_edges = not show_edges
            print(f"  Edge view: {'ON' if show_edges else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()