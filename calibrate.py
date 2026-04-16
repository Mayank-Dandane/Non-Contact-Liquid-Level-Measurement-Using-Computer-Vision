import cv2
import numpy as np
import json

CAMERA_INDEX = 0
CALIB_FILE = "calibration.json"

# Define your known heights here — matches your container markings
KNOWN_HEIGHTS_MM = [36, 72, 108, 144, 180]  # 100ml, 300ml, 500ml, 700ml, 900ml

points = []
click_y = None

def mouse_callback(event, x, y, flags, param):
    global click_y
    if event == cv2.EVENT_LBUTTONDOWN:
        click_y = y

def main():
    global click_y

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)

    current_index = 0  # which height we are currently recording

    print("=" * 50)
    print("  CALIBRATION TOOL")
    print("=" * 50)
    print(f"  You have {len(KNOWN_HEIGHTS_MM)} points to record:")
    for i, h in enumerate(KNOWN_HEIGHTS_MM):
        label = [100, 300, 500, 700, 900][i]
        print(f"    Point {i+1}: {label} ml line = {h} mm")
    print("=" * 50)
    print("  Click exactly on the mark shown at top of window")
    print("  Press SPACE to confirm each click")
    print("  Press U to undo last point")
    print("  Press S to save when all points done")
    print("=" * 50)

    ml_labels = [100, 300, 500, 700, 900]

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()

        # Draw already recorded points
        for i, (py, mm) in enumerate(points):
            cv2.line(display, (0, py), (display.shape[1], py), (0, 255, 0), 1)
            cv2.putText(display, f"{ml_labels[i]} ml = {mm}mm (y={py})",
                        (10, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Show current target
        if current_index < len(KNOWN_HEIGHTS_MM):
            target_ml = ml_labels[current_index]
            target_mm = KNOWN_HEIGHTS_MM[current_index]
            cv2.putText(display,
                        f"CLICK ON: {target_ml} ml line ({target_mm} mm)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(display,
                        f"Then press SPACE to confirm",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Show pending click as yellow line
            if click_y is not None:
                cv2.line(display, (0, click_y), (display.shape[1], click_y),
                         (0, 255, 255), 1)
                cv2.putText(display, f"Pending: y={click_y} → press SPACE",
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(display,
                        f"All {len(KNOWN_HEIGHTS_MM)} points done! Press S to save.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)

        # Points counter
        cv2.putText(display, f"Recorded: {len(points)}/{len(KNOWN_HEIGHTS_MM)}",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        # SPACE = confirm current click
        if key == ord(' '):
            if click_y is not None and current_index < len(KNOWN_HEIGHTS_MM):
                mm = KNOWN_HEIGHTS_MM[current_index]
                points.append((click_y, mm))
                print(f"  ✓ Point {current_index+1}: {ml_labels[current_index]} ml "
                      f"→ pixel_y={click_y}, {mm} mm")
                current_index += 1
                click_y = None
            else:
                print("  Click on the line first, then press SPACE.")

        # U = undo
        elif key == ord('u') or key == ord('U'):
            if points:
                removed = points.pop()
                current_index -= 1
                print(f"  Undone point {current_index+1}")

        # S = save
        elif key == ord('s') or key == ord('S'):
            if len(points) < 3:
                print(f"  Need at least 3 points! Only have {len(points)}.")
                continue

            pixel_ys = np.array([p[0] for p in points], dtype=float)
            real_mms = np.array([p[1] for p in points], dtype=float)

            coeffs = np.polyfit(pixel_ys, real_mms, 1)
            slope, intercept = coeffs

            predicted = np.polyval(coeffs, pixel_ys)
            ss_res = np.sum((real_mms - predicted) ** 2)
            ss_tot = np.sum((real_mms - np.mean(real_mms)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

            calib_data = {
                "slope":     float(slope),
                "intercept": float(intercept),
                "r_squared": float(r2),
                "points":    [{"pixel_y": int(p[0]), "real_mm": float(p[1])}
                              for p in points]
            }

            with open(CALIB_FILE, "w") as f:
                json.dump(calib_data, f, indent=2)

            print("\n" + "=" * 50)
            print(f"  Saved to '{CALIB_FILE}'")
            print(f"  slope     = {slope:.5f}")
            print(f"  intercept = {intercept:.5f}")
            print(f"  R²        = {r2:.4f}  (1.0 = perfect)")
            print("=" * 50)
            break

        elif key == ord('q') or key == ord('Q'):
            print("Cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()