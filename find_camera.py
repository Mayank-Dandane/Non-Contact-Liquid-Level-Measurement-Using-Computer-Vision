import cv2

print("Scanning camera indexes...")
for i in range(3):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW = Windows DirectShow, faster
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"  Index {i} → {'WORKS ✓' if ret else 'Opens but no frame'}")
        cap.release()
    else:
        print(f"  Index {i} → not found")

print("Done.")