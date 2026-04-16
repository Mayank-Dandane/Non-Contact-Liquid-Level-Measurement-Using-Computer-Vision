import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def nothing(x): pass
cv2.namedWindow("HSV Tuner")
cv2.createTrackbar("H Low",  "HSV Tuner", 140, 180, nothing)
cv2.createTrackbar("H High", "HSV Tuner", 180, 180, nothing)
cv2.createTrackbar("S Low",  "HSV Tuner",  30, 255, nothing)
cv2.createTrackbar("S High", "HSV Tuner", 180, 255, nothing)
cv2.createTrackbar("V Low",  "HSV Tuner", 100, 255, nothing)
cv2.createTrackbar("V High", "HSV Tuner", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret: continue

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hl   = cv2.getTrackbarPos("H Low",  "HSV Tuner")
    hh   = cv2.getTrackbarPos("H High", "HSV Tuner")
    sl   = cv2.getTrackbarPos("S Low",  "HSV Tuner")
    sh   = cv2.getTrackbarPos("S High", "HSV Tuner")
    vl   = cv2.getTrackbarPos("V Low",  "HSV Tuner")
    vh   = cv2.getTrackbarPos("V High", "HSV Tuner")

    mask = cv2.inRange(hsv, np.array([hl,sl,vl]), np.array([hh,sh,vh]))
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Original", frame)
    cv2.imshow("Mask",     mask)
    cv2.imshow("HSV Tuner", result)

    print(f"\rHSV Lower: [{hl},{sl},{vl}]  Upper: [{hh},{sh},{vh}]", end="")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()