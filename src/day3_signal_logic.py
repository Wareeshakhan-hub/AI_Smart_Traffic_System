import cv2
import numpy as np
import time

# Load traffic video
cap = cv2.VideoCapture("data/traffic.mp4")

# Background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=40
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 500))

    fg_mask = bg_subtractor.apply(frame)
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    vehicle_count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            vehicle_count += 1

    # Traffic level decision
    if vehicle_count < 5:
        traffic_level = "LOW"
        green_time = 15
    elif vehicle_count < 15:
        traffic_level = "MEDIUM"
        green_time = 30
    else:
        traffic_level = "HIGH"
        green_time = 45

    # Display info
    cv2.putText(
        frame,
        f"Traffic: {traffic_level}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Green Signal Time: {green_time} sec",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv2.imshow("Day 3 - Smart Signal Logic", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
