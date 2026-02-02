import cv2
import numpy as np

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

    # Resize for consistency
    frame = cv2.resize(frame, (800, 500))

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Remove noise
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours (moving objects)
    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    vehicle_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:   # filter small objects
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            vehicle_count += 1

    # Traffic level logic
    if vehicle_count < 5:
        traffic_level = "LOW"
    elif vehicle_count < 15:
        traffic_level = "MEDIUM"
    else:
        traffic_level = "HIGH"

    # Display text
    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count} | Traffic: {traffic_level}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv2.imshow("Day 2 - Vehicle Detection", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
