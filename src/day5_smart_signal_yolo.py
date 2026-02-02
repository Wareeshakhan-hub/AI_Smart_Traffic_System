# Day 5: Smart Traffic Signal using YOLO Vehicle Detection
import cv2
from ultralytics import YOLO

# -------------------------------
# 1. Traffic Signal Logic (Day 3)
# -------------------------------
def get_traffic_level(vehicle_count):
    if vehicle_count <= 5:
        return "LOW", 15
    elif vehicle_count <= 15:
        return "MEDIUM", 30
    else:
        return "HIGH", 45


# -------------------------------
# 2. Load YOLO Model
# -------------------------------
model = YOLO("yolov8n.pt")

# -------------------------------
# 3. Video Source
# -------------------------------
# 0 = webcam
# OR replace with video path like: "data/traffic.mp4"
cap = cv2.VideoCapture("data/traffic.mp4")
# Vehicle classes to count
VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]

# -------------------------------
# 4. Main Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (900, 550))

    # YOLO Detection
    results = model(frame, stream=True)
    vehicle_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in VEHICLE_CLASSES:
                vehicle_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

    # -------------------------------
    # 5. Smart Signal Decision
    # -------------------------------
    traffic_level, green_time = get_traffic_level(vehicle_count)

    # -------------------------------
    # 6. Display Information
    # -------------------------------
    cv2.putText(frame, f"Vehicles Detected: {vehicle_count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    cv2.putText(frame, f"Traffic Level: {traffic_level}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)

    cv2.putText(frame, f"Green Signal Time: {green_time} sec",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 128, 0), 2)

    cv2.imshow("Day 5 - Smart Traffic Signal (YOLO)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# 7. Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
