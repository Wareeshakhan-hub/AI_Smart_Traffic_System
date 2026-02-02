import cv2

cap = cv2.VideoCapture("data/traffic.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Day 1 - Traffic Video Test", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

