from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov5m.pt")

# Initialize the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform license plate detection
    results = model.predict(frame)
    result = results[0]

    # Filter results to only include license plates
    license_plate_results = [box for box in result.boxes if result.names[box.cls[0].item()] == 'license_plate']

    for box in license_plate_results:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)

        # Print license plate information
        print("License Plate Detected!")
        print("Object type:", class_id)
        print("Coordinates:", cords)
        print("Probability:", conf)
        print("---")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
