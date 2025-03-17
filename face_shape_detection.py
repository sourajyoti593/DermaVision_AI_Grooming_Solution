from ultralytics import YOLO
import cv2

# Load a trained YOLOv8 model for face shape detection
model = YOLO("face_shape_yolov8.pt")  # Replace with your trained model

# Load image
image_path = "face_image.jpg"
image = cv2.imread(image_path)

# Run YOLOv8 on the image
results = model(image)

# Define face shape labels
face_shapes = ["Oval", "Round", "Square", "Heart", "Diamond", "Oblong"]

# Draw detections
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        confidence = box.conf[0]  # Confidence score
        class_id = int(box.cls[0])  # Class ID
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{face_shapes[class_id]} ({confidence:.2f})"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show image
cv2.imshow("Face Shape Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
