import cv2
import numpy as np

def analyze_beard_health(image_path):
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize to 0-255
    sobel_magnitude = (sobel_magnitude / sobel_magnitude.max() * 255).astype(np.uint8)

    # Compute average edge intensity
    edge_intensity = np.mean(sobel_magnitude)

    # Classify beard health based on edge intensity
    if edge_intensity > 100:
        beard_health = "Thick & Healthy"
    elif edge_intensity > 50:
        beard_health = "Moderate Density"
    else:
        beard_health = "Thin / Patchy Beard"

    print(f"Beard Health: {beard_health}")

    # Display result
    cv2.imshow("Beard Edge Detection", sobel_magnitude)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return beard_health

# Example usage
image_path = "beard.jpg"
health_status = analyze_beard_health(image_path)
print("Beard Health Status:", health_status)



from ultralytics import YOLO
import cv2

# Load a trained YOLOv8 model for beard classification
model = YOLO("beard_classification_yolov8.pt")  # Replace with trained model

# Beard style labels
beard_styles = ["Clean Shaven", "Stubble", "Goatee", "Full Beard"]

def classify_beard(image_path):
    image = cv2.imread(image_path)
    results = model(image)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{beard_styles[class_id]} ({confidence:.2f})"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display result
    cv2.imshow("Beard Classification", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "beard.jpg"
classify_beard(image_path)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Sobel Edge Detection for Beard Health
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_intensity = np.mean(sobel_magnitude)

    # Classify Beard Health
    if edge_intensity > 100:
        beard_health = "Thick & Healthy"
    elif edge_intensity > 50:
        beard_health = "Moderate Density"
    else:
        beard_health = "Thin / Patchy Beard"

    # Beard Classification with YOLOv8
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{beard_styles[class_id]} ({confidence:.2f}) - {beard_health}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Beard Health & Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
