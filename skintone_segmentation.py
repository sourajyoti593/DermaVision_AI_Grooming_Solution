import cv2
import numpy as np

def convert_to_lab(image_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # Convert RGB to LAB
    
    # Split LAB channels
    L, A, B = cv2.split(lab_image)
    
    return L, A, B, lab_image

# Example usage
image_path = "face.jpg"
L, A, B, lab_image = convert_to_lab(image_path)

cv2.imshow("L Channel (Lightness)", L)
cv2.imshow("A Channel (Red-Green)", A)
cv2.imshow("B Channel (Blue-Yellow)", B)
cv2.waitKey(0)
cv2.destroyAllWindows()



def extract_lab_values(image_path):
    L, A, B, _ = convert_to_lab(image_path)
    
    # Compute mean values
    mean_L = np.mean(L)
    mean_A = np.mean(A)
    mean_B = np.mean(B)
    
    return mean_L, mean_A, mean_B

# Example
image_path = "face.jpg"
lab_values = extract_lab_values(image_path)
print("Mean L, A, B Values:", lab_values)



from sklearn.neighbors import KNeighborsClassifier

# Sample dataset (L, A, B values mapped to skin tones)
skin_tone_data = np.array([
    [80, 128, 128],  # Fair
    [70, 135, 140],  # Medium
    [50, 145, 155],  # Dark
    [90, 125, 130],  # Very Fair
    [60, 140, 150],  # Brown
])

skin_tone_labels = ["Fair", "Medium", "Dark", "Very Fair", "Brown"]

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(skin_tone_data, skin_tone_labels)

# Function to classify skin tone
def classify_skin_tone(image_path):
    mean_L, mean_A, mean_B = extract_lab_values(image_path)
    skin_tone = knn.predict([[mean_L, mean_A, mean_B]])
    return skin_tone[0]

# Example usage
image_path = "face.jpg"
result = classify_skin_tone(image_path)
print("Predicted Skin Tone:", result)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to LAB and get mean L, A, B values
    mean_L, mean_A, mean_B = extract_lab_values(frame)

    # Predict skin tone
    skin_tone = knn.predict([[mean_L, mean_A, mean_B]])[0]

    # Display result
    cv2.putText(frame, f"Skin Tone: {skin_tone}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Skin Tone Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
