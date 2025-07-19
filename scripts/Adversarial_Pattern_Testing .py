# Adversarial Pattern Testing 
import numpy as np
import matplotlib.pyplot as plt
import os

# Use cv2 from opencv-python-headless
try:
    import cv2
except ImportError:
    print("Please install opencv-python-headless (pip install opencv-python-headless) for server/headless environments.")
    raise

def load_test_image(filename=None):
    """
    Load a local test adversarial pattern. User provides path, or uses default.
    """
    if filename is None:
        filename = "/filename.png"
    if os.path.exists(filename):
        print(f"Loading test pattern: {filename}")
        return filename
    else:
        print(f"Error: File not found at {filename}")
        print("Please check the file path or upload the image.")
        return None

def test_adversarial_pattern(image_path):
    """
    Test adversarial properties of a pattern.
    """
    if image_path is None:
        print("No image provided for testing.")
        return

    pattern = cv2.imread(image_path)
    if pattern is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    print(f"Testing pattern: {image_path}")
    print(f"Image dimensions: {pattern.shape}")

    # Initialize detectors
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    # Detection
    faces = face_cascade.detectMultiScale(pattern, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    eyes = eye_cascade.detectMultiScale(pattern, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
    smiles = smile_cascade.detectMultiScale(pattern, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

    print("\n=== DETECTION RESULTS ===")
    print(f"False face detections: {len(faces)}")
    print(f"False eye detections: {len(eyes)}")
    print(f"False smile detections: {len(smiles)}")

    gray = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_count = np.sum(edges > 0)
    edge_density = edge_count / (gray.shape[0] * gray.shape[1])
    contrast = gray.std()
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    high_freq_energy = np.sum(magnitude_spectrum > np.percentile(magnitude_spectrum, 80))

    print("\n=== PATTERN ANALYSIS ===")
    print(f"Edge density: {edge_density:.4f} ({edge_count} edge pixels)")
    print(f"Contrast level: {contrast:.2f}")
    print(f"High frequency energy: {high_freq_energy}")

    false_positive_score = len(faces) + len(eyes) + len(smiles)
    complexity_score = (edge_density * 100) + (contrast / 10) + (high_freq_energy / 1000)
    adversarial_score = false_positive_score + complexity_score

    print(f"\n=== ADVERSARIAL EFFECTIVENESS ===")
    print(f"False positive score: {false_positive_score}")
    print(f"Complexity score: {complexity_score:.2f}")
    print(f"Overall adversarial score: {adversarial_score:.2f}")

    if false_positive_score > 5:
        print("✅ EXCELLENT: High false positive rate - very confusing to AI")
    elif false_positive_score > 2:
        print("✅ GOOD: Moderate false positives - effective confusion")
    elif false_positive_score > 0:
        print("⚠️  FAIR: Some false positives - limited effectiveness")
    else:
        print("❌ POOR: No false positives - may not confuse AI effectively")

    if edge_density > 0.1:
        print("✅ HIGH COMPLEXITY: Dense edge patterns good for disruption")
    elif edge_density > 0.05:
        print("⚠️  MEDIUM COMPLEXITY: Moderate edge density")
    else:
        print("❌ LOW COMPLEXITY: May need more geometric detail")

    # Visualization
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(pattern, cv2.COLOR_BGR2RGB))
    plt.title('Original Pattern')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title(f'Edges (Density: {edge_density:.4f})')
    plt.axis('off')

    detection_img = pattern.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(detection_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(detection_img, 'FACE', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for (x, y, w, h) in eyes:
        cv2.rectangle(detection_img, (x, y), (x+w, y+h), (255, 0, 0), 1)
        cv2.putText(detection_img, 'EYE', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
    plt.title(f'False Detections ({false_positive_score} total)')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(magnitude_spectrum, cmap='hot')
    plt.title('Frequency Spectrum')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.hist(gray.ravel(), 256, [0, 256], alpha=0.7)
    plt.title(f'Intensity Distribution (σ={contrast:.1f})')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    return {
        'false_positives': false_positive_score,
        'edge_density': edge_density,
        'contrast': contrast,
        'adversarial_score': adversarial_score
    }

if __name__ == "__main__":
    test_image = load_test_image()  # Optionally, pass a different file path as argument
    results = test_adversarial_pattern(test_image)
    if results is not None:
        print("\n=== SUMMARY ===")
        print("Pattern tested successfully!")
        print("Replace the pattern file path in load_test_image() to test custom designs.")
