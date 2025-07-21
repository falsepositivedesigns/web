"""
Adversarial Pattern Testing 
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from pathlib import Path


class AdversarialPatternTester:
    """Test adversarial properties of patterns using OpenCV detectors."""
    
    def __init__(self):
        """Initialize cascade classifiers."""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
    
    def load_image(self, filename):
        """Load and validate image file."""
        if not filename or not Path(filename).exists():
            print(f"Error: File not found at {filename}")
            return None
        
        image = cv2.imread(filename)
        if image is None:
            print(f"Error: Unable to load image from {filename}")
            return None
        
        print(f"Loaded image: {filename} - Shape: {image.shape}")
        return image
    
    def detect_features(self, image):
        """Detect faces, eyes, and smiles in the image."""
        faces = self.face_cascade.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )
        eyes = self.eye_cascade.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10)
        )
        smiles = self.smile_cascade.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
        )
        
        return faces, eyes, smiles
    
    def analyze_pattern_complexity(self, image):
        """Analyze image complexity metrics."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_count = np.sum(edges > 0)
        edge_density = edge_count / (gray.shape[0] * gray.shape[1])
        
        # Contrast analysis
        contrast = gray.std()
        
        # Frequency analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        high_freq_energy = np.sum(magnitude_spectrum > np.percentile(magnitude_spectrum, 80))
        
        return {
            'gray': gray,
            'edges': edges,
            'edge_density': edge_density,
            'contrast': contrast,
            'magnitude_spectrum': magnitude_spectrum,
            'high_freq_energy': high_freq_energy
        }
    
    def calculate_scores(self, faces, eyes, smiles, analysis):
        """Calculate adversarial effectiveness scores."""
        false_positive_score = len(faces) + len(eyes) + len(smiles)
        complexity_score = (
            (analysis['edge_density'] * 100) + 
            (analysis['contrast'] / 10) + 
            (analysis['high_freq_energy'] / 1000)
        )
        adversarial_score = false_positive_score + complexity_score
        
        return false_positive_score, complexity_score, adversarial_score
    
    def print_results(self, faces, eyes, smiles, analysis, scores):
        """Print analysis results."""
        false_positive_score, complexity_score, adversarial_score = scores
        
        print("\n=== DETECTION RESULTS ===")
        print(f"False face detections: {len(faces)}")
        print(f"False eye detections: {len(eyes)}")
        print(f"False smile detections: {len(smiles)}")
        
        print("\n=== PATTERN ANALYSIS ===")
        print(f"Edge density: {analysis['edge_density']:.4f}")
        print(f"Contrast level: {analysis['contrast']:.2f}")
        print(f"High frequency energy: {analysis['high_freq_energy']}")
        
        print(f"\n=== ADVERSARIAL EFFECTIVENESS ===")
        print(f"False positive score: {false_positive_score}")
        print(f"Complexity score: {complexity_score:.2f}")
        print(f"Overall adversarial score: {adversarial_score:.2f}")
        
        # Effectiveness assessment
        if false_positive_score > 5:
            print("✅ EXCELLENT: High false positive rate")
        elif false_positive_score > 2:
            print("✅ GOOD: Moderate false positives")
        elif false_positive_score > 0:
            print("⚠️  FAIR: Some false positives")
        else:
            print("❌ POOR: No false positives")
        
        if analysis['edge_density'] > 0.1:
            print("✅ HIGH COMPLEXITY: Dense edge patterns")
        elif analysis['edge_density'] > 0.05:
            print("⚠️  MEDIUM COMPLEXITY: Moderate edge density")
        else:
            print("❌ LOW COMPLEXITY: May need more detail")
    
    def visualize_results(self, image, faces, eyes, analysis, scores):
        """Create visualization of results."""
        false_positive_score = scores[0]
        
        plt.figure(figsize=(15, 10))
        
        # Original pattern
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Pattern')
        plt.axis('off')
        
        # Grayscale
        plt.subplot(2, 3, 2)
        plt.imshow(analysis['gray'], cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')
        
        # Edges
        plt.subplot(2, 3, 3)
        plt.imshow(analysis['edges'], cmap='gray')
        plt.title(f'Edges (Density: {analysis["edge_density"]:.4f})')
        plt.axis('off')
        
        # Detection overlay
        detection_img = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(detection_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(detection_img, 'FACE', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for (x, y, w, h) in eyes:
            cv2.rectangle(detection_img, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv2.putText(detection_img, 'EYE', (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
        plt.title(f'False Detections ({false_positive_score} total)')
        plt.axis('off')
        
        # Frequency spectrum
        plt.subplot(2, 3, 5)
        plt.imshow(analysis['magnitude_spectrum'], cmap='hot')
        plt.title('Frequency Spectrum')
        plt.axis('off')
        
        # Intensity histogram
        plt.subplot(2, 3, 6)
        plt.hist(analysis['gray'].ravel(), 256, [0, 256], alpha=0.7)
        plt.title(f'Intensity Distribution (σ={analysis["contrast"]:.1f})')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def test_pattern(self, image_path):
        """Main testing function."""
        image = self.load_image(image_path)
        if image is None:
            return None
        
        faces, eyes, smiles = self.detect_features(image)
        analysis = self.analyze_pattern_complexity(image)
        scores = self.calculate_scores(faces, eyes, smiles, analysis)
        
        self.print_results(faces, eyes, smiles, analysis, scores)
        self.visualize_results(image, faces, eyes, analysis, scores)
        
        return {
            'false_positives': scores[0],
            'edge_density': analysis['edge_density'],
            'contrast': analysis['contrast'],
            'adversarial_score': scores[2]
        }


class AdversarialAttackGenerator:
    """Generate adversarial examples using various attack methods."""
    
    def __init__(self):
        """Initialize with pre-trained model."""
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
            from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method
            from art.estimators.classification import TensorFlowV2Classifier
            
            self.tf = tf
            self.preprocess_input = preprocess_input
            self.decode_predictions = decode_predictions
            
            # Load model
            self.model = ResNet50(weights='imagenet')
            
            # Create ART classifier wrapper
            self.classifier = TensorFlowV2Classifier(
                model=self.model,
                nb_classes=1000,
                input_shape=(224, 224, 3),
                loss_object=tf.keras.losses.CategoricalCrossentropy()
            )
            
            # Initialize attacks
            self.attacks = {
                'FGSM': FastGradientMethod(estimator=self.classifier, eps=4/255),
                'PGD': ProjectedGradientDescent(
                    estimator=self.classifier, eps=8/255, eps_step=2/255, max_iter=40
                ),
                'CW': CarliniL2Method(classifier=self.classifier, confidence=0.0, max_iter=10)
            }
            
        except ImportError as e:
            print(f"Missing dependencies for adversarial attacks: {e}")
            print("Install with: pip install adversarial-robustness-toolbox tensorflow")
            self.model = None
    
    def load_and_preprocess(self, img_path):
        """Load and preprocess image for model input."""
        from tensorflow.keras.preprocessing import image
        
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = self.preprocess_input(np.expand_dims(x, axis=0))
        return x
    
    def visualize_attack(self, original, adversarial, perturbation):
        """Visualize attack results."""
        plt.figure(figsize=(15, 4))
        titles = ["Original", "Adversarial", "Perturbation (amplified)"]
        images = [original, adversarial, perturbation]
        
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(titles[i])
            plt.imshow(images[i].astype('uint8'))
            plt.axis('off')
        plt.show()
    
    def generate_attacks(self, image_paths):
        """Generate adversarial examples for given images."""
        if self.model is None:
            print("Model not available. Please install required dependencies.")
            return
        
        results = []
        
        for img_path in image_paths:
            if not Path(img_path).exists():
                print(f"Image not found: {img_path}")
                continue
            
            x = self.load_and_preprocess(img_path)
            y_pred = self.model.predict(x, verbose=0)
            true_class = np.argmax(y_pred)
            
            print(f"\n=== {img_path} ===")
            print(f"Original class: {self.decode_predictions(y_pred, top=1)[0][0][1]}")
            
            attack_results = {}
            
            for attack_name, attack in self.attacks.items():
                try:
                    adv_x = attack.generate(x=x)
                    adv_pred = self.model.predict(adv_x, verbose=0)
                    adv_class = np.argmax(adv_pred)
                    adv_label = self.decode_predictions(adv_pred, top=1)[0][0][1]
                    
                    is_successful = adv_class != true_class
                    attack_results[attack_name] = is_successful
                    
                    print(f"[{attack_name}] New class: {adv_label} {'✅' if is_successful else '❌'}")
                    
                    # Visualize
                    from tensorflow.keras.preprocessing import image
                    orig_img = image.array_to_img(x[0])
                    adv_img = image.array_to_img(adv_x[0])
                    diff = np.clip((adv_x[0] - x[0]) * 40 + 127, 0, 255)
                    
                    self.visualize_attack(np.array(orig_img), np.array(adv_img), diff)
                    
                except Exception as e:
                    print(f"[{attack_name}] Failed: {e}")
                    attack_results[attack_name] = False
            
            results.append({
                'image': img_path,
                'original_class': true_class,
                'attacks': attack_results
            })
        
        # Summary statistics
        if results:
            attack_names = list(self.attacks.keys())
            success_rates = {name: 0 for name in attack_names}
            
            for result in results:
                for attack_name in attack_names:
                    if result['attacks'].get(attack_name, False):
                        success_rates[attack_name] += 1
            
            n_images = len(results)
            print(f"\n=== SUMMARY ({n_images} images) ===")
            for attack_name, successes in success_rates.items():
                print(f"{attack_name} Success Rate: {successes/n_images:.2%}")
        
        return results


def main():
    """Main function to run the adversarial pattern testing."""
    # Test adversarial patterns
    tester = AdversarialPatternTester()
    
    # Example usage - replace with your image path
    test_image_path = "/content/climate1.png"  # Update this path
    
    if Path(test_image_path).exists():
        print("Testing adversarial pattern...")
        results = tester.test_pattern(test_image_path)
        if results:
            print("\n=== SUMMARY ===")
            print("Pattern tested successfully!")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please update the path or upload an image.")
    
    # Generate adversarial attacks (optional)
    # generator = AdversarialAttackGenerator()
    # generator.generate_attacks([test_image_path])


if __name__ == "__main__":
    main()
