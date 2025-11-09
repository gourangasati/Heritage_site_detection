"""
Test script for the trained heritage monument classification model
Tests the model on sample images from the test dataset
"""
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pathlib import Path

# Paths
MODEL_PATH = os.path.join('..', 'model', 'heritage_model.h5')
CLASS_PATH = os.path.join('..', 'model', 'class_indices.json')
TEST_DIR = os.path.join('dataset', 'Indian-monuments', 'images', 'test')

# Load model and class indices
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

with open(CLASS_PATH, 'r') as f:
    class_indices = json.load(f)
    idx_to_label = {v: k for k, v in class_indices.items()}
    label_to_idx = class_indices

print(f"\nClass indices: {class_indices}")
print(f"Number of classes: {len(class_indices)}\n")

def preprocess_image(img_path, target_size=(224, 224)):
    """Preprocess image for prediction"""
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def predict_image(img_path):
    """Predict class for a single image"""
    x = preprocess_image(img_path)
    preds = model.predict(x, verbose=0)[0]
    top_idx = preds.argmax()
    confidence = float(preds[top_idx])
    label = idx_to_label[top_idx]
    
    # Get all predictions
    all_predictions = {idx_to_label[i]: float(preds[i]) for i in range(len(preds))}
    
    return label, confidence, all_predictions

def test_class(class_name, num_samples=5):
    """Test model on samples from a specific class"""
    class_dir = os.path.join(TEST_DIR, class_name)
    
    if not os.path.exists(class_dir):
        print(f"[WARNING] Class directory not found: {class_dir}")
        return None
    
    # Get image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.jfif']:
        image_files.extend(Path(class_dir).glob(ext))
    
    if not image_files:
        print(f"[WARNING] No images found in {class_dir}")
        return None
    
    # Test on first num_samples images
    test_images = image_files[:num_samples]
    correct = 0
    total = len(test_images)
    
    print(f"\n{'='*60}")
    print(f"Testing on class: {class_name}")
    print(f"Expected class index: {label_to_idx.get(class_name, 'N/A')}")
    print(f"{'='*60}")
    
    for img_path in test_images:
        predicted_label, confidence, all_preds = predict_image(str(img_path))
        is_correct = predicted_label == class_name
        
        if is_correct:
            correct += 1
            status = "[CORRECT]"
        else:
            status = "[WRONG]"
        
        print(f"\nImage: {img_path.name}")
        print(f"  Predicted: {predicted_label} (confidence: {confidence:.4f})")
        print(f"  Expected: {class_name}")
        print(f"  Status: {status}")
        print(f"  All predictions:")
        for label, prob in sorted(all_preds.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {label}: {prob:.4f}")
    
    accuracy = (correct / total) * 100
    print(f"\n[RESULTS] for {class_name}:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    return accuracy

def main():
    """Main testing function"""
    print("="*60)
    print("HERITAGE MONUMENT CLASSIFICATION MODEL TEST")
    print("="*60)
    
    # Test on the 3 trained classes
    test_classes = ['tajmahal', 'qutub_minar']
    
    # Find India Gate folder (might be "India_gate" or "Gateway of India")
    test_folders = [f for f in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, f))]
    india_gate_folder = None
    for folder in test_folders:
        if 'india' in folder.lower() and 'gate' in folder.lower():
            india_gate_folder = folder
            break
    
    if india_gate_folder:
        test_classes.append(india_gate_folder)
    
    print(f"\nTesting on {len(test_classes)} classes: {test_classes}\n")
    
    results = {}
    for class_name in test_classes:
        acc = test_class(class_name, num_samples=10)
        if acc is not None:
            results[class_name] = acc
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL TEST SUMMARY")
    print(f"{'='*60}")
    if results:
        avg_accuracy = sum(results.values()) / len(results)
        print(f"\nAverage Accuracy: {avg_accuracy:.2f}%")
        print(f"\nPer-class results:")
        for class_name, acc in results.items():
            print(f"  - {class_name}: {acc:.2f}%")
    else:
        print("No results to display")

if __name__ == "__main__":
    main()

