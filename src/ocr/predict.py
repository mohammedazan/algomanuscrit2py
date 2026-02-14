"""
OCR Prediction Module - Handwritten Algorithm Recognition
=========================================================
This module performs OCR inference on handwritten algorithm images
using the trained Lightweight CRNN model.

Pipeline:
1. Load trained CRNN model
2. Preprocess input image
3. Run model inference
4. Decode CTC output to text
5. Return recognized algorithm text

Input: Handwritten algorithm image
Output: Recognized text string

Author: Deep Learning Project Team
Date: 2026-02-10
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.model import build_lightweight_crnn, get_character_set
from preprocessing.image_preprocess import PreprocessConfig, preprocess_image


# ============================================================================
# CONFIGURATION
# ============================================================================

class PredictionConfig:
    """
    Configuration for OCR prediction.
    """
    def __init__(self):
        # Model paths
        self.model_weights_path = "checkpoints/best_model.weights.h5"
        self.fallback_weights_path = "checkpoints/final_inference_model.weights.h5"
        
        # Model parameters
        self.input_shape = (128, 512, 1)
        
        # Preprocessing
        self.preprocess_mode = "robust"  # Use robust mode for better quality


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_trained_model(config):
    """
    Load the trained CRNN model with weights.
    
    Args:
        config: PredictionConfig object
        
    Returns:
        tuple: (model, char_to_num, num_to_char)
    """
    print("\n🔨 Loading trained CRNN model...")
    
    # Get character set
    chars, char_to_num, num_to_char = get_character_set()
    num_classes = len(chars) + 1  # +1 for CTC blank
    
    # Build model architecture
    model = build_lightweight_crnn(config.input_shape, num_classes)
    
    # Try to load weights
    weights_path = None
    if os.path.exists(config.model_weights_path):
        weights_path = config.model_weights_path
    elif os.path.exists(config.fallback_weights_path):
        weights_path = config.fallback_weights_path
        print(f"   ⚠ Best model not found, using fallback: {config.fallback_weights_path}")
    else:
        raise FileNotFoundError(
            f"Model weights not found!\n"
            f"   Tried: {config.model_weights_path}\n"
            f"   Tried: {config.fallback_weights_path}\n"
            f"   Please train the model first using: python src/ocr/train.py"
        )
    
    # Load weights
    model.load_weights(weights_path)
    print(f"✓ Model loaded from: {weights_path}")
    print(f"   Character set size: {num_classes}")
    
    return model, char_to_num, num_to_char


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def load_and_preprocess_image(image_path, config):
    """
    Load and preprocess image for OCR inference.
    
    Args:
        image_path: Path to input image
        config: PredictionConfig object
        
    Returns:
        numpy.ndarray: Preprocessed image of shape (1, 128, 512, 1)
    """
    print(f"\n📷 Loading image: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    print(f"   Original shape: {image.shape}")
    
    # Preprocess using the preprocessing module
    preprocess_config = PreprocessConfig(mode=config.preprocess_mode)
    processed_image = preprocess_image(image, preprocess_config)
    
    print(f"   Preprocessed shape: {processed_image.shape}")
    print(f"   Preprocessing mode: {config.preprocess_mode}")
    
    # Normalize to [0, 1] and add batch dimension
    processed_image = processed_image.astype(np.float32) / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    print("   Before expand:", processed_image.shape)
    processed_image = np.expand_dims(processed_image, axis=-1)  # Add channel dimension
    
    print(f"   Final shape for model: {processed_image.shape}")
    
    return processed_image


# ============================================================================
# CTC DECODING
# ============================================================================

def decode_ctc_predictions(predictions, num_to_char):
    """
    Decode CTC predictions to text.
    
    CTC Decoding Process:
    1. Get the most likely character at each time step (greedy decoding)
    2. Remove consecutive duplicate characters
    3. Remove blank tokens (CTC blank label)
    4. Convert indices to characters
    
    Args:
        predictions: Model output of shape (batch, time_steps, num_classes)
        num_to_char: Dictionary mapping indices to characters
        
    Returns:
        str: Decoded text string
    """
    # Get the most likely character index at each time step
    # Shape: (batch, time_steps)
    predicted_indices = np.argmax(predictions, axis=-1)
    
    # Process first (and only) item in batch
    indices = predicted_indices[0]
    
    # CTC decoding: remove blanks and consecutive duplicates
    decoded_chars = []
    previous_idx = -1
    
    for idx in indices:
        # Skip CTC blank (index = num_classes - 1 = len(chars))
        # Also skip consecutive duplicates
        if idx != previous_idx and idx < len(num_to_char):
            if idx in num_to_char:
                decoded_chars.append(num_to_char[idx])
        previous_idx = idx
    
    # Join characters to form text
    decoded_text = ''.join(decoded_chars)
    
    return decoded_text


def decode_ctc_beam_search(predictions, num_to_char, beam_width=10):
    """
    Decode CTC predictions using beam search (more accurate but slower).
    
    This is an alternative to greedy decoding that explores multiple
    possible decode paths.
    
    Args:
        predictions: Model output of shape (batch, time_steps, num_classes)
        num_to_char: Dictionary mapping indices to characters
        beam_width: Number of beams to keep (higher = more accurate but slower)
        
    Returns:
        str: Decoded text string
    """
    # Use TensorFlow's CTC beam search decoder
    input_length = np.array([predictions.shape[1]])  # Time steps
    
    # Decode using beam search
    decoded, _ = tf.nn.ctc_beam_search_decoder(
        inputs=tf.transpose(predictions, perm=[1, 0, 2]),  # (time, batch, classes)
        sequence_length=input_length,
        beam_width=beam_width
    )
    
    # Get the best path (first beam)
    decoded_indices = decoded[0].values.numpy()
    
    # Convert indices to characters
    decoded_chars = [num_to_char[idx] for idx in decoded_indices if idx in num_to_char]
    decoded_text = ''.join(decoded_chars)
    
    return decoded_text


# ============================================================================
# MAIN PREDICTION FUNCTION
# ============================================================================

def predict_ocr(model, num_to_char, image_path, config=None, use_beam_search=False):

    """
    Perform OCR prediction on a handwritten algorithm image.
    
    Complete pipeline:
    1. Load trained model
    2. Preprocess image
    3. Run inference
    4. Decode CTC output
    5. Return recognized text
    
    Args:
        image_path: Path to input image
        config: PredictionConfig object (optional)
        use_beam_search: Whether to use beam search decoding (default: False)
        
    Returns:
        str: Recognized text from the image
    """
    # Use default config if not provided
    if config is None:
        config = PredictionConfig()
    
    print("=" * 80)
    print("OCR PREDICTION - HANDWRITTEN ALGORITHM RECOGNITION")
    print("=" * 80)
    
    # Step 1: Load model
    #model, char_to_num, num_to_char = load_trained_model(config)
    
    # Step 2: Load and preprocess image
    processed_image = load_and_preprocess_image(image_path, config)
    
    # Step 3: Run inference
    print("\n🔮 Running OCR inference...")
    predictions = model.predict(processed_image, verbose=0)
    print(f"   Prediction shape: {predictions.shape}")
    print(f"   Time steps: {predictions.shape[1]}")
    print(f"   Character probabilities per step: {predictions.shape[2]}")
    
    # Step 4: Decode CTC output
    print("\n📝 Decoding CTC output...")
    
    if use_beam_search:
        print("   Using beam search decoding...")
        decoded_text = decode_ctc_beam_search(predictions, num_to_char, beam_width=10)
    else:
        print("   Using greedy decoding...")
        decoded_text = decode_ctc_predictions(predictions, num_to_char)
    
    # Step 5: Clean and format output
    # Replace escaped newlines with actual newlines for readability
    formatted_text = decoded_text.replace('\\n', '\n')
    
    # Print results
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    print(f"\n📄 Image: {os.path.basename(image_path)}")
    print(f"\n🔤 Raw decoded text:")
    print(f"   {repr(decoded_text)}")
    print(f"\n✨ Formatted OCR output:")
    print("-" * 80)
    print(formatted_text)
    print("-" * 80)
    
    return decoded_text


def predict_batch(image_paths, config=None):
    """
    Perform OCR prediction on multiple images.
    
    Args:
        image_paths: List of image paths
        config: PredictionConfig object (optional)
        
    Returns:
        list: List of recognized texts
    """
    if config is None:
        config = PredictionConfig()
    
    # Load model once
    model, char_to_num, num_to_char = load_trained_model(config)
    
    results = []
    
    print("=" * 80)
    print(f"BATCH OCR PREDICTION - {len(image_paths)} images")
    print("=" * 80)
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
        
        try:
            # Preprocess
            processed_image = load_and_preprocess_image(image_path, config)
            
            # Predict
            predictions = model.predict(processed_image, verbose=0)
            
            # Decode
            decoded_text = decode_ctc_predictions(predictions, num_to_char)
            
            results.append({
                'image_path': image_path,
                'text': decoded_text,
                'success': True
            })
            
            print(f"   ✓ Recognized: {decoded_text[:50]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'image_path': image_path,
                'text': None,
                'success': False,
                'error': str(e)
            })
    
    print("\n" + "=" * 80)
    print(f"✓ Batch prediction completed: {sum(r['success'] for r in results)}/{len(image_paths)} successful")
    print("=" * 80)
    
    return results


# ============================================================================
# DEMONSTRATION
# ============================================================================

def main():
    """
    Demonstration of OCR prediction on a sample image.
    """
    # Sample image path (adjust to your dataset)
    sample_image = "Dataset/images/alg_01.jpeg"
    
    # Alternative paths to try
    alternative_paths = [
        "Dataset/images/alg_01.jpeg",
        "Dataset/images/alg_02.jpeg",
        "Dataset/images/alg_03.jpeg",
    ]
    
    # Find first existing image
    test_image = None
    for path in alternative_paths:
        if os.path.exists(path):
            test_image = path
            break
    
    if test_image is None:
        print("❌ No sample images found!")
        print("   Please ensure images exist in Dataset/images/")
        print("   Or provide an image path as command line argument:")
        print("   python src/ocr/predict.py path/to/image.jpg")
        return
    
    # Run prediction
    try:
        config = PredictionConfig()
        
        model, char_to_num, num_to_char = load_trained_model(config)

        recognized_text = predict_ocr(
            model=model,
            num_to_char=num_to_char,
            image_path=test_image,
            config=config,
            use_beam_search=False
        )

        print("\n" + "=" * 80)
        print("✓ OCR prediction demonstration completed!")
        print("=" * 80)
        
        print("\n💡 Usage:")
        print("   # Predict on single image")
        print("   from src.ocr.predict import predict_ocr")
        print("   text = predict_ocr('path/to/image.jpg')")
        print("")
        print("   # Predict on multiple images")
        print("   from src.ocr.predict import predict_batch")
        print("   results = predict_batch(['img1.jpg', 'img2.jpg'])")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Please train the model first:")
        print("   python src/ocr/train.py")
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    config = PredictionConfig()
    model, char_to_num, num_to_char = load_trained_model(config)

    sample_image = "Dataset/images/alg_01.jpeg"

    if os.path.exists(sample_image):
        predict_ocr(
            model=model,
            num_to_char=num_to_char,
            image_path=sample_image,
            config=config
        )
    else:
        print("No sample image found.")
