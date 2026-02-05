"""
CRNN Model Architecture for Handwritten OCR
===========================================
This module defines a Convolutional Recurrent Neural Network (CRNN) for 
handwritten algorithm text recognition.

Architecture Overview:
1. CNN Layers - Extract visual features from input images
2. Reshape Layer - Convert 2D feature maps to sequence format
3. Bidirectional LSTM - Model sequential dependencies in text
4. Dense Layer - Output character probabilities for CTC loss

Input: (batch, 128, 512, 1) - Preprocessed grayscale images
Output: (batch, time_steps, num_characters) - Character probability sequences

Reference: 
    "An End-to-End Trainable Neural Network for Image-based Sequence 
    Recognition and Its Application to Scene Text Recognition" (Shi et al., 2015)

Author: Deep Learning Project Team
Date: 2026-02-05
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model


# ============================================================================
# CHARACTER SET CONFIGURATION
# ============================================================================

def get_character_set():
    """
    Define the character set for French algorithm text recognition.
    
    Includes:
    - Lowercase letters (a-z)
    - Uppercase letters (A-Z)
    - Digits (0-9)
    - Common symbols and punctuation
    - French accented characters
    - Blank character for CTC (automatically added by TensorFlow)
    
    Returns:
        list: List of characters
        dict: Character to index mapping
        dict: Index to character mapping
    """
    # Basic alphanumeric
    chars = list('abcdefghijklmnopqrstuvwxyz')
    chars += list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    chars += list('0123456789')
    
    # French accented characters
    chars += list('àâäæçéèêëïîôùûüÿœ')
    chars += list('ÀÂÄÆÇÉÈÊËÏÎÔÙÛÜŸŒ')
    
    # Common symbols in algorithms
    chars += [' ', '(', ')', '[', ']', '{', '}', 
              '+', '-', '*', '/', '=', '<', '>', 
              ':', ';', ',', '.', '!', '?', '"', "'",
              '\\', '_']
    
    # Create mappings
    char_to_num = {char: idx for idx, char in enumerate(chars)}
    num_to_char = {idx: char for idx, char in enumerate(chars)}
    
    return chars, char_to_num, num_to_char


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_crnn_model(input_shape=(128, 512, 1), num_classes=None):
    """
    Build CRNN model for handwritten text recognition.
    
    Architecture:
    - Convolutional blocks (feature extraction)
    - Reshape to sequence (2D → 1D sequence)
    - Bidirectional LSTM layers (sequence modeling)
    - Dense output layer (character probabilities)
    
    Args:
        input_shape: Tuple (height, width, channels), default (128, 512, 1)
        num_classes: Number of character classes (auto-calculated if None)
        
    Returns:
        keras.Model: Compiled CRNN model
    """
    # Calculate number of classes if not provided
    if num_classes is None:
        chars, _, _ = get_character_set()
        num_classes = len(chars) + 1  # +1 for CTC blank token
    
    # Input layer
    input_data = layers.Input(shape=input_shape, name='image_input')
    
    # ========================================================================
    # PART 1: CONVOLUTIONAL LAYERS (Feature Extraction)
    # ========================================================================
    # Purpose: Extract visual features from the input image
    # Each conv block reduces spatial dimensions while increasing feature depth
    
    # Conv Block 1: (128, 512, 1) → (64, 256, 64)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                     name='conv1')(input_data)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    
    # Conv Block 2: (64, 256, 64) → (32, 128, 128)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', 
                     name='conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    
    # Conv Block 3: (32, 128, 128) → (16, 64, 256)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', 
                     name='conv3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    
    # Conv Block 4: (16, 64, 256) → (8, 32, 512)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', 
                     name='conv4')(x)
    x = layers.MaxPooling2D((2, 2), name='pool4')(x)
    
    # Optional: Batch Normalization for training stability
    x = layers.BatchNormalization(name='bn_conv')(x)
    
    # ========================================================================
    # PART 2: RESHAPE TO SEQUENCE
    # ========================================================================
    # Purpose: Convert 2D feature maps to 1D sequence for RNN processing
    # Shape: (batch, height, width, features) → (batch, time_steps, features)
    
    # Get shape after convolutions
    # Expected: (batch, 8, 32, 512)
    # We'll collapse height into features: (batch, 32, 8*512)
    
    # Permute to (batch, width, height, features)
    x = layers.Permute((2, 1, 3), name='permute')(x)
    
    # Reshape: flatten height and features into single feature vector
    # (batch, width, height * features) = (batch, 32, 4096)
    new_shape = (32, 8 * 512)  # (time_steps, features)
    x = layers.Reshape(target_shape=new_shape, name='reshape')(x)
    
    # Dense layer to reduce feature dimension
    x = layers.Dense(256, activation='relu', name='dense_features')(x)
    
    # ========================================================================
    # PART 3: RECURRENT LAYERS (Sequence Modeling)
    # ========================================================================
    # Purpose: Model sequential dependencies in the text
    # Bidirectional LSTM reads sequence forward and backward
    
    # First Bidirectional LSTM layer
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.2),
        name='bilstm1'
    )(x)
    
    # Second Bidirectional LSTM layer
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2),
        name='bilstm2'
    )(x)
    
    # ========================================================================
    # PART 4: OUTPUT LAYER (Character Probabilities)
    # ========================================================================
    # Purpose: Convert LSTM output to character probabilities
    # Shape: (batch, time_steps, num_classes)
    
    output = layers.Dense(num_classes, activation='softmax', 
                         name='output')(x)
    
    # Create model
    model = Model(inputs=input_data, outputs=output, name='CRNN_OCR')
    
    return model


def build_crnn_with_ctc(input_shape=(128, 512, 1), num_classes=None):
    """
    Build CRNN model with CTC loss layer for training.
    
    This version includes the CTC loss layer as part of the model,
    which is useful for training with CTC loss.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of character classes
        
    Returns:
        keras.Model: Model with CTC loss layer
    """
    # Build base CRNN model
    crnn_model = build_crnn_model(input_shape, num_classes)
    
    # Define additional inputs for CTC loss
    labels = layers.Input(name='label', shape=(None,), dtype='int32')
    input_length = layers.Input(name='input_length', shape=(1,), dtype='int32')
    label_length = layers.Input(name='label_length', shape=(1,), dtype='int32')
    
    # CTC loss layer
    # Note: In newer TensorFlow versions, use tf.keras.backend.ctc_batch_cost
    ctc_loss = layers.Lambda(
        lambda args: tf.keras.backend.ctc_batch_cost(
            args[0], args[1], args[2], args[3]
        ),
        name='ctc_loss'
    )([labels, crnn_model.output, input_length, label_length])
    
    # Create training model
    training_model = Model(
        inputs=[crnn_model.input, labels, input_length, label_length],
        outputs=ctc_loss,
        name='CRNN_CTC_Training'
    )
    
    return training_model, crnn_model


# ============================================================================
# LIGHTWEIGHT MODEL (for testing)
# ============================================================================

def build_lightweight_crnn(input_shape=(128, 512, 1), num_classes=None):
    """
    Build a lightweight CRNN model for faster training/testing.
    
    Useful for:
    - Quick prototyping
    - Limited computational resources
    - Educational demonstrations
    
    Args:
        input_shape: Input image shape
        num_classes: Number of character classes
        
    Returns:
        keras.Model: Lightweight CRNN model
    """
    if num_classes is None:
        chars, _, _ = get_character_set()
        num_classes = len(chars) + 1
    
    input_data = layers.Input(shape=input_shape, name='image_input')
    
    # Lightweight CNN (fewer filters)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_data)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Reshape to sequence
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape(target_shape=(64, 16 * 128))(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Lightweight LSTM (fewer units)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    
    # Output
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=input_data, outputs=output, name='Lightweight_CRNN')
    
    return model


# ============================================================================
# MODEL SUMMARY AND UTILITIES
# ============================================================================

def get_model_info():
    """
    Print information about the model architecture.
    """
    print("=" * 80)
    print("CRNN MODEL FOR HANDWRITTEN OCR")
    print("=" * 80)
    
    chars, char_to_num, num_to_char = get_character_set()
    
    print(f"\n📊 Character Set:")
    print(f"   Total characters: {len(chars)}")
    print(f"   With CTC blank: {len(chars) + 1}")
    print(f"   Sample characters: {chars[:20]}...")
    
    print(f"\n🏗️ Model Architecture:")
    print(f"   Input shape: (128, 512, 1)")
    print(f"   Output shape: (32, {len(chars) + 1})")
    print(f"   Architecture: CNN + BiLSTM + CTC")
    
    print(f"\n📐 Architecture Details:")
    print(f"   - 4 Convolutional blocks (64, 128, 256, 512 filters)")
    print(f"   - MaxPooling after each conv block")
    print(f"   - Reshape to sequence (32 time steps)")
    print(f"   - 2 Bidirectional LSTM layers (256, 128 units)")
    print(f"   - Dense output layer (softmax)")
    
    return chars, char_to_num, num_to_char


def main():
    """
    Demonstration of model creation and summary.
    """
    print("=" * 80)
    print("CRNN MODEL ARCHITECTURE DEMONSTRATION")
    print("=" * 80)
    
    # Get character set info
    chars, char_to_num, num_to_char = get_model_info()
    
    # Build model
    print("\n🔨 Building CRNN model...")
    model = build_crnn_model(input_shape=(128, 512, 1))
    
    print(f"\n✓ Model created successfully!")
    print(f"   Model name: {model.name}")
    print(f"   Total layers: {len(model.layers)}")
    print(f"   Trainable parameters: {model.count_params():,}")
    
    # Display model summary
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    model.summary()
    
    # Build lightweight model for comparison
    print("\n" + "=" * 80)
    print("LIGHTWEIGHT MODEL (for comparison)")
    print("=" * 80)
    lightweight_model = build_lightweight_crnn()
    print(f"   Trainable parameters: {lightweight_model.count_params():,}")
    
    print("\n" + "=" * 80)
    print("✓ Model architecture demonstration completed!")
    print("=" * 80)
    
    print("\n💡 Next Steps:")
    print("   1. Implement training pipeline (train.py)")
    print("   2. Implement prediction/inference (predict.py)")
    print("   3. Train model on handwritten dataset")
    print("   4. Evaluate model performance")


if __name__ == "__main__":
    main()
