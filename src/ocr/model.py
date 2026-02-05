"""
Lightweight CRNN Model for Handwritten OCR
===========================================
Production-grade CRNN optimized for small-to-medium handwritten OCR datasets.

Architecture: Lightweight CNN + BiLSTM + CTC
Target Dataset Size: 100-300 samples
Parameters: ~1-2M (vs ~10M in standard CRNN)

Key Design Principles:
1. Prevent overfitting on small datasets through reduced model capacity
2. Maintain strong representational power for handwritten text
3. Scalable: can grow with dataset expansion to ~300 samples
4. Production-ready: not a toy model, robust architecture

Architecture Overview:
- 3 CNN blocks (32→64→128 filters) with selective pooling
- Feature map height preserved (≥16) for better OCR accuracy
- 2 Bidirectional LSTM layers (128→64 units)
- Dense output with softmax for CTC loss

Input: (batch, 128, 512, 1) - Preprocessed grayscale images
Output: (batch, time_steps, num_characters) - Character probabilities

Author: Deep Learning Project Team
Date: 2026-02-06
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
# LIGHTWEIGHT CRNN MODEL (PRODUCTION-GRADE)
# ============================================================================

def build_lightweight_crnn(input_shape=(128, 512, 1), num_classes=None):
    """
    Build a lightweight CRNN optimized for small-to-medium datasets.
    
    WHY LIGHTWEIGHT?
    ----------------
    With ~102 training samples (growing to ~300), a large model would:
    - Overfit severely (memorize training data)
    - Fail to generalize to new handwriting styles
    - Require excessive training time and data augmentation
    
    This lightweight architecture:
    - Reduces parameter count from ~10M to ~1-2M
    - Maintains sufficient capacity for handwritten OCR
    - Generalizes better on small datasets
    - Trains faster and more stably
    
    SCALABILITY:
    -----------
    - Adequate for 100-300 samples (current scope)
    - Can be scaled up by:
      • Adding more filters (32→64 or 64→96)
      • Using deeper LSTM (128→256 units)
      • Adding dropout layers
    
    Architecture Details:
    --------------------
    CNN Backbone: 3 blocks with [32, 64, 128] filters
    - Selective pooling: pool after 1st and 2nd blocks only
    - Preserves feature map height ≥ 16 for better OCR
    - BatchNorm for training stability
    
    RNN Layers: 2 Bidirectional LSTM
    - Layer 1: 128 units (captures main sequential patterns)
    - Layer 2: 64 units (refines representations)
    - Dropout 0.2 (prevents overfitting)
    
    Output: Dense + Softmax
    - CTC-compatible output layer
    
    Args:
        input_shape: Tuple (height, width, channels), default (128, 512, 1)
        num_classes: Number of character classes (auto-calculated if None)
        
    Returns:
        keras.Model: Lightweight CRNN model
    """
    # Calculate number of classes if not provided
    if num_classes is None:
        chars, _, _ = get_character_set()
        num_classes = len(chars) + 1  # +1 for CTC blank token
    
    # Input layer
    input_data = layers.Input(shape=input_shape, name='image_input')
    
    # ========================================================================
    # PART 1: LIGHTWEIGHT CNN BACKBONE (Feature Extraction)
    # ========================================================================
    # Design: 3 conv blocks with [32, 64, 128] filters
    # Goal: Extract visual features while avoiding excessive downsampling
    
    # Conv Block 1: (128, 512, 1) → (64, 256, 32)
    # -----------------------------------------------
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv1')(input_data)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    # After pooling: height=64, width=256
    
    # Conv Block 2: (64, 256, 32) → (32, 128, 64)
    # -----------------------------------------------
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    # After pooling: height=32, width=128
    
    # Conv Block 3: (32, 128, 64) → (32, 128, 128)
    # -----------------------------------------------
    # NO POOLING here to preserve feature map height
    # Rationale: Maintain sufficient height for better text representation
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)
    # Shape maintained: height=32, width=128
    
    # Optional: Additional conv without pooling for richer features
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Activation('relu', name='relu4')(x)
    # Final CNN output: (batch, 32, 128, 128)
    
    # ========================================================================
    # PART 2: RESHAPE TO SEQUENCE (2D → 1D)
    # ========================================================================
    # Convert feature maps to sequence for RNN processing
    # Strategy: Collapse height dimension into features, keep width as time
    
    # Permute: (batch, height, width, channels) → (batch, width, height, channels)
    x = layers.Permute((2, 1, 3), name='permute')(x)
    # Shape: (batch, 128, 32, 128)
    
    # Reshape: Flatten height and channels
    # (batch, width, height * channels) = (batch, 128, 32*128) = (batch, 128, 4096)
    new_shape = (128, 32 * 128)  # (time_steps, features)
    x = layers.Reshape(target_shape=new_shape, name='reshape')(x)
    
    # Dense layer to reduce feature dimension (4096 → 256)
    # Rationale: 4096 is too large for LSTM, causes overfitting on small data
    x = layers.Dense(256, activation='relu', name='dense_features')(x)
    x = layers.Dropout(0.2, name='dropout_features')(x)
    # Shape: (batch, 128, 256) - 128 time steps, 256-dim features
    
    # ========================================================================
    # PART 3: BIDIRECTIONAL LSTM LAYERS (Sequence Modeling)
    # ========================================================================
    # 2 BiLSTM layers: sufficient for handwritten text patterns
    # More layers would overfit on ~102 samples
    
    # First BiLSTM: 128 units
    # Captures primary sequential dependencies (character sequences)
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2),
        name='bilstm1'
    )(x)
    # Output: (batch, 128, 256) - 256 from bidirectional (128*2)
    
    # Second BiLSTM: 64 units
    # Refines representations, learns higher-order patterns
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2),
        name='bilstm2'
    )(x)
    # Output: (batch, 128, 128) - 128 from bidirectional (64*2)
    
    # ========================================================================
    # PART 4: OUTPUT LAYER (Character Probabilities)
    # ========================================================================
    # Dense layer with softmax for character prediction
    # CTC loss will handle alignment during training
    
    output = layers.Dense(num_classes, activation='softmax', name='char_output')(x)
    # Shape: (batch, 128, num_classes)
    # 128 time steps, each predicting character probabilities
    
    # Create model
    model = Model(inputs=input_data, outputs=output, name='Lightweight_CRNN_OCR')
    
    return model


# ============================================================================
# CTC TRAINING MODEL (Optional)
# ============================================================================

def build_crnn_with_ctc(input_shape=(128, 512, 1), num_classes=None):
    """
    Build lightweight CRNN with CTC loss layer for training.
    
    This version includes the CTC loss computation as part of the model,
    useful for end-to-end training with CTC loss.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of character classes
        
    Returns:
        tuple: (training_model, inference_model)
    """
    # Build base CRNN model
    crnn_model = build_lightweight_crnn(input_shape, num_classes)
    
    # Define additional inputs for CTC loss
    labels = layers.Input(name='label', shape=(None,), dtype='int32')
    input_length = layers.Input(name='input_length', shape=(1,), dtype='int32')
    label_length = layers.Input(name='label_length', shape=(1,), dtype='int32')
    
    # CTC loss layer
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
# MODEL SUMMARY AND ANALYSIS
# ============================================================================

def analyze_model_complexity():
    """
    Analyze and compare model complexity for different architectures.
    """
    print("=" * 80)
    print("MODEL COMPLEXITY ANALYSIS")
    print("=" * 80)
    
    # Get character set info
    chars, _, _ = get_character_set()
    num_classes = len(chars) + 1
    
    print(f"\n📊 Dataset Context:")
    print(f"   Current training samples: ~102")
    print(f"   Expected growth: up to ~300 samples")
    print(f"   Character set size: {num_classes} (including CTC blank)")
    
    print(f"\n🔍 Model Complexity Trade-offs:")
    print(f"\n   Large CRNN (original):")
    print(f"   - Parameters: ~10M")
    print(f"   - CNN: 4 blocks [64, 128, 256, 512]")
    print(f"   - LSTM: 2 layers [256, 128]")
    print(f"   - Risk: SEVERE overfitting on 102 samples")
    print(f"   - Training: Slow, requires heavy regularization")
    
    print(f"\n   Lightweight CRNN (refactored):")
    print(f"   - Parameters: ~1-2M")
    print(f"   - CNN: 3 blocks [32, 64, 128]")
    print(f"   - LSTM: 2 layers [128, 64]")
    print(f"   - Risk: Balanced - sufficient capacity, better generalization")
    print(f"   - Training: Faster, more stable")
    
    print(f"\n💡 Key Design Decisions:")
    print(f"   ✓ Reduced filters (32-128 vs 64-512) - prevents overfitting")
    print(f"   ✓ Selective pooling - preserves vertical features (height=32)")
    print(f"   ✓ Smaller LSTM (128→64) - matches dataset size")
    print(f"   ✓ Dropout 0.2 - regularization without killing capacity")
    print(f"   ✓ BatchNorm - training stability")
    
    print(f"\n🎯 Scalability Plan:")
    print(f"   Current (102 samples): Use lightweight CRNN as-is")
    print(f"   Medium (150-200 samples): Consider 64→96 filters")
    print(f"   Large (250-300 samples): Can increase to [64, 128, 256] filters")
    print(f"   Beyond 300: Transition to standard CRNN architecture")


def get_model_info():
    """
    Print detailed information about the lightweight model.
    """
    print("=" * 80)
    print("LIGHTWEIGHT CRNN FOR HANDWRITTEN OCR")
    print("=" * 80)
    
    chars, char_to_num, num_to_char = get_character_set()
    
    print(f"\n📊 Character Set:")
    print(f"   Total characters: {len(chars)}")
    print(f"   With CTC blank: {len(chars) + 1}")
    print(f"   Sample characters: {chars[:20]}...")
    
    print(f"\n🏗️ Model Architecture:")
    print(f"   Type: Lightweight CRNN (optimized for small datasets)")
    print(f"   Input shape: (128, 512, 1)")
    print(f"   Output shape: (128, {len(chars) + 1})")
    print(f"   Time steps: 128 (sufficient for CTC decoding)")
    
    print(f"\n📐 Architecture Details:")
    print(f"   CNN Blocks: 3 layers")
    print(f"     - Conv1: 32 filters + Pool")
    print(f"     - Conv2: 64 filters + Pool")
    print(f"     - Conv3-4: 128 filters (no pool)")
    print(f"   Feature Maps: (32, 128, 128)")
    print(f"   Sequence Length: 128 time steps")
    print(f"   BiLSTM Layers: 2")
    print(f"     - BiLSTM1: 128 units × 2 directions = 256")
    print(f"     - BiLSTM2: 64 units × 2 directions = 128")
    print(f"   Output: Dense({len(chars) + 1}, softmax)")
    
    return chars, char_to_num, num_to_char


def main():
    """
    Demonstration of lightweight CRNN model creation and analysis.
    """
    print("=" * 80)
    print("LIGHTWEIGHT CRNN - OPTIMIZED FOR SMALL DATASETS")
    print("=" * 80)
    
    # Display model analysis
    analyze_model_complexity()
    
    # Get character set info
    chars, char_to_num, num_to_char = get_model_info()
    
    # Build model
    print("\n🔨 Building Lightweight CRNN model...")
    model = build_lightweight_crnn(input_shape=(128, 512, 1))
    
    print(f"\n✓ Model created successfully!")
    print(f"   Model name: {model.name}")
    print(f"   Total layers: {len(model.layers)}")
    print(f"   Trainable parameters: {model.count_params():,}")
    
    # Display model summary
    print("\n" + "=" * 80)
    print("DETAILED MODEL SUMMARY")
    print("=" * 80)
    model.summary()
    
    # Calculate parameter efficiency
    params = model.count_params()
    params_million = params / 1_000_000
    
    print("\n" + "=" * 80)
    print("ARCHITECTURAL TRADE-OFFS SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Advantages of Lightweight Architecture:")
    print(f"   • Parameters: ~{params_million:.2f}M (vs ~10M) - 80-85% reduction")
    print(f"   • Better generalization on small datasets (102-300 samples)")
    print(f"   • Faster training convergence")
    print(f"   • Lower risk of overfitting")
    print(f"   • Reduced computational requirements")
    print(f"   • Maintains production-grade architecture (not a toy model)")
    
    print(f"\n⚠️ Limitations (Trade-offs):")
    print(f"   • Slightly lower capacity vs large CRNN")
    print(f"   • May plateau earlier with very large datasets (>1000 samples)")
    print(f"   • Requires careful hyperparameter tuning")
    
    print(f"\n🎯 Recommended Use Cases:")
    print(f"   ✓ Current dataset: ~102 samples - PERFECT FIT")
    print(f"   ✓ Growing to ~300 samples - Still optimal")
    print(f"   ✓ Handwritten algorithm OCR - Ideal complexity")
    print(f"   ⚠ If dataset grows >500 samples - Consider scaling up")
    
    print("\n" + "=" * 80)
    print("✓ Model architecture demonstration completed!")
    print("=" * 80)
    
    print("\n💡 Next Steps:")
    print("   1. Implement training pipeline with data augmentation")
    print("   2. Use learning rate scheduling and early stopping")
    print("   3. Monitor validation accuracy to detect overfitting")
    print("   4. Adjust model capacity based on dataset growth")


if __name__ == "__main__":
    main()
