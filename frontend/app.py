"""
OCR Prediction Module
Used by Streamlit frontend
"""

import os
import numpy as np
import tensorflow as tf
import cv2

from src.ocr.model import build_lightweight_crnn, get_character_set
from src.preprocessing.image_preprocess import PreprocessConfig, preprocess_image
from src.preprocessing.line_segmentation import segment_lines


# ============================================================================
# CONFIGURATION
# ============================================================================

class PredictionConfig:
    def __init__(self):
        self.model_weights_path = "checkpoints/best_model.weights.h5"
        self.fallback_weights_path = "checkpoints/final_inference_model.weights.h5"
        self.input_shape = (128, 512, 1)
        self.preprocess_mode = "robust"


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_trained_model(config):
    chars, char_to_num, num_to_char = get_character_set()
    num_classes = len(chars) + 1

    model = build_lightweight_crnn(config.input_shape, num_classes)

    weights_path = None
    if os.path.exists(config.model_weights_path):
        weights_path = config.model_weights_path
    elif os.path.exists(config.fallback_weights_path):
        weights_path = config.fallback_weights_path
    else:
        raise FileNotFoundError("Model weights not found. Train the model first.")

    model.load_weights(weights_path)

    return model, char_to_num, num_to_char


# ============================================================================
# CTC DECODING
# ============================================================================

def decode_ctc_predictions(predictions, num_to_char):
    predicted_indices = np.argmax(predictions, axis=-1)
    indices = predicted_indices[0]

    decoded_chars = []
    previous_idx = -1

    for idx in indices:
        if idx != previous_idx and idx in num_to_char:
            decoded_chars.append(num_to_char[idx])
        previous_idx = idx

    return ''.join(decoded_chars)


# ============================================================================
# OCR FUNCTION WITH DEBUG VISUALIZATION
# ============================================================================

def predict_ocr_debug(model, num_to_char, image_path, config=None, show_steps=False):
    """
    Enhanced OCR with optional visualization of preprocessing steps.
    
    Args:
        model: Trained CRNN model
        num_to_char: Index to character mapping
        image_path: Path to input image
        config: PredictionConfig
        show_steps: Whether to return preprocessed images for visualization
    
    Returns:
        If show_steps: (text, original, preprocessed, lines)
        Else: text
    """
    if config is None:
        config = PredictionConfig()

    # 1️⃣ Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError("Impossible de charger l'image.")

    # 2️⃣ Preprocessing
    preprocess_config = PreprocessConfig(mode=config.preprocess_mode)
    processed_full = preprocess_image(original_image, preprocess_config)

    # 3️⃣ Line segmentation
    lines = segment_lines(processed_full)

    if len(lines) == 0:
        return ("⚠️ Aucune ligne détectée.", original_image, processed_full, [])

    full_text = ""
    line_results = []

    for idx, line_img in enumerate(lines):
        # Resize to model input
        line_resized = cv2.resize(line_img, (512, 128))

        # Ensure grayscale
        if len(line_resized.shape) == 3:
            line_resized = cv2.cvtColor(line_resized, cv2.COLOR_BGR2GRAY)

        # Normalize
        line_resized = line_resized.astype(np.float32) / 255.0

        # Add batch + channel dims
        line_resized_expanded = np.expand_dims(line_resized, axis=0)
        line_resized_expanded = np.expand_dims(line_resized_expanded, axis=-1)

        # Predict
        predictions = model.predict(line_resized_expanded, verbose=0)

        # Decode
        decoded_text = decode_ctc_predictions(predictions, num_to_char)
        
        # Store for debugging
        line_results.append({
            'image': (line_resized * 255).astype(np.uint8),
            'text': decoded_text,
            'confidence': np.max(predictions)
        })

        full_text += decoded_text + "\n"

    if show_steps:
        return full_text.strip(), original_image, processed_full, line_results
    else:
        return full_text.strip()

# Keep original function for backward compatibility
def predict_ocr(model, num_to_char, image_path, config=None):
    return predict_ocr_debug(model, num_to_char, image_path, config, show_steps=False)


# ============================================================================
# STREAMLIT INTERFACE
# ============================================================================

import streamlit as st
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Handwritten OCR Recognizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📝 Handwritten Algorithm OCR Recognizer")
st.markdown("Recognize handwritten algorithms using Deep Learning")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
preprocess_mode = st.sidebar.radio(
    "Preprocessing Mode",
    options=["normal", "robust"],
    help="Choose 'normal' for clean images, 'robust' for noisy/dark images"
)

show_debug = st.sidebar.checkbox(
    "🔬 Show Debug Information",
    value=False,
    help="Display preprocessing steps and intermediate results"
)

# Load model cache
@st.cache_resource
def get_model():
    config = PredictionConfig()
    model, char_to_num, num_to_char = load_trained_model(config)
    return model, char_to_num, num_to_char

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image of handwritten algorithm",
        type=["jpg", "jpeg", "png", "bmp", "tiff"]
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Save temporarily to process
        temp_path = "temp_image.png"
        image.save(temp_path)
        
        # Process button
        if st.button("🔍 Recognize Text", key="recognize_btn"):
            try:
                with st.spinner("Loading model..."):
                    model, char_to_num, num_to_char = get_model()
                
                with st.spinner("Processing image..."):
                    config = PredictionConfig()
                    config.preprocess_mode = preprocess_mode
                    
                    if show_debug:
                        result_text, original, preprocessed, lines = predict_ocr_debug(
                            model, num_to_char, temp_path, config, show_steps=True
                        )
                        st.session_state.debug_info = {
                            'original': original,
                            'preprocessed': preprocessed,
                            'lines': lines
                        }
                    else:
                        result_text = predict_ocr(model, num_to_char, temp_path, config)
                
                st.success("✅ Recognition complete!")
                
                # Store result in session state
                st.session_state.ocr_result = result_text
                
                # Display in right column
                st.session_state.show_result = True
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
            finally:
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)

# Result column
with col2:
    st.header("📄 Recognition Result")
    
    if "show_result" in st.session_state and st.session_state.show_result:
        if "ocr_result" in st.session_state:
            result = st.session_state.ocr_result
            
            # Display result in text area
            st.text_area(
                "Recognized Text:",
                value=result,
                height=300,
                disabled=True
            )
            
            # Copy to clipboard button
            st.button("📋 Copy to Clipboard", key="copy_btn",
                     help="Click to copy result")
            
            # Download result
            st.download_button(
                label="📥 Download Result",
                data=result,
                file_name="ocr_result.txt",
                mime="text/plain"
            )
    else:
        st.info("👈 Upload an image and click 'Recognize Text' to see results")

# Debug section
if show_debug and "debug_info" in st.session_state:
    st.markdown("---")
    st.header("🔬 Debug Information")
    
    debug = st.session_state.debug_info
    
    # Show preprocessing stages
    debug_col1, debug_col2 = st.columns(2)
    
    with debug_col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(debug['original'], cv2.COLOR_BGR2RGB), use_column_width=True)
    
    with debug_col2:
        st.subheader("After Preprocessing")
        st.image(debug['preprocessed'], cmap='gray', use_column_width=True)
    
    # Show detected lines
    st.subheader(f"📊 Detected {len(debug['lines'])} Lines")
    
    for idx, line_info in enumerate(debug['lines']):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.image(line_info['image'], caption=f"Line {idx + 1}", use_column_width=True)
        
        with col2:
            st.write(f"**Text:** {line_info['text']}")
        
        with col3:
            confidence = line_info.get('confidence', 0)
            st.write(f"**Conf:** {confidence:.2%}" if confidence else "N/A")
    
    # Statistics
    st.subheader("📈 Statistics")
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        st.metric("Lines Detected", len(debug['lines']))
    
    with stats_col2:
        avg_confidence = np.mean([line['confidence'] for line in debug['lines']]) if debug['lines'] else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    with stats_col3:
        preprocessed_mode = preprocess_mode
        st.metric("Preprocessing", preprocessed_mode.capitalize())

# Footer
st.markdown("---")
st.markdown(
    """
    **About this App**
    - Uses Lightweight CRNN deep learning model
    - Recognizes handwritten algorithm text
    - Supports multiple preprocessing modes
    - Dataset: 152 images (small training set)
    
    **Tips for best results:**
    - Use clear, well-lit images
    - Ensure text is legible
    - Try 'robust' mode for noisy images
    - Check debug info to see preprocessing steps
    """
)

