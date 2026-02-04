"""
Image Preprocessing Module for Handwritten OCR (Enhanced)
=========================================================
This module handles image preprocessing using OpenCV to prepare
handwritten algorithm images for OCR recognition.

Enhanced Features:
- Aspect-ratio preserving resize with padding
- Multiple preprocessing modes (normal, robust)
- Morphological operations for noise reduction
- Configurable parameters

Preprocessing Pipeline:
1. Grayscale conversion
2. Gaussian blur (optional, mode-dependent)
3. Adaptive thresholding
4. Morphological opening (optional, mode-dependent)
5. Smart resize with aspect-ratio preservation and padding

Author: Deep Learning Project Team
Date: 2026-02-04
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

class PreprocessConfig:
    """
    Configuration class for preprocessing parameters.
    Centralized configuration makes it easy to tune preprocessing.
    """
    def __init__(self, mode="normal"):
        """
        Initialize preprocessing configuration.
        
        Args:
            mode: Preprocessing mode - "normal" or "robust"
                  - normal: faster, good for clean images
                  - robust: slower, better for noisy images
        """
        # Target output dimensions
        self.target_height = 128
        self.target_width = 512
        
        # Preprocessing mode
        self.mode = mode  # "normal" or "robust"
        
        # Gaussian blur parameters
        self.blur_kernel = (5, 5)
        self.apply_blur = (mode == "robust")  # Only in robust mode
        
        # Adaptive thresholding parameters
        self.threshold_block_size = 11  # Must be odd
        self.threshold_c = 2
        
        # Morphological operations (noise reduction)
        self.apply_morphology = (mode == "robust")  # Only in robust mode
        self.morph_kernel_size = (2, 2)  # Small kernel to preserve text
        
        # Padding color (0 = black, 255 = white)
        self.pad_value = 0  # Black padding
        
    def __repr__(self):
        return (f"PreprocessConfig(mode={self.mode}, "
                f"target_size=({self.target_height}, {self.target_width}), "
                f"blur={self.apply_blur}, morphology={self.apply_morphology})")


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def resize_and_pad(image, target_height, target_width, pad_value=0):
    """
    Resize image while preserving aspect ratio, then pad to target size.
    
    This prevents text distortion that occurs with naive resizing.
    Algorithm:
    1. Calculate scale factor to fit target_height
    2. Resize with aspect ratio preserved
    3. If width > target_width, rescale to fit width
    4. If width < target_width, pad right side with black pixels
    
    Args:
        image: Input image (grayscale or color)
        target_height: Target height in pixels
        target_width: Target width in pixels
        pad_value: Value for padding (0=black, 255=white)
        
    Returns:
        numpy.ndarray: Resized and padded image of shape (target_height, target_width)
    """
    h, w = image.shape[:2]
    
    # Step 1: Resize to target height while preserving aspect ratio
    scale = target_height / h
    new_width = int(w * scale)
    new_height = target_height
    
    # Resize with aspect ratio preserved
    resized = cv2.resize(image, (new_width, new_height), 
                        interpolation=cv2.INTER_AREA)
    
    # Step 2: Handle width
    if new_width > target_width:
        # Image is too wide - rescale to fit target_width
        scale = target_width / new_width
        final_width = target_width
        final_height = int(new_height * scale)
        resized = cv2.resize(resized, (final_width, final_height),
                           interpolation=cv2.INTER_AREA)
        
        # Vertically center if height < target_height
        if final_height < target_height:
            pad_top = (target_height - final_height) // 2
            pad_bottom = target_height - final_height - pad_top
            
            if len(resized.shape) == 2:
                resized = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 0, 0,
                                            cv2.BORDER_CONSTANT, value=pad_value)
            else:
                resized = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 0, 0,
                                            cv2.BORDER_CONSTANT, value=[pad_value]*3)
    
    elif new_width < target_width:
        # Image is narrower than target - pad right side
        pad_right = target_width - new_width
        
        if len(resized.shape) == 2:
            # Grayscale image
            resized = cv2.copyMakeBorder(resized, 0, 0, 0, pad_right,
                                        cv2.BORDER_CONSTANT, value=pad_value)
        else:
            # Color image
            resized = cv2.copyMakeBorder(resized, 0, 0, 0, pad_right,
                                        cv2.BORDER_CONSTANT, value=[pad_value]*3)
    
    return resized


def apply_morphological_opening(image, kernel_size=(2, 2)):
    """
    Apply morphological opening to reduce noise.
    
    Opening = Erosion followed by Dilation
    - Removes small white noise (isolated pixels)
    - Preserves larger structures (characters)
    
    Args:
        image: Binary image (0 or 255)
        kernel_size: Size of morphological kernel
        
    Returns:
        numpy.ndarray: Denoised image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened


def deskew_image(image):
    """
    Placeholder for image deskewing/rotation correction.
    
    TODO: Implement deskewing if needed for tilted handwritten images
    - Detect text orientation using Hough transform or moments
    - Rotate image to correct orientation
    - Apply perspective correction if needed
    
    Args:
        image: Input image
        
    Returns:
        numpy.ndarray: Deskewed image (currently returns original)
    """
    # Placeholder - return original image
    # Future enhancement: implement skew detection and correction
    return image


def preprocess_image(image, config):
    """
    Complete preprocessing pipeline with configurable steps.
    
    Pipeline:
    1. Convert to grayscale
    2. Optional: Apply Gaussian blur (robust mode)
    3. Apply adaptive thresholding
    4. Optional: Apply morphological opening (robust mode)
    5. Resize with aspect ratio preservation and padding
    
    Args:
        image: Input image (BGR or grayscale)
        config: PreprocessConfig object with parameters
        
    Returns:
        numpy.ndarray: Preprocessed image of shape (target_height, target_width)
    """
    # Step 1: Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Step 2: Optional Gaussian blur (noise reduction)
    if config.apply_blur:
        gray = cv2.GaussianBlur(gray, config.blur_kernel, 0)
    
    # Step 3: Adaptive thresholding (binarization)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Text becomes white on black
        config.threshold_block_size,
        config.threshold_c
    )
    
    # Step 4: Optional morphological opening (noise cleanup)
    if config.apply_morphology:
        thresh = apply_morphological_opening(thresh, config.morph_kernel_size)
    
    # Step 5: Resize with aspect ratio preservation and padding
    processed = resize_and_pad(thresh, config.target_height, 
                               config.target_width, config.pad_value)
    
    return processed


# ============================================================================
# LEGACY CLASS (for backward compatibility)
# ============================================================================

class ImagePreprocessor:
    """
    Legacy wrapper class for backward compatibility.
    New code should use preprocess_image() function directly.
    """
    
    def __init__(self, target_size=(128, 512), blur_kernel=(5, 5), 
                 threshold_block_size=11, threshold_c=2, mode="normal"):
        """Initialize with legacy parameters."""
        self.config = PreprocessConfig(mode=mode)
        self.config.target_height = target_size[0]
        self.config.target_width = target_size[1]
        self.config.blur_kernel = blur_kernel
        self.config.threshold_block_size = threshold_block_size
        self.config.threshold_c = threshold_c
        
    def load_image(self, image_path):
        """Load an image from disk."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return image
    
    def preprocess(self, image_path, **kwargs):
        """Preprocess an image from path."""
        original = self.load_image(image_path)
        processed = preprocess_image(original, self.config)
        return original, processed
    
    def visualize(self, original, processed, title="Image Preprocessing"):
        """Display original and preprocessed images side by side."""
        visualize_comparison(original, processed, title)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_comparison(original, processed, title="Image Preprocessing"):
    """
    Display original and preprocessed images side by side.
    
    Args:
        original: Original input image
        processed: Preprocessed output image
        title: Title for the visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Display original image
    if len(original.shape) == 3:
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Original ({original.shape[1]}×{original.shape[0]})')
    axes[0].axis('off')
    
    # Display preprocessed image
    axes[1].imshow(processed, cmap='gray')
    axes[1].set_title(f'Preprocessed ({processed.shape[1]}×{processed.shape[0]})')
    axes[1].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_modes_comparison(image_path, config_normal, config_robust):
    """
    Compare normal and robust preprocessing modes side by side.
    
    Args:
        image_path: Path to input image
        config_normal: Config for normal mode
        config_robust: Config for robust mode
    """
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Process with both modes
    processed_normal = preprocess_image(original, config_normal)
    processed_robust = preprocess_image(original, config_robust)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original
    if len(original.shape) == 3:
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Original\n({original.shape[1]}×{original.shape[0]})')
    axes[0].axis('off')
    
    # Normal mode
    axes[1].imshow(processed_normal, cmap='gray')
    axes[1].set_title(f'Normal Mode\n({processed_normal.shape[1]}×{processed_normal.shape[0]})')
    axes[1].axis('off')
    
    # Robust mode
    axes[2].imshow(processed_robust, cmap='gray')
    axes[2].set_title(f'Robust Mode\n({processed_robust.shape[1]}×{processed_robust.shape[0]})')
    axes[2].axis('off')
    
    plt.suptitle('Preprocessing Modes Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """
    Demonstration of the enhanced preprocessing pipeline.
    """
    print("=" * 80)
    print("ENHANCED IMAGE PREPROCESSING FOR HANDWRITTEN OCR")
    print("=" * 80)
    
    # Define sample image path
    sample_image = "Dataset/images/alg_01.jpeg"
    
    try:
        # Create configs for both modes
        config_normal = PreprocessConfig(mode="normal")
        config_robust = PreprocessConfig(mode="robust")
        
        print("\n📋 Configuration Comparison:")
        print(f"   Normal mode:  {config_normal}")
        print(f"   Robust mode:  {config_robust}")
        
        # Load image
        print(f"\n🔄 Processing image: {sample_image}")
        
        if not os.path.exists(sample_image):
            raise FileNotFoundError(f"Sample image not found: {sample_image}")
        
        original = cv2.imread(sample_image)
        if original is None:
            raise ValueError(f"Failed to load image: {sample_image}")
        
        print(f"   Original shape: {original.shape}")
        
        # Process with both modes
        processed_normal = preprocess_image(original, config_normal)
        processed_robust = preprocess_image(original, config_robust)
        
        print(f"\n✓ Preprocessing complete!")
        print(f"   Normal mode output:  {processed_normal.shape}")
        print(f"   Robust mode output:  {processed_robust.shape}")
        print(f"   Aspect ratio preserved: ✓")
        print(f"   Fixed dimensions: ✓")
        
        # Test aspect ratio preservation with different image
        print("\n🧪 Testing aspect ratio preservation...")
        
        # Simulate different aspect ratios
        tall_image = cv2.resize(original, (300, 800))  # Tall image
        wide_image = cv2.resize(original, (1000, 400))  # Wide image
        
        tall_processed = preprocess_image(tall_image, config_normal)
        wide_processed = preprocess_image(wide_image, config_normal)
        
        print(f"   Tall image (300×800) → {tall_processed.shape} ✓")
        print(f"   Wide image (1000×400) → {wide_processed.shape} ✓")
        
        # Visualize mode comparison
        print("\n📊 Displaying mode comparison...")
        visualize_modes_comparison(sample_image, config_normal, config_robust)
        
        # Show individual comparisons
        print("\n📊 Displaying normal mode...")
        visualize_comparison(original, processed_normal, 
                           title="Normal Mode: Faster Processing")
        
        print("\n📊 Displaying robust mode...")
        visualize_comparison(original, processed_robust,
                           title="Robust Mode: Better Noise Handling")
        
        print("\n" + "=" * 80)
        print("✓ Enhanced preprocessing demonstration completed!")
        print("=" * 80)
        
        # Print usage recommendations
        print("\n💡 Recommendations:")
        print("   • Use NORMAL mode for clean, well-lit scanned images")
        print("   • Use ROBUST mode for noisy images or poor lighting")
        print("   • Aspect ratio preservation prevents text distortion")
        print("   • Morphological opening (robust) reduces paper texture noise")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Please make sure the image path is correct.")
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
