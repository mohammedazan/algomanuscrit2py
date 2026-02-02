"""
Image Preprocessing Module for Handwritten OCR
==============================================
This module handles image preprocessing using OpenCV to prepare
handwritten algorithm images for OCR recognition.

Preprocessing steps:
1. Grayscale conversion - Reduces image complexity
2. Gaussian blur - Reduces noise (optional)
3. Adaptive thresholding - Binarizes text for better OCR
4. Resizing - Normalizes input size for neural network

Author: Deep Learning Project Team
Date: 2026-02-02
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


class ImagePreprocessor:
    """
    Image preprocessing pipeline for handwritten algorithm OCR.
    
    This class provides configurable preprocessing steps to enhance
    handwritten text recognition accuracy.
    """
    
    def __init__(self, target_size=(128, 512), blur_kernel=(5, 5), 
                 threshold_block_size=11, threshold_c=2):
        """
        Initialize the image preprocessor with configurable parameters.
        
        Args:
            target_size: (height, width) tuple for output image size
                        Default: (128, 512) - suitable for text recognition
            blur_kernel: Gaussian blur kernel size (must be odd numbers)
                        Default: (5, 5)
            threshold_block_size: Block size for adaptive thresholding (must be odd)
                                 Default: 11
            threshold_c: Constant subtracted from weighted mean in thresholding
                        Default: 2
        """
        self.target_size = target_size
        self.blur_kernel = blur_kernel
        self.threshold_block_size = threshold_block_size
        self.threshold_c = threshold_c
        
    def load_image(self, image_path):
        """
        Load an image from disk.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy.ndarray: Loaded image in BGR format (OpenCV default)
            
        Raises:
            FileNotFoundError: If image doesn't exist
            ValueError: If image couldn't be loaded
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return image
    
    def convert_to_grayscale(self, image):
        """
        Convert a color image to grayscale.
        
        Why: Reduces computational complexity and removes color information
        that is not relevant for text recognition.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            numpy.ndarray: Grayscale image
        """
        # Check if already grayscale
        if len(image.shape) == 2:
            return image
        
        # Convert BGR to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    
    def apply_blur(self, image, apply=True):
        """
        Apply Gaussian blur to reduce noise.
        
        Why: Smooths the image and reduces noise, which can improve
        thresholding results and OCR accuracy.
        
        Args:
            image: Input grayscale image
            apply: Whether to apply blur (default: True)
            
        Returns:
            numpy.ndarray: Blurred image (or original if apply=False)
        """
        if not apply:
            return image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, self.blur_kernel, 0)
        return blurred
    
    def apply_adaptive_threshold(self, image):
        """
        Apply adaptive thresholding to binarize the image.
        
        Why: Converts grayscale to binary (black/white) which is ideal for OCR.
        Adaptive thresholding works better than global thresholding for
        varying lighting conditions across the image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            numpy.ndarray: Binary image (0 or 255)
        """
        # Apply adaptive thresholding
        # ADAPTIVE_THRESH_GAUSSIAN_C: threshold value is weighted sum of neighborhood
        # THRESH_BINARY_INV: inverse binary (text becomes white on black background)
        thresh = cv2.adaptiveThreshold(
            image,
            255,  # Maximum value
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.threshold_block_size,
            self.threshold_c
        )
        return thresh
    
    def resize_image(self, image):
        """
        Resize image to target dimensions.
        
        Why: Neural networks require fixed input sizes. Resizing ensures
        all images have the same dimensions for batch processing.
        
        Args:
            image: Input image
            
        Returns:
            numpy.ndarray: Resized image
        """
        # Resize with INTER_AREA interpolation (good for shrinking)
        resized = cv2.resize(image, self.target_size[::-1], 
                            interpolation=cv2.INTER_AREA)
        return resized
    
    def preprocess(self, image_path, apply_blur=True, resize=True):
        """
        Complete preprocessing pipeline.
        
        Applies all preprocessing steps in sequence:
        1. Load image
        2. Convert to grayscale
        3. Apply blur (optional)
        4. Apply adaptive thresholding
        5. Resize (optional)
        
        Args:
            image_path: Path to input image
            apply_blur: Whether to apply Gaussian blur (default: True)
            resize: Whether to resize to target size (default: True)
            
        Returns:
            tuple: (original_image, preprocessed_image)
        """
        # Step 1: Load image
        original = self.load_image(image_path)
        
        # Step 2: Convert to grayscale
        gray = self.convert_to_grayscale(original)
        
        # Step 3: Apply blur (if enabled)
        blurred = self.apply_blur(gray, apply=apply_blur)
        
        # Step 4: Apply adaptive thresholding
        thresh = self.apply_adaptive_threshold(blurred)
        
        # Step 5: Resize (if enabled)
        if resize:
            processed = self.resize_image(thresh)
        else:
            processed = thresh
        
        return original, processed
    
    def visualize(self, original, processed, title="Image Preprocessing"):
        """
        Display original and preprocessed images side by side.
        
        Args:
            original: Original input image
            processed: Preprocessed output image
            title: Title for the visualization (default: "Image Preprocessing")
        """
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Display original image
        if len(original.shape) == 3:
            # Convert BGR to RGB for matplotlib
            axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Display preprocessed image
        axes[1].imshow(processed, cmap='gray')
        axes[1].set_title(f'Preprocessed ({processed.shape[1]}x{processed.shape[0]})')
        axes[1].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    """
    Demonstration of the preprocessing pipeline.
    """
    print("=" * 80)
    print("IMAGE PREPROCESSING FOR HANDWRITTEN OCR")
    print("=" * 80)
    
    # Define sample image path
    # Adjust this to point to an actual image from your dataset
    sample_image = "Dataset/images/alg_01.jpeg"
    
    # Alternative: Use absolute path
    # sample_image = "d:/2025-01-30/Bureau/master/S3/RESEAUX DE NEURONES ARTIFICIELS ET DEEP APPRENTISSAGE/PFE antigravity/Dataset/images/alg_01.jpeg"
    
    try:
        # Initialize preprocessor with default parameters
        print("\n📋 Preprocessing Configuration:")
        print("   Target size: (128, 512)")
        print("   Blur kernel: (5, 5)")
        print("   Threshold block size: 11")
        print("   Threshold constant: 2")
        
        preprocessor = ImagePreprocessor(
            target_size=(128, 512),
            blur_kernel=(5, 5),
            threshold_block_size=11,
            threshold_c=2
        )
        
        # Process the image
        print(f"\n🔄 Processing image: {sample_image}")
        original, processed = preprocessor.preprocess(
            sample_image,
            apply_blur=True,
            resize=True
        )
        
        print(f"✓ Preprocessing complete!")
        print(f"   Original shape: {original.shape}")
        print(f"   Processed shape: {processed.shape}")
        print(f"   Processed range: [{processed.min()}, {processed.max()}]")
        
        # Visualize results
        print("\n📊 Displaying results...")
        preprocessor.visualize(original, processed)
        
        # Demonstrate preprocessing without blur
        print("\n🔄 Testing without blur...")
        original2, processed2 = preprocessor.preprocess(
            sample_image,
            apply_blur=False,
            resize=True
        )
        preprocessor.visualize(original2, processed2, 
                              title="Preprocessing (No Blur)")
        
        print("\n" + "=" * 80)
        print("✓ Preprocessing demonstration completed!")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Please make sure the image path is correct.")
        print("   You can modify the 'sample_image' variable in the main() function.")
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
