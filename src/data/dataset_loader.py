"""
Dataset Loader and Validator
============================
This module loads and validates the handwritten algorithm dataset.

Responsibilities:
- Load dataset from CSV
- Verify image paths exist
- Validate text labels are not empty
- Provide dataset statistics and sample entries

Author: Deep Learning Project Team
Date: 2026-02-02
"""

import os
import pandas as pd
from pathlib import Path


class DatasetLoader:
    """
    Handles loading and validation of the handwritten algorithm dataset.
    
    This class provides methods to:
    - Load the dataset from CSV
    - Validate image file existence
    - Verify text label completeness
    - Display dataset statistics
    """
    
    def __init__(self, dataset_path: str, base_dir: str = None):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path: Path to the dataset CSV file
            base_dir: Base directory to resolve relative image paths
                     (defaults to the directory containing the CSV)
        """
        self.dataset_path = dataset_path
        
        # If base_dir not provided, use the directory containing the CSV
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.abspath(dataset_path))
        else:
            self.base_dir = base_dir
            
        self.df = None
        self.valid_samples = []
        
    def load_dataset(self):
        """
        Load the dataset from CSV file.
        
        Returns:
            pandas.DataFrame: The loaded dataset
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            Exception: For other CSV reading errors
        """
        try:
            # Check if CSV exists
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(
                    f"Dataset file not found: {self.dataset_path}"
                )
            
            # Load the CSV
            print(f"📁 Loading dataset from: {self.dataset_path}")
            self.df = pd.read_csv(self.dataset_path)
            
            print(f"✓ Dataset loaded successfully!")
            print(f"   Columns: {list(self.df.columns)}")
            
            return self.df
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            raise
    
    def validate_dataset(self):
        """
        Validate each entry in the dataset.
        
        Checks:
        1. Image file exists on disk
        2. Text label is not empty
        
        Returns:
            list: List of valid sample indices
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print("\n🔍 Validating dataset entries...")
        print("-" * 60)
        
        self.valid_samples = []
        missing_images = []
        empty_texts = []
        
        for idx, row in self.df.iterrows():
            image_path = row['image_path']
            text = row['text']
            
            # Resolve the full image path
            # If path starts with './', it's relative to base_dir
            if image_path.startswith('./'):
                full_path = os.path.join(self.base_dir, image_path[2:])
            else:
                full_path = os.path.join(self.base_dir, image_path)
            
            # Validation checks
            image_exists = os.path.exists(full_path)
            text_valid = pd.notna(text) and str(text).strip() != ''
            
            # Record validation results
            if not image_exists:
                missing_images.append((idx, image_path))
            
            if not text_valid:
                empty_texts.append((idx, image_path))
            
            # Sample is valid if both checks pass
            if image_exists and text_valid:
                self.valid_samples.append(idx)
        
        # Print validation summary
        print(f"✓ Validation complete!")
        print(f"   Total samples: {len(self.df)}")
        print(f"   Valid samples: {len(self.valid_samples)}")
        
        if missing_images:
            print(f"   ⚠ Missing images: {len(missing_images)}")
            # Show first 3 missing images
            for idx, path in missing_images[:3]:
                print(f"      - Row {idx}: {path}")
                
        if empty_texts:
            print(f"   ⚠ Empty text labels: {len(empty_texts)}")
            # Show first 3 empty texts
            for idx, path in empty_texts[:3]:
                print(f"      - Row {idx}: {path}")
        
        return self.valid_samples
    
    def show_samples(self, n: int = 3):
        """
        Display sample entries from the dataset.
        
        Args:
            n: Number of samples to display (default: 3)
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print(f"\n📋 Sample Entries (showing {n} samples):")
        print("=" * 80)
        
        # Get the first n valid samples if available, otherwise first n samples
        if self.valid_samples:
            sample_indices = self.valid_samples[:n]
        else:
            sample_indices = range(min(n, len(self.df)))
        
        for i, idx in enumerate(sample_indices, 1):
            row = self.df.iloc[idx]
            
            print(f"\n[Sample {i}]")
            print(f"  ID: {row['id']}")
            print(f"  Image Path: {row['image_path']}")
            print(f"  Category: {row['category']}")
            print(f"  Algorithm Text:")
            
            # Display text with indentation
            text_lines = str(row['text']).split('\\n')
            for line in text_lines:
                print(f"    {line}")
            
            print(f"  Python Code:")
            code_lines = str(row['python_code']).split('\\n')
            for line in code_lines:
                print(f"    {line}")
            print("-" * 80)
    
    def get_statistics(self):
        """
        Get dataset statistics.
        
        Returns:
            dict: Dictionary containing dataset statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        stats = {
            'total_samples': len(self.df),
            'valid_samples': len(self.valid_samples),
            'invalid_samples': len(self.df) - len(self.valid_samples),
            'categories': self.df['category'].unique().tolist(),
            'category_counts': self.df['category'].value_counts().to_dict()
        }
        
        return stats


def main():
    """
    Main function to demonstrate dataset loading and validation.
    """
    print("=" * 80)
    print("HANDWRITTEN ALGORITHM DATASET - LOADER & VALIDATOR")
    print("=" * 80)
    
    # Define dataset path (adjust if needed)
    # Assuming the script is run from project root
    dataset_csv = "Dataset/dataset.csv"
    
    # Alternative: Use absolute path
    # dataset_csv = "d:/2025-01-30/Bureau/master/S3/RESEAUX DE NEURONES ARTIFICIELS ET DEEP APPRENTISSAGE/PFE antigravity/Dataset/dataset.csv"
    
    try:
        # Initialize the loader
        loader = DatasetLoader(dataset_csv)
        
        # Load the dataset
        df = loader.load_dataset()
        
        # Validate the dataset
        valid_samples = loader.validate_dataset()
        
        # Show sample entries
        loader.show_samples(n=3)
        
        # Get and display statistics
        stats = loader.get_statistics()
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Valid samples: {stats['valid_samples']}")
        print(f"   Invalid samples: {stats['invalid_samples']}")
        print(f"   Categories: {stats['categories']}")
        print(f"\n   Category Distribution:")
        for category, count in stats['category_counts'].items():
            print(f"      - {category}: {count} samples")
        
        print("\n" + "=" * 80)
        print("✓ Dataset loading and validation completed successfully!")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("   Please make sure the dataset path is correct.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
