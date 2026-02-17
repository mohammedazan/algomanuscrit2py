"""
Unified Dataset Loader for Multiple Data Sources
================================================
Combines datasets from different formats:
- Dataset 1: JSON format (Dataset/dataset.json)
- Dataset 2: CSV format (dataset (1)/dataset/labels.csv)

This enables training on combined large datasets for better model accuracy.

Author: Deep Learning Project Team  
Date: 2026-02-17
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path


class UnifiedDatasetLoader:
    """
    Loads and combines multiple dataset sources into one unified dataset.
    
    Supports:
    - JSON format with image_path and text columns
    - CSV format with file_name and text columns
    """
    
    def __init__(self, dataset_configs=None, base_dir=None):
        """
        Initialize unified dataset loader.
        
        Args:
            dataset_configs: List of dicts with format:
                {
                    'type': 'json' or 'csv',
                    'data_path': path to dataset file,
                    'images_dir': path to images directory,
                    'name': dataset name for tracking
                }
            base_dir: Base directory for relative paths (default: current dir)
        """
        self.dataset_configs = dataset_configs or self._default_configs()
        self.base_dir = base_dir or os.getcwd()
        self.df = None
        self.total_samples = 0
        self.stats = {}
        
    def _default_configs(self):
        """Define default dataset configurations."""
        return [
            {
                'type': 'json',
                'data_path': 'Dataset/dataset.json',
                'images_dir': 'Dataset/images',
                'name': 'Dataset_JSON'
            },
            {
                'type': 'csv',
                'data_path': 'dataset (1)/dataset/labels.csv',
                'images_dir': 'dataset (1)/dataset/images',
                'name': 'Dataset_CSV_Large'
            }
        ]
    
    def load_json_dataset(self, config):
        """Load JSON format dataset."""
        data_path = os.path.join(self.base_dir, config['data_path'])
        images_dir = os.path.join(self.base_dir, config['images_dir'])
        
        print(f"\n[JSON] Loading dataset: {config['name']}")
        print(f"   Source: {data_path}")
        
        if not os.path.exists(data_path):
            print(f"   [!] File not found, skipping...")
            return None
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            records = []
            for item in data:
                img_path = item.get('image_path', '')
                text = item.get('text', '')
                
                # Resolve full image path
                if img_path.startswith('./'):
                    img_path = img_path[2:]
                full_path = os.path.join(images_dir, os.path.basename(img_path))
                
                # Verify file exists
                if os.path.exists(full_path):
                    records.append({
                        'image_path': full_path,
                        'text': text,
                        'dataset': config['name']
                    })
            
            df = pd.DataFrame(records)
            print(f"   [OK] Loaded {len(df)} samples")
            return df
            
        except Exception as e:
            print(f"   [ERROR] Error loading JSON dataset: {e}")
            return None
    
    def load_csv_dataset(self, config):
        """Load CSV format dataset."""
        data_path = os.path.join(self.base_dir, config['data_path'])
        images_dir = os.path.join(self.base_dir, config['images_dir'])
        
        print(f"\n[CSV] Loading dataset: {config['name']}")
        print(f"   Source: {data_path}")
        
        if not os.path.exists(data_path):
            print(f"   [!] File not found, skipping...")
            return None
        
        try:
            # Load CSV with proper encoding
            df = pd.read_csv(
                data_path,
                sep=',',
                quotechar='"',
                escapechar='\\',
                engine='python',
                encoding='utf-8'
            )
            
            records = []
            valid_count = 0
            invalid_count = 0
            
            for idx, row in df.iterrows():
                file_name = str(row.get('file_name', ''))
                text = str(row.get('text', ''))
                
                # Build full path
                full_path = os.path.join(images_dir, file_name)
                
                # Verify file exists
                if os.path.exists(full_path) and text.strip():
                    records.append({
                        'image_path': full_path,
                        'text': text,
                        'dataset': config['name']
                    })
                    valid_count += 1
                else:
                    invalid_count += 1
                
                # Progress: every 10k samples
                if (idx + 1) % 10000 == 0:
                    print(f"   Processed {idx + 1} rows...")
            
            result_df = pd.DataFrame(records)
            print(f"   [OK] Loaded {valid_count} valid samples")
            if invalid_count > 0:
                print(f"   [!] Skipped {invalid_count} invalid samples")
            
            return result_df
            
        except Exception as e:
            print(f"   [ERROR] Error loading CSV dataset: {e}")
            return None
    
    def load_all_datasets(self):
        """Load and combine all configured datasets."""
        print("\n" + "="*60)
        print("[INFO] LOADING UNIFIED DATASET")
        print("="*60)
        
        datasets = []
        
        for config in self.dataset_configs:
            if config['type'].lower() == 'json':
                df = self.load_json_dataset(config)
            elif config['type'].lower() == 'csv':
                df = self.load_csv_dataset(config)
            else:
                print(f"⚠ Unknown dataset type: {config['type']}")
                continue
            
            if df is not None and len(df) > 0:
                datasets.append(df)
                self.stats[config['name']] = len(df)
        
        # Combine datasets
        if datasets:
            self.df = pd.concat(datasets, ignore_index=True)
            self.total_samples = len(self.df)
            
            print("\n" + "="*60)
            print("[INFO] DATASET STATISTICS")
            print("="*60)
            for name, count in self.stats.items():
                percentage = (count / self.total_samples) * 100 if self.total_samples > 0 else 0
                print(f"  {name}: {count:,} samples ({percentage:.1f}%)")
            print(f"\n  Total Combined: {self.total_samples:,} samples")
            print("="*60)
            
            return self.df
        else:
            print("[ERROR] No datasets loaded successfully!")
            return None
    
    def get_image_paths(self):
        """Get list of image paths."""
        if self.df is None:
            return []
        return self.df['image_path'].tolist()
    
    def get_labels(self):
        """Get list of text labels."""
        if self.df is None:
            return []
        return self.df['text'].tolist()
    
    def get_dataset_df(self):
        """Get the complete dataframe."""
        return self.df
    
    def get_split(self, train_ratio=0.8, random_state=42):
        """
        Split dataset into train/validation sets.
        
        Args:
            train_ratio: Proportion for training (0.8 = 80% train, 20% val)
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (train_df, val_df)
        """
        if self.df is None:
            print("⚠ No dataset loaded")
            return None, None
        
        from sklearn.model_selection import train_test_split
        
        train_df, val_df = train_test_split(
            self.df,
            train_size=train_ratio,
            random_state=random_state
        )
        
        print(f"\n[OK] Dataset split:")
        print(f"  Training: {len(train_df):,} samples ({train_ratio*100:.0f}%)")
        print(f"  Validation: {len(val_df):,} samples ({(1-train_ratio)*100:.0f}%)")
        
        return train_df, val_df
    
    def verify_images(self, sample_size=100):
        """
        Verify that image files exist (sampling for large datasets).
        
        Args:
            sample_size: Number of samples to check
        """
        if self.df is None:
            print("⚠ No dataset loaded")
            return
        
        print(f"\n🔍 Verifying {sample_size} random samples...")
        
        sample_df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)
        missing_count = 0
        
        for idx, row in sample_df.iterrows():
            if not os.path.exists(row['image_path']):
                missing_count += 1
                print(f"   ✗ Missing: {row['image_path']}")
        
        valid_count = len(sample_df) - missing_count
        print(f"\n✓ {valid_count}/{len(sample_df)} samples verified")
        
        if missing_count == 0:
            print("   All sampled files exist!")
        
        return missing_count == 0


def create_unified_loader(base_dir=None):
    """
    Convenience function to create a unified dataset loader.
    
    Args:
        base_dir: Root directory (default: algomanuscrit2py/)
        
    Returns:
        UnifiedDatasetLoader instance
    """
    loader = UnifiedDatasetLoader(base_dir=base_dir)
    loader.load_all_datasets()
    return loader


if __name__ == "__main__":
    # Test the unified loader
    loader = create_unified_loader()
    
    if loader.get_dataset_df() is not None:
        print("\n✓ Datasets loaded successfully!")
        print(f"Available methods:")
        print(f"  - loader.get_image_paths(): Get all image paths")
        print(f"  - loader.get_labels(): Get all text labels")
        print(f"  - loader.get_split(): Split into train/validation")
        print(f"  - loader.verify_images(): Verify image files exist")
