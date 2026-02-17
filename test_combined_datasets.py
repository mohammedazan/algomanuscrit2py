#!/usr/bin/env python
"""
Test Script for Combined Datasets
=================================
Verify that both datasets load correctly before training.

Usage:
    python test_combined_datasets.py
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.unified_dataset_loader import UnifiedDatasetLoader


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_paths():
    """Test that dataset paths exist."""
    print_section("1. CHECKING DATASET PATHS")
    
    paths_to_check = [
        ("Dataset/dataset.json", "Dataset 1 metadata"),
        ("Dataset/images", "Dataset 1 images directory"),
        ("dataset (1)/dataset/labels.csv", "Dataset 2 metadata"),
        ("dataset (1)/dataset/images", "Dataset 2 images directory"),
    ]
    
    all_exist = True
    for path, description in paths_to_check:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {description:.<45} {path}")
        if not exists:
            all_exist = False
    
    return all_exist


def test_load_datasets():
    """Test loading both datasets."""
    print_section("2. LOADING DATASETS")
    
    try:
        loader = UnifiedDatasetLoader()
        df = loader.load_all_datasets()
        
        if df is None or len(df) == 0:
            print("✗ No datasets loaded!")
            return False
        
        print(f"\n✓ Successfully loaded {len(df):,} samples")
        
        # Show statistics
        print("\n📊 Dataset Statistics:")
        print(f"  Total samples: {len(df):,}")
        print(f"  Unique datasets: {df['dataset'].nunique()}")
        
        for dataset_name in df['dataset'].unique():
            count = len(df[df['dataset'] == dataset_name])
            pct = (count / len(df)) * 100
            print(f"    - {dataset_name}: {count:,} ({pct:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading datasets: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_images_exist():
    """Test that image files exist."""
    print_section("3. VERIFYING IMAGE FILES")
    
    try:
        loader = UnifiedDatasetLoader()
        df = loader.load_all_datasets()
        
        if df is None:
            print("✗ No datasets loaded to test")
            return False
        
        print(f"  Checking {len(df):,} image paths...")
        print(f"  (This may take a minute for large datasets...)")
        
        missing_count = 0
        checked = 0
        
        for idx, row in df.iterrows():
            if checked % 10000 == 0 and checked > 0:
                print(f"    Checked {checked:,} files...")
            
            if not os.path.exists(row['image_path']):
                missing_count += 1
                if missing_count <= 5:  # Show first 5 missing
                    print(f"    ✗ Missing: {row['image_path']}")
            
            checked += 1
        
        valid_count = len(df) - missing_count
        pct = (valid_count / len(df)) * 100 if len(df) > 0 else 0
        
        print(f"\n✓ {valid_count:,}/{len(df):,} images found ({pct:.1f}%)")
        
        if missing_count > 0:
            print(f"⚠ {missing_count:,} images are missing")
            return pct >= 0.95  # OK if 95%+ exist
        
        return True
        
    except Exception as e:
        print(f"✗ Error verifying images: {e}")
        return False


def test_text_labels():
    """Test text labels."""
    print_section("4. ANALYZING TEXT LABELS")
    
    try:
        loader = UnifiedDatasetLoader()
        df = loader.load_all_datasets()
        
        if df is None:
            print("✗ No datasets loaded to test")
            return False
        
        # Remove empty labels
        df_valid = df[df['text'].str.len() > 0].copy()
        empty_count = len(df) - len(df_valid)
        
        print(f"  Total labels: {len(df):,}")
        print(f"  Non-empty labels: {len(df_valid):,}")
        if empty_count > 0:
            print(f"  ⚠ Empty labels: {empty_count:,}")
        
        # Analyze text lengths
        text_lengths = df_valid['text'].str.len()
        
        print(f"\n📊 Text Length Statistics:")
        print(f"  Min: {text_lengths.min()} characters")
        print(f"  Max: {text_lengths.max()} characters")
        print(f"  Mean: {text_lengths.mean():.1f} characters")
        print(f"  Median: {text_lengths.median():.1f} characters")
        
        # Check for texts longer than model can handle
        max_length = 100  # From training config
        too_long = len(df_valid[text_lengths > max_length])
        if too_long > 0:
            pct = (too_long / len(df_valid)) * 100
            print(f"\n⚠ Texts longer than {max_length}: {too_long:,} ({pct:.1f}%)")
            print(f"  These will be truncated during training")
        
        # Show sample texts
        print(f"\n📝 Sample Text Labels:")
        for i in range(min(3, len(df_valid))):
            text = df_valid.iloc[i]['text']
            preview = text[:60] + "..." if len(text) > 60 else text
            print(f"  [{i+1}] {preview}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error analyzing labels: {e}")
        return False


def test_data_split():
    """Test train/validation split."""
    print_section("5. TESTING TRAIN/VALIDATION SPLIT")
    
    try:
        loader = UnifiedDatasetLoader()
        df = loader.load_all_datasets()
        
        if df is None:
            print("✗ No datasets loaded to test")
            return False
        
        train_df, val_df = loader.get_split(train_ratio=0.85)
        
        if train_df is None or val_df is None:
            return False
        
        print(f"\n✓ Split successful")
        print(f"  Train set: {len(train_df):,} samples (85%)")
        print(f"  Val set: {len(val_df):,} samples (15%)")
        print(f"  Total: {len(train_df) + len(val_df):,} samples")
        
        # Verify no overlap
        train_indices = set(range(len(df))[:len(train_df)])
        val_indices = set(range(len(df))[len(train_df):])
        overlap = train_indices & val_indices
        
        if overlap:
            print(f"✗ Overlap detected: {len(overlap)} samples")
            return False
        
        print(f"✓ No overlap between train and validation")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing data split: {e}")
        return False


def generate_summary():
    """Generate test summary."""
    print_section("TEST SUMMARY")
    
    tests = [
        ("Dataset Paths", test_paths),
        ("Load Datasets", test_load_datasets),
        ("Image Files Exist", test_images_exist),
        ("Text Labels", test_text_labels),
        ("Train/Val Split", test_data_split),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_section("RESULTS")
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:10} {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    # Final recommendation
    print_section("NEXT STEPS")
    
    if passed == total:
        print("✓ All tests passed! Ready to train.")
        print("\nRun:")
        print("  python src/ocr/train_combined.py")
        return True
    else:
        print("✗ Some tests failed. Please fix issues before training.")
        print("\nCommon issues:")
        print("  1. Dataset paths don't exist - check folder structure")
        print("  2. Images missing - verify download/extraction")
        print("  3. Labels empty - check CSV/JSON formatting")
        return False


def main():
    """Main test runner."""
    print("\n" + "█"*70)
    print("  COMBINED DATASETS TEST SUITE")
    print("█"*70)
    
    try:
        success = generate_summary()
        
        if success:
            print("\n🎉 All checks passed!")
            return 0
        else:
            print("\n⚠️  Please resolve issues before training")
            return 1
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
