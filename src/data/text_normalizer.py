"""
Text Normalization and Vocabulary Builder
=========================================
This module normalizes dataset text labels and builds a clean character vocabulary
for OCR training.

Goals:
1. Standardize text format (lowercase, remove accents)
2. Reduce vocabulary size (filter allowed characters)
3. Build character-to-index mappings
4. Save vocabulary for model consistency

Operations:
- Text normalization (accents removal, lowercase conversion)
- Character filtering (keep only allowed symbols)
- Vocabulary extraction and sorting
- JSON vocabulary export

Author: Deep Learning Project Team
Date: 2026-02-12
"""

import os
import json
import pandas as pd
import unicodedata
import re
from collections import Counter


# ============================================================================
# TEXT NORMALIZATION FUNCTIONS
# ============================================================================

def remove_accents(text):
    """
    Remove accents from text using Unicode normalization.
    
    Examples:
        é → e
        à → a
        ç → c
        ü → u
    
    Args:
        text: Input text with potential accents
        
    Returns:
        str: Text without accents
    """
    # Normalize to NFD (decomposed form)
    # This separates base characters from combining diacritical marks
    nfd = unicodedata.normalize('NFD', text)
    
    # Filter out combining characters (accents)
    # Category 'Mn' = Mark, Nonspacing (accents, umlauts, etc.)
    without_accents = ''.join(
        char for char in nfd 
        if unicodedata.category(char) != 'Mn'
    )
    
    # Normalize back to NFC (composed form)
    return unicodedata.normalize('NFC', without_accents)


def normalize_text(text, allowed_chars=None):
    """
    Normalize text for OCR training.
    
    Normalization steps:
    1. Remove accents (é → e)
    2. Convert to lowercase
    3. Normalize whitespace
    4. Filter to allowed characters only
    
    Args:
        text: Input text to normalize
        allowed_chars: Set of allowed characters (if None, use default)
        
    Returns:
        str: Normalized text
    """
    if text is None or text == '':
        return ''
    
    # Default allowed characters
    if allowed_chars is None:
        allowed_chars = get_allowed_characters()
    
    # Step 1: Remove accents
    text = remove_accents(text)
    
    # Step 2: Convert to lowercase
    text = text.lower()
    
    # Step 3: Normalize whitespace (replace tabs, multiple spaces)
    # Preserve newlines
    lines = text.split('\n')
    lines = [' '.join(line.split()) for line in lines]
    text = '\n'.join(lines)
    
    # Step 4: Filter to allowed characters only
    normalized = ''.join(char for char in text if char in allowed_chars)
    
    # Step 5: Clean up multiple consecutive spaces
    normalized = re.sub(r' +', ' ', normalized)
    
    # Step 6: Strip leading/trailing whitespace per line
    lines = normalized.split('\n')
    lines = [line.strip() for line in lines]
    normalized = '\n'.join(lines)
    
    return normalized


def get_allowed_characters():
    """
    Define the allowed character set for normalized text.
    
    Allowed characters:
    - Lowercase letters: a-z
    - Digits: 0-9
    - Whitespace: space, newline
    - Operators: + - * / = < > :
    - Delimiters: ( ) , . _ "
    
    Returns:
        set: Set of allowed characters
    """
    chars = set()
    
    # Lowercase letters
    chars.update('abcdefghijklmnopqrstuvwxyz')
    
    # Digits
    chars.update('0123456789')
    
    # Whitespace
    chars.add(' ')
    chars.add('\n')
    
    # Mathematical operators
    chars.update('+-*/=<>:')
    
    # Delimiters and punctuation
    chars.update('(),._"')
    
    return chars


# ============================================================================
# VOCABULARY BUILDING
# ============================================================================

def extract_vocabulary(texts):
    """
    Extract unique characters from a list of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        set: Set of unique characters
    """
    vocab = set()
    
    for text in texts:
        if text:
            vocab.update(text)
    
    return vocab


def build_vocabulary(characters, add_ctc_blank=True):
    """
    Build vocabulary mappings from character set.
    
    Args:
        characters: Set or list of characters
        add_ctc_blank: Whether to add CTC blank token at index 0
        
    Returns:
        tuple: (char_list, char_to_index, index_to_char)
    """
    # Sort characters for consistency
    char_list = sorted(list(characters))
    
    # Add CTC blank token at the beginning if requested
    if add_ctc_blank:
        # CTC blank will be implicit (not in char_list)
        # Indices: 0 (blank), 1 (first char), 2 (second char), ...
        pass
    
    # Create mappings
    # Start from index 0 (CTC blank is implicitly handled by TensorFlow)
    char_to_index = {char: idx for idx, char in enumerate(char_list)}
    index_to_char = {idx: char for idx, char in enumerate(char_list)}
    
    return char_list, char_to_index, index_to_char


def save_vocabulary(char_list, char_to_index, index_to_char, output_path):
    """
    Save vocabulary to JSON file.
    
    Args:
        char_list: List of characters
        char_to_index: Character to index mapping
        index_to_char: Index to character mapping
        output_path: Path to save JSON file
    """
    # Convert index_to_char keys to strings (JSON requires string keys)
    index_to_char_str = {str(k): v for k, v in index_to_char.items()}
    
    vocab_data = {
        'characters': char_list,
        'char_to_index': char_to_index,
        'index_to_char': index_to_char_str,
        'vocab_size': len(char_list),
        'includes_ctc_blank': True,
        'total_classes': len(char_list) + 1  # +1 for CTC blank
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    print(f"[+] Vocabulary saved to: {output_path}")


def load_vocabulary(vocab_path):
    """
    Load vocabulary from JSON file.
    
    Args:
        vocab_path: Path to vocabulary JSON file
        
    Returns:
        tuple: (char_list, char_to_index, index_to_char)
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    char_list = vocab_data['characters']
    char_to_index = vocab_data['char_to_index']
    
    # Convert string keys back to integers
    index_to_char = {int(k): v for k, v in vocab_data['index_to_char'].items()}
    
    return char_list, char_to_index, index_to_char


# ============================================================================
# DATASET PROCESSING
# ============================================================================

def process_dataset(dataset_path, output_vocab_path):
    """
    Process dataset to normalize text and build vocabulary.
    
    Args:
        dataset_path: Path to dataset.csv
        output_vocab_path: Path to save vocab.json
        
    Returns:
        tuple: (normalized_df, vocab_stats)
    """
    print("=" * 80)
    print("TEXT NORMALIZATION AND VOCABULARY BUILDER")
    print("=" * 80)
    
    # Load dataset
    print(f"\n[*] Loading dataset from: {dataset_path}")
    
    # Try to read with different options to handle quotes and commas in text
    try:
        # First attempt: standard CSV with quote handling
        df = pd.read_csv(dataset_path, quotechar='"', escapechar='\\')
    except Exception as e1:
        print(f"   ⚠ Standard parsing failed, trying with error handling...")
        try:
            # Second attempt: skip bad lines
            df = pd.read_csv(dataset_path, quotechar='"', on_bad_lines='skip')
            print(f"   ⚠ Some malformed lines were skipped")
        except Exception as e2:
            print(f"   ⚠ Both attempts failed, trying line-by-line parsing...")
            # Third attempt: read line by line (most robust but slower)
            try:
                df = pd.read_csv(dataset_path, engine='python', quotechar='"', 
                               escapechar='\\', on_bad_lines='skip')
            except Exception as e3:
                raise Exception(f"Failed to read CSV: {e1}\nAlternative: {e2}\nFinal: {e3}")
    
    print(f"   Total samples: {len(df)}")
    
    # Get allowed characters
    allowed_chars = get_allowed_characters()
    print(f"\n[+] Allowed characters defined: {len(allowed_chars)} characters")
    print(f"   {sorted(allowed_chars)}")
    
    # Extract original vocabulary from text and python_code columns
    print("\n[*] Analyzing original vocabulary...")
    original_texts = []
    
    if 'text' in df.columns:
        original_texts.extend(df['text'].dropna().tolist())
    if 'python_code' in df.columns:
        original_texts.extend(df['python_code'].dropna().tolist())
    
    original_vocab = extract_vocabulary(original_texts)
    print(f"   Original vocabulary size: {len(original_vocab)} characters")
    print(f"   Original characters: {sorted(original_vocab)}")
    
    # Normalize text column
    print("\n[*] Normalizing text labels...")
    normalized_df = df.copy()
    
    if 'text' in df.columns:
        normalized_df['text'] = df['text'].apply(
            lambda x: normalize_text(x, allowed_chars) if pd.notna(x) else ''
        )
        print("   [+] 'text' column normalized")
    
    if 'python_code' in df.columns:
        normalized_df['python_code'] = df['python_code'].apply(
            lambda x: normalize_text(x, allowed_chars) if pd.notna(x) else ''
        )
        print("   [+] 'python_code' column normalized")
    
    # Extract normalized vocabulary
    print("\n[*] Building normalized vocabulary...")
    normalized_texts = []
    
    if 'text' in normalized_df.columns:
        normalized_texts.extend(normalized_df['text'].dropna().tolist())
    if 'python_code' in normalized_df.columns:
        normalized_texts.extend(normalized_df['python_code'].dropna().tolist())
    
    normalized_vocab = extract_vocabulary(normalized_texts)
    print(f"   Normalized vocabulary size: {len(normalized_vocab)} characters")
    print(f"   Normalized characters: {sorted(normalized_vocab)}")
    
    # Build vocabulary mappings
    char_list, char_to_index, index_to_char = build_vocabulary(
        normalized_vocab, 
        add_ctc_blank=True
    )
    
    # Save vocabulary
    print(f"\n[*] Saving vocabulary...")
    save_vocabulary(char_list, char_to_index, index_to_char, output_vocab_path)
    
    # Statistics
    vocab_stats = {
        'original_vocab_size': len(original_vocab),
        'normalized_vocab_size': len(normalized_vocab),
        'reduction': len(original_vocab) - len(normalized_vocab),
        'reduction_percent': ((len(original_vocab) - len(normalized_vocab)) / len(original_vocab) * 100) if len(original_vocab) > 0 else 0,
        'total_classes_with_ctc': len(normalized_vocab) + 1
    }
    
    # Print statistics
    print("\n" + "=" * 80)
    print("VOCABULARY STATISTICS")
    print("=" * 80)
    print(f"\n[STATS] Vocabulary reduction:")
    print(f"   Original size: {vocab_stats['original_vocab_size']} characters")
    print(f"   Normalized size: {vocab_stats['normalized_vocab_size']} characters")
    print(f"   Reduction: {vocab_stats['reduction']} characters ({vocab_stats['reduction_percent']:.1f}%)")
    print(f"   Total classes (with CTC blank): {vocab_stats['total_classes_with_ctc']}")
    
    # Show character differences
    removed_chars = original_vocab - normalized_vocab
    if removed_chars:
        print(f"\n[REMOVED] Characters ({len(removed_chars)}):")
        print(f"   {sorted(removed_chars)}")
    
    # Show example normalization
    print("\n" + "=" * 80)
    print("NORMALIZATION EXAMPLES")
    print("=" * 80)
    
    if 'text' in df.columns and len(df) > 0:
        for i in range(min(3, len(df))):
            original = df['text'].iloc[i]
            normalized = normalized_df['text'].iloc[i]
            
            if original != normalized:
                print(f"\nExample {i+1}:")
                print(f"   Original:   {repr(original[:100])}")
                print(f"   Normalized: {repr(normalized[:100])}")
            else:
                print(f"\nExample {i+1}: (No change)")
                print(f"   Text: {repr(normalized[:100])}")
    
    print("\n" + "=" * 80)
    print("[DONE] NORMALIZATION COMPLETED")
    print("=" * 80)
    
    return normalized_df, vocab_stats


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main entry point for text normalization and vocabulary building.
    """
    # Configuration
    dataset_path = "Dataset/dataset.csv"
    output_vocab_path = "src/ocr/vocab.json"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at {dataset_path}")
        print("\n[INFO] Please ensure the dataset exists:")
        print(f"   {os.path.abspath(dataset_path)}")
        return
    
    try:
        # Process dataset
        normalized_df, stats = process_dataset(dataset_path, output_vocab_path)
        
        print("\n[INFO] Next steps:")
        print("   1. Review the normalized vocabulary in: src/ocr/vocab.json")
        print("   2. Update model.py to use the normalized vocabulary")
        print("   3. Update train.py to use normalized text labels")
        print("   4. Retrain the model with normalized data")
        
        print("\n[INFO] Usage in code:")
        print("   from src.data.text_normalizer import load_vocabulary")
        print("   chars, char_to_idx, idx_to_char = load_vocabulary('src/ocr/vocab.json')")
        
        # Optionally save normalized dataset
        save_normalized = input("\n[?] Save normalized dataset to CSV? (y/n): ").strip().lower()
        if save_normalized == 'y':
            output_csv = "Dataset/dataset_normalized.csv"
            normalized_df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"[+] Normalized dataset saved to: {output_csv}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
