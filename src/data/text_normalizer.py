"""
Text Normalization Module
==========================
Normalizes algorithm text for OCR training.

Handles:
- Whitespace normalization
- Character encoding fixes
- Newline handling
- Special character standardization

Author: Deep Learning Project Team
Date: 2026-02-17
"""

import re
import unicodedata


def normalize_text(text):
    """
    Normalize text for OCR model.
    
    Args:
        text: Input text string or non-string
        
    Returns:
        str: Normalized text
    """
    # Handle non-string inputs
    if not isinstance(text, str):
        return ""
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Handle escaped newlines
    text = text.replace("\\n", "\n")
    
    # Fix common broken newlines in data
    text = text.replace(")n", ")\n")
    text = text.replace("nLire", "\nLire")
    text = text.replace("nAfficher", "\nAfficher")
    text = text.replace("nEcrire", "\nEcrire")
    text = text.replace("nprint", "\nprint")
    text = text.replace("nsi", "\nsi")
    text = text.replace("nalgorithme", "\nalgorithme")
    text = text.replace("npour", "\npour")
    text = text.replace("ntant", "\ntant")
    
    # Standardize multiple spaces to single space (per line)
    lines = text.split('\n')
    lines = [re.sub(r' +', ' ', line) for line in lines]
    text = '\n'.join(lines)
    
    # Remove tabs and replace with spaces
    text = text.replace('\t', ' ')
    
    # Normalize unicode characters (NFD)
    text = unicodedata.normalize('NFKD', text)
    
    # Remove any null characters
    text = text.replace('\x00', '')
    
    return text


def get_character_set():
    """
    Get the set of valid characters for OCR.
    
    Returns:
        list: List of valid characters
    """
    # Base ASCII printable characters
    chars = set()
    
    # Digits
    chars.update('0123456789')
    
    # Lowercase French alphabet
    chars.update('abcdefghijklmnopqrstuvwxyzàâäæéèêëìîïóòôöœùûüçñ')
    
    # Uppercase French alphabet
    chars.update('ABCDEFGHIJKLMNOPQRSTUVWXYZÀÂÄÆÉÈÊËÌÎÏÓÒÔÖŒÙÛÜÇÑ')
    
    # Common punctuation and operators
    chars.update('()[]{},.;:?!\'\"')
    
    # Mathematical and logical operators
    chars.update('+-*/%=<>≤≥≠±₁₂₃₄₅₆₇₈₉₀')
    
    # Special algorithm symbols
    chars.update('←→↑↓⟵⟶∈∉⊂⊃∪∩∧∨¬')
    
    # Whitespace (space, newline, etc)
    chars.add(' ')
    chars.add('\n')
    chars.add('\t')
    
    # Underscore and other common characters
    chars.update('_^~#@&|\\/')
    
    # Add padding token
    chars_list = sorted(list(chars))
    chars_list.append('<pad>')
    
    return chars_list


def text_to_indices(text, char_to_num):
    """
    Convert text to character indices.
    
    Args:
        text: Input text
        char_to_num: Dictionary mapping characters to indices
        
    Returns:
        list: List of character indices
    """
    indices = []
    for char in text:
        if char in char_to_num:
            indices.append(char_to_num[char])
        else:
            # Unknown character - use space as fallback
            if ' ' in char_to_num:
                indices.append(char_to_num[' '])
    
    return indices


def indices_to_text(indices, num_to_char):
    """
    Convert character indices back to text.
    
    Args:
        indices: List of character indices
        num_to_char: Dictionary mapping indices to characters
        
    Returns:
        str: Reconstructed text
    """
    text = ''
    for idx in indices:
        if idx in num_to_char:
            char = num_to_char[idx]
            if char != '<pad>':
                text += char
    
    return text


def remove_pad_tokens(indices):
    """
    Remove padding tokens from indices.
    
    Args:
        indices: List of character indices
        
    Returns:
        list: Filtered indices without padding
    """
    # Assuming padding index is typically 0 or len(char_set)
    # Usually marked as a special token
    filtered = [idx for idx in indices if idx > 0]
    return filtered


if __name__ == "__main__":
    # Test normalization
    test_texts = [
        "Lire(a)nAfficher(a)",
        "pour IDX_0 dans LIST_0 faire\nsi ( LIST_0 [ IDX_0 ] > 50 ) alors",
        "VAR_0  ←   taille ( LIST_0 )",
    ]
    
    print("Text Normalization Tests:")
    print("="*60)
    
    for text in test_texts:
        normalized = normalize_text(text)
        print(f"\nOriginal:\n{repr(text)}")
        print(f"\nNormalized:\n{repr(normalized)}")
        print("-"*60)
    
    # Test character set
    char_set = get_character_set()
    print(f"\n\nCharacter set size: {len(char_set)}")
    print(f"Includes: digits, French letters, operators, whitespace")
