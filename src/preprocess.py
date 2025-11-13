import os
import re
import pickle
import string
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
import pickle
from pathlib import Path
from .config import DATA_DIR, VOCAB_SIZE, RANDOM_SEED
from typing import Tuple

class SimpleSentTok:
    def tokenize(self, s):
        return [s]

class TextPreprocessor:
    
    def __init__(self):
        self.punct_translator = str.maketrans('', '', string.punctuation)
    
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        
        #lowercase
        text = text.lower()
        
        #removing punctuation and special characters
        text = text.translate(self.punct_translator)
        
        #removing extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:

        if not isinstance(text, str):
            text = str(text)
        
        cleaned_text = self.clean_text(text)
        
        # Tokenize using NLTK.
        try:
            tokens = word_tokenize(cleaned_text)
        except LookupError:

            #Use the fallback sentence tokenizer and then TreebankWordTokenizer
            sentences = SimpleSentTok().tokenize(cleaned_text)
            word_tok = TreebankWordTokenizer()
            tokens = []
            for sent in sentences:
                tokens.extend(word_tok.tokenize(sent))
        
        #filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens

class SequenceProcessor:
    
    def __init__(self, vocab_size: int = VOCAB_SIZE):
        """
        Initialize the sequence processor.
        
        Args:
            vocab_size (int): Maximum vocabulary size (default from config)
        """
        self.vocab_size = vocab_size
        self.vocabulary = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = Counter()
        self.preprocessor = TextPreprocessor()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.PAD_IDX = 0
        self.UNK_IDX = 1
    
    def build_vocabulary(self, texts: List[str], vocab_size: int = None) -> Dict[str, int]:
        """
        Args:
            texts (List[str]): List of texts to build vocabulary from
            vocab_size (int, optional): Maximum vocabulary size
            
        Returns:
            Dict[str, int]: Word to index mapping
        """
        if vocab_size is None:
            vocab_size = self.vocab_size
        
        print(f"Building vocabulary with top {vocab_size} words...")
        
        #counting word frequencies
        word_counts = Counter()
        for text in texts:
            tokens = self.preprocessor.tokenize(text)
            word_counts.update(tokens)
        
        # Get top vocab_size - 2 words (reserve space for PAD and UNK tokens)
        most_common_words = word_counts.most_common(vocab_size - 2)
        
        # Build word to index mapping
        word_to_idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX
        }
        
        # Add most frequent words
        for idx, (word, count) in enumerate(most_common_words, start=2):
            word_to_idx[word] = idx
        
        # Build reverse mapping
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        
        # Store vocabulary information
        self.word_counts = word_counts
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.vocabulary = word_to_idx  # For backward compatibility
        
        print(f"Vocabulary built: {len(word_to_idx)} words")
        print(f"Most common words: {[word for word, _ in most_common_words[:10]]}")
        
        return word_to_idx
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Convert texts to sequences of token IDs using the built vocabulary.
        
        Args:
            texts (List[str]): List of texts to convert
            
        Returns:
            List[List[int]]: List of token ID sequences
        """

        sequences = []
        unknown_word_count = 0
        total_tokens = 0
        
        for text in texts:
            tokens = self.preprocessor.tokenize(text)
            total_tokens += len(tokens)
            
            # Convert tokens to IDs, using UNK_IDX for unknown words
            sequence = []
            for token in tokens:
                if token in self.word_to_idx:
                    sequence.append(self.word_to_idx[token])
                else:
                    sequence.append(self.UNK_IDX)
                    unknown_word_count += 1
            
            sequences.append(sequence)
        
        # Print conversion statistics
        if total_tokens > 0:
            unknown_rate = (unknown_word_count / total_tokens) * 100
            print(f"Text-to-sequence conversion completed:")
            print(f"  - Total tokens: {total_tokens:,}")
            print(f"  - Unknown tokens: {unknown_word_count:,} ({unknown_rate:.2f}%)")
            print(f"  - Sequences created: {len(sequences):,}")
        
        return sequences
    
    def pad_sequences(self, sequences: List[List[int]], max_length: int) -> np.ndarray:
        
        padded_sequences = np.full((len(sequences), max_length), self.PAD_IDX, dtype=np.int32)
        
        for i, sequence in enumerate(sequences):
            if len(sequence) > max_length:
                # Truncate if too long
                padded_sequences[i] = sequence[:max_length]
            else:
                # Pad if too short
                padded_sequences[i, :len(sequence)] = sequence
        
        return padded_sequences

class IMDbDataLoader:

    def __init__(self, data_dir: str = DATA_DIR, filename: str = 'IMDB Dataset.csv'):
        self.data_dir = data_dir
        self.filename = filename
        self.filepath = os.path.join(self.data_dir, self.filename)

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize expected columns to 'text' and 'sentiment'"""
        # Common column name mappings
        col_map = {}
        lower_cols = {c.lower(): c for c in df.columns}

        # Map text column
        for candidate in ['review', 'text', 'content']:
            if candidate in lower_cols:
                col_map[lower_cols[candidate]] = 'text'
                break

        # Map sentiment/label column
        for candidate in ['sentiment', 'label', 'sentiment_label', 'polarity']:
            if candidate in lower_cols:
                col_map[lower_cols[candidate]] = 'sentiment'
                break

        df = df.rename(columns=col_map)

        if 'text' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("CSV must contain review text and sentiment label columns")

        # Convert sentiment to binary 0/1 if needed
        if df['sentiment'].dtype == object:
            df['sentiment'] = df['sentiment'].str.strip().str.lower()
            df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1}).astype(int)
        else:
            # Ensure integers
            df['sentiment'] = df['sentiment'].astype(int)

        return df[['text', 'sentiment']]

    def load_data(self) -> Tuple[list, list, list, list]:

        df = pd.read_csv(self.filepath)

        df_shuffled = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        mid = len(df_shuffled) // 2
        train_df = df_shuffled.iloc[:mid].reset_index(drop=True)
        test_df = df_shuffled.iloc[mid:].reset_index(drop=True)

        # Standardize and return lists
        train_df = self._standardize_dataframe(train_df)
        test_df = self._standardize_dataframe(test_df)

        return train_df['text'].astype(str).tolist(), train_df['sentiment'].astype(int).tolist(), test_df['text'].astype(str).tolist(), test_df['sentiment'].astype(int).tolist()