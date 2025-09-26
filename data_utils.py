"""
Data preprocessing utilities for neural machine translation
Implements tokenization and sequence padding matching the reference notebook
"""
import torch
from tensor import Tensor
from collections import Counter, OrderedDict
import re
import pandas as pd
import numpy as np


class Tokenizer:
    """
    Custom tokenizer implementation matching TensorFlow's Tokenizer behavior
    """
    
    def __init__(self, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
                 lower=True, split=' ', char_level=False, oov_token=None):
        self.num_words = num_words
        self.filters = filters
        self.lower = lower
        self.split = split
        self.char_level = char_level
        self.oov_token = oov_token
        
        self.word_index = {}
        self.index_word = {}
        self.word_counts = Counter()
        self.document_count = 0
        self.word_docs = {}
        
        # Reserve index 0 for padding
        # Reserve index 1 for OOV token if specified
        self._index_counter = 1
        if self.oov_token:
            self.word_index[self.oov_token] = 1
            self.index_word[1] = self.oov_token
            self._index_counter = 2
    
    def _preprocess_text(self, text):
        """Preprocess text according to tokenizer settings"""
        if self.lower:
            text = text.lower()
        
        # Remove filters (if not empty string)
        if self.filters:
            # Create translation table to remove filter characters
            translator = str.maketrans('', '', self.filters)
            text = text.translate(translator)
        
        return text
    
    def _text_to_word_sequence(self, text):
        """Convert text to sequence of words"""
        text = self._preprocess_text(text)
        
        if self.char_level:
            return list(text)
        else:
            return text.split(self.split)
    
    def fit_on_texts(self, texts):
        """
        Fit tokenizer on texts
        
        Args:
            texts: List of text strings
        """
        for text in texts:
            self.document_count += 1
            words = self._text_to_word_sequence(text)
            
            # Count words
            for word in words:
                self.word_counts[word] += 1
                if word not in self.word_docs:
                    self.word_docs[word] = 0
                self.word_docs[word] += 1
        
        # Create word index based on frequency
        # Sort by count (descending) and then alphabetically for consistency
        sorted_words = sorted(self.word_counts.items(), key=lambda x: (-x[1], x[0]))
        
        for word, count in sorted_words:
            if word not in self.word_index:
                self.word_index[word] = self._index_counter
                self.index_word[self._index_counter] = word
                self._index_counter += 1
                
                # Stop if we've reached the word limit
                if self.num_words and len(self.word_index) >= self.num_words:
                    break
    
    def texts_to_sequences(self, texts):
        """
        Convert texts to sequences of indices
        
        Args:
            texts: List of text strings
            
        Returns:
            sequences: List of sequences of indices
        """
        sequences = []
        
        for text in texts:
            words = self._text_to_word_sequence(text)
            sequence = []
            
            for word in words:
                if word in self.word_index:
                    sequence.append(self.word_index[word])
                elif self.oov_token:
                    sequence.append(self.word_index[self.oov_token])
                # Skip words not in vocabulary if no OOV token
            
            sequences.append(sequence)
        
        return sequences
    
    def sequences_to_texts(self, sequences):
        """
        Convert sequences of indices back to texts
        
        Args:
            sequences: List of sequences of indices
            
        Returns:
            texts: List of text strings
        """
        texts = []
        
        for sequence in sequences:
            words = []
            for index in sequence:
                if index in self.index_word:
                    words.append(self.index_word[index])
            
            if self.char_level:
                text = ''.join(words)
            else:
                text = self.split.join(words)
            
            texts.append(text)
        
        return texts


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', 
                 truncating='pre', value=0.0):
    """
    Pad sequences to the same length
    Matches TensorFlow's pad_sequences behavior
    
    Args:
        sequences: List of sequences
        maxlen: Maximum length of sequences (if None, use max length in sequences)
        dtype: Data type of output
        padding: 'pre' or 'post' padding
        truncating: 'pre' or 'post' truncating
        value: Padding value
        
    Returns:
        padded_sequences: Tensor of padded sequences
    """
    if not sequences:
        return torch.empty((0, 0))
    
    lengths = [len(seq) for seq in sequences]
    
    if maxlen is None:
        maxlen = max(lengths)
    
    # Convert to numpy array first
    num_samples = len(sequences)
    
    # Create output array
    if dtype == 'int32':
        x = np.full((num_samples, maxlen), value, dtype=np.int32)
    else:
        x = np.full((num_samples, maxlen), value, dtype=np.float32)
    
    for idx, seq in enumerate(sequences):
        if not seq:
            continue
        
        seq = np.asarray(seq, dtype=dtype)
        
        if len(seq) > maxlen:
            # Truncate
            if truncating == 'pre':
                seq = seq[-maxlen:]
            else:  # 'post'
                seq = seq[:maxlen]
        
        # Pad
        if padding == 'pre':
            x[idx, -len(seq):] = seq
        else:  # 'post'
            x[idx, :len(seq)] = seq
    
    return torch.tensor(x)


def load_translation_data(file_path, sample_size=None, random_state=42):
    """
    Load and preprocess translation dataset
    
    Args:
        file_path: Path to CSV file with English and French columns
        sample_size: Number of samples to use (if None, use all)
        random_state: Random seed for shuffling
        
    Returns:
        english: Array of English sentences
        french: Array of French sentences (with SOS and EOS tokens)
    """
    # Load dataset
    dataset = pd.read_csv(file_path)
    
    # Check for required columns
    if 'English words/sentences' not in dataset.columns:
        raise ValueError("Dataset must contain 'English words/sentences' column")
    if 'French words/sentences' not in dataset.columns:
        raise ValueError("Dataset must contain 'French words/sentences' column")
    
    # Remove duplicates
    dataset = dataset.drop_duplicates()
    
    # Remove null values
    dataset = dataset.dropna()
    
    # Shuffle the data
    dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Sample if requested
    if sample_size and sample_size < len(dataset):
        dataset = dataset.head(sample_size)
    
    # Extract sentences
    english = np.array(dataset['English words/sentences'])
    french = np.array(dataset['French words/sentences'])
    
    # Add SOS and EOS tokens to French sentences (target language)
    french = np.array(['sos ' + sent + ' eos' for sent in french])
    
    return english, french


def create_tokenizers(english_texts, french_texts, filters=''):
    """
    Create tokenizers for English and French
    
    Args:
        english_texts: List of English texts
        french_texts: List of French texts
        filters: Characters to filter out (empty string keeps all)
        
    Returns:
        eng_tokenizer: English tokenizer
        fre_tokenizer: French tokenizer
    """
    # English tokenizer
    eng_tokenizer = Tokenizer(filters=filters)
    eng_tokenizer.fit_on_texts(english_texts)
    
    # French tokenizer
    fre_tokenizer = Tokenizer(filters=filters)
    fre_tokenizer.fit_on_texts(french_texts)
    
    return eng_tokenizer, fre_tokenizer


def prepare_data(english, french, test_size=0.2, random_state=42):
    """
    Prepare data for training (tokenization and padding)
    
    Args:
        english: Array of English sentences
        french: Array of French sentences
        test_size: Proportion of data to use for validation
        random_state: Random seed
        
    Returns:
        Dictionary containing all prepared data
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    eng_train, eng_val, fre_train, fre_val = train_test_split(
        english, french, test_size=test_size, random_state=random_state
    )
    
    # Create tokenizers
    eng_tokenizer, fre_tokenizer = create_tokenizers(eng_train, fre_train)
    
    # Convert to sequences
    eng_train_seq = eng_tokenizer.texts_to_sequences(eng_train)
    eng_val_seq = eng_tokenizer.texts_to_sequences(eng_val)
    fre_train_seq = fre_tokenizer.texts_to_sequences(fre_train)
    fre_val_seq = fre_tokenizer.texts_to_sequences(fre_val)
    
    # Calculate max lengths
    max_eng_length = max(len(seq) for seq in eng_train_seq)
    max_fre_length = max(len(seq) for seq in fre_train_seq)
    
    # Pad sequences
    eng_train_pad = pad_sequences(eng_train_seq, maxlen=max_eng_length, padding='post')
    eng_val_pad = pad_sequences(eng_val_seq, maxlen=max_eng_length, padding='post')
    fre_train_pad = pad_sequences(fre_train_seq, maxlen=max_fre_length, padding='post')
    fre_val_pad = pad_sequences(fre_val_seq, maxlen=max_fre_length, padding='post')
    
    # Calculate vocabulary sizes
    eng_vocab_size = len(eng_tokenizer.word_index) + 1  # +1 for padding
    fre_vocab_size = len(fre_tokenizer.word_index) + 1  # +1 for padding
    
    return {
        'eng_train_pad': eng_train_pad,
        'eng_val_pad': eng_val_pad,
        'fre_train_pad': fre_train_pad,
        'fre_val_pad': fre_val_pad,
        'eng_tokenizer': eng_tokenizer,
        'fre_tokenizer': fre_tokenizer,
        'eng_vocab_size': eng_vocab_size,
        'fre_vocab_size': fre_vocab_size,
        'max_eng_length': max_eng_length,
        'max_fre_length': max_fre_length
    }


def create_training_data(fre_padded_sequences):
    """
    Create decoder input and target data from padded French sequences
    
    Args:
        fre_padded_sequences: Padded French sequences with SOS and EOS tokens
        
    Returns:
        decoder_input_data: Input sequences (without EOS)
        decoder_target_data: Target sequences (without SOS)
    """
    # Decoder input: remove EOS token (last token)
    decoder_input_data = fre_padded_sequences[:, :-1]
    
    # Decoder target: remove SOS token (first token)  
    decoder_target_data = fre_padded_sequences[:, 1:]
    
    return decoder_input_data, decoder_target_data


class DataLoader:
    """
    Simple data loader for batching
    """
    
    def __init__(self, encoder_inputs, decoder_inputs, targets, batch_size=32, shuffle=True):
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(encoder_inputs)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        
        self.indices = list(range(self.num_samples))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_batch = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_batch >= self.num_batches:
            # Reset for next epoch
            self.current_batch = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
            raise StopIteration
        
        # Get batch indices
        start_idx = self.current_batch * self.batch_size
        end_idx = min((self.current_batch + 1) * self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Get batch data
        batch_encoder = self.encoder_inputs[batch_indices]
        batch_decoder = self.decoder_inputs[batch_indices]
        batch_targets = self.targets[batch_indices]
        
        self.current_batch += 1
        
        return batch_encoder, batch_decoder, batch_targets
    
    def __len__(self):
        return self.num_batches