"""
Training pipeline for neural machine translation
"""
import time
import torch
from tensor import Tensor
from model import create_translation_model
from optim import Adam, SparseCategoricalCrossentropy
from data_utils import DataLoader, create_training_data
import numpy as np


class Trainer:
    """
    Trainer class for the neural machine translation model
    """
    
    def __init__(self, model, optimizer, loss_function, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'teacher_forcing_ratio': []
        }
    
    def compute_accuracy(self, predictions, targets):
        """
        Compute accuracy for sequence prediction
        
        Args:
            predictions: Model predictions (batch_size, seq_len, vocab_size)
            targets: Target sequences (batch_size, seq_len)
            
        Returns:
            accuracy: Accuracy score
        """
        # Get predicted tokens
        pred_tokens = predictions.data.argmax(dim=-1)  # (batch_size, seq_len)
        
        # Convert targets to long tensor
        if isinstance(targets, Tensor):
            target_tokens = targets.data.long()
        else:
            target_tokens = torch.tensor(targets).long()
        
        # Calculate accuracy (ignoring padding tokens - assuming 0 is padding)
        mask = target_tokens != 0  # Non-padding mask
        correct = (pred_tokens == target_tokens) * mask
        accuracy = correct.sum().float() / mask.sum().float()
        
        return accuracy.item()
    
    def get_teacher_forcing_ratio(self, epoch, total_epochs, initial_ratio=1.0, final_ratio=0.3):
        """
        Gradually decrease teacher forcing ratio during training (curriculum learning)
        
        Args:
            epoch: Current epoch (0-based)
            total_epochs: Total number of epochs
            initial_ratio: Starting teacher forcing ratio
            final_ratio: Final teacher forcing ratio
            
        Returns:
            teacher_forcing_ratio: Current ratio for this epoch
        """
        progress = epoch / max(total_epochs - 1, 1)  # Avoid division by zero
        ratio = initial_ratio - (initial_ratio - final_ratio) * progress
        return max(ratio, final_ratio)  # Ensure we don't go below final_ratio
    
    def train_step(self, encoder_inputs, decoder_inputs, targets, teacher_forcing_ratio=1.0):
        """
        Single training step with configurable teacher forcing
        
        Args:
            encoder_inputs: Encoder input sequences
            decoder_inputs: Decoder input sequences  
            targets: Target sequences
            teacher_forcing_ratio: Probability of using teacher forcing (0.0-1.0)
            
        Returns:
            loss: Training loss
            accuracy: Training accuracy
        """
        # Convert to tensors and move to device
        encoder_inputs = Tensor(encoder_inputs.float(), requires_grad=False).to(self.device)
        decoder_inputs = Tensor(decoder_inputs.float(), requires_grad=False).to(self.device)
        targets = Tensor(targets.float(), requires_grad=False).to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with scheduled sampling
        if teacher_forcing_ratio == 1.0:
            # Use standard forward pass for full teacher forcing
            predictions = self.model(encoder_inputs, decoder_inputs)
        else:
            # Use scheduled sampling
            predictions = self.model.forward_with_scheduled_sampling(
                encoder_inputs, decoder_inputs, targets, teacher_forcing_ratio
            )
        
        # Compute loss
        loss = self.loss_function(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        # Compute accuracy
        accuracy = self.compute_accuracy(predictions, targets)
        
        return loss.data.item(), accuracy
    
    def validate_step(self, encoder_inputs, decoder_inputs, targets):
        """
        Single validation step
        
        Args:
            encoder_inputs: Encoder input sequences
            decoder_inputs: Decoder input sequences
            targets: Target sequences
            
        Returns:
            loss: Validation loss
            accuracy: Validation accuracy
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Convert to tensors and move to device
        encoder_inputs = Tensor(encoder_inputs.float(), requires_grad=False).to(self.device)
        decoder_inputs = Tensor(decoder_inputs.float(), requires_grad=False).to(self.device)
        targets = Tensor(targets.float(), requires_grad=False).to(self.device)
        
        # Forward pass (no gradients needed)
        with torch.no_grad():
            predictions = self.model(encoder_inputs, decoder_inputs)
            loss = self.loss_function(predictions, targets)
        
        # Compute accuracy
        accuracy = self.compute_accuracy(predictions, targets)
        
        # Set model back to training mode
        self.model.train()
        
        return loss.data.item(), accuracy
    
    def train_epoch(self, train_loader, teacher_forcing_ratio=1.0):
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            avg_loss: Average training loss
            avg_accuracy: Average training accuracy
        """
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for encoder_inputs, decoder_inputs, targets in train_loader:
            loss, accuracy = self.train_step(encoder_inputs, decoder_inputs, targets, teacher_forcing_ratio)
            
            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, avg_accuracy
    
    def validate_epoch(self, val_loader):
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            avg_loss: Average validation loss
            avg_accuracy: Average validation accuracy
        """
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for encoder_inputs, decoder_inputs, targets in val_loader:
            loss, accuracy = self.validate_step(encoder_inputs, decoder_inputs, targets)
            
            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, avg_accuracy
    
    def fit(self, train_data, val_data, epochs=10, batch_size=64, verbose=True):
        """
        Train the model
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary  
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Print training progress
            
        Returns:
            history: Training history
        """
        # Extract training data
        eng_train = train_data['encoder_inputs']
        dec_train_input = train_data['decoder_inputs'] 
        dec_train_target = train_data['targets']
        
        # Extract validation data
        eng_val = val_data['encoder_inputs']
        dec_val_input = val_data['decoder_inputs']
        dec_val_target = val_data['targets']
        
        # Create data loaders
        train_loader = DataLoader(eng_train, dec_train_input, dec_train_target, 
                                 batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(eng_val, dec_val_input, dec_val_target,
                               batch_size=batch_size, shuffle=False)
        
        if verbose:
            print(f"Training on {len(eng_train)} samples, validating on {len(eng_val)} samples")
            print(f"Batch size: {batch_size}, Epochs: {epochs}")
            print("-" * 60)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Calculate teacher forcing ratio for this epoch
            tf_ratio = self.get_teacher_forcing_ratio(epoch, epochs)
            
            # Train epoch with current teacher forcing ratio
            train_loss, train_acc = self.train_epoch(train_loader, teacher_forcing_ratio=tf_ratio)
            
            # Validate epoch (always use full teacher forcing for validation)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['teacher_forcing_ratio'].append(tf_ratio)
            
            epoch_time = time.time() - start_time
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - "
                      f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
                      f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - "
                      f"tf_ratio: {tf_ratio:.3f}")
        
        return self.history


def create_trainer(eng_vocab_size, fre_vocab_size, embedding_dim=256, 
                  lstm_units=256, learning_rate=0.001, device='cpu'):
    """
    Create trainer with model, optimizer, and loss function
    
    Args:
        eng_vocab_size: English vocabulary size
        fre_vocab_size: French vocabulary size
        embedding_dim: Embedding dimension
        lstm_units: LSTM hidden units
        learning_rate: Learning rate for optimizer
        device: Device to train on
        
    Returns:
        trainer: Configured Trainer instance
    """
    # Create model
    model = create_translation_model(
        eng_vocab_size=eng_vocab_size,
        fre_vocab_size=fre_vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        device=device
    )
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Create loss function
    loss_function = SparseCategoricalCrossentropy(from_logits=True)
    
    # Create trainer
    trainer = Trainer(model, optimizer, loss_function, device)
    
    return trainer


def prepare_training_data(data_dict):
    """
    Prepare training data from preprocessed data dictionary
    
    Args:
        data_dict: Dictionary from prepare_data function
        
    Returns:
        train_data: Training data dictionary
        val_data: Validation data dictionary
    """
    # Create decoder input and target data
    dec_train_input, dec_train_target = create_training_data(data_dict['fre_train_pad'])
    dec_val_input, dec_val_target = create_training_data(data_dict['fre_val_pad'])
    
    train_data = {
        'encoder_inputs': data_dict['eng_train_pad'],
        'decoder_inputs': dec_train_input,
        'targets': dec_train_target
    }
    
    val_data = {
        'encoder_inputs': data_dict['eng_val_pad'],
        'decoder_inputs': dec_val_input,
        'targets': dec_val_target
    }
    
    return train_data, val_data


def train_model(data_file_path, epochs=10, batch_size=64, embedding_dim=256,
               lstm_units=256, learning_rate=0.001, device='cpu', 
               sample_size=None, verbose=True):
    """
    Complete training pipeline
    
    Args:
        data_file_path: Path to training data CSV
        epochs: Number of training epochs
        batch_size: Batch size
        embedding_dim: Embedding dimension  
        lstm_units: LSTM hidden units
        learning_rate: Learning rate
        device: Training device
        sample_size: Number of samples to use (None for all)
        verbose: Print progress
        
    Returns:
        trainer: Trained model trainer
        data_dict: Preprocessed data dictionary
    """
    from data_utils import load_translation_data, prepare_data
    
    if verbose:
        print("Loading and preprocessing data...")
    
    # Load data
    english, french = load_translation_data(data_file_path, sample_size=sample_size)
    
    # Prepare data (tokenization, padding, etc.)
    data_dict = prepare_data(english, french)
    
    if verbose:
        print(f"English vocabulary size: {data_dict['eng_vocab_size']}")
        print(f"French vocabulary size: {data_dict['fre_vocab_size']}")
        print(f"Max English length: {data_dict['max_eng_length']}")
        print(f"Max French length: {data_dict['max_fre_length']}")
    
    # Prepare training data
    train_data, val_data = prepare_training_data(data_dict)
    
    # Create trainer
    trainer = create_trainer(
        eng_vocab_size=data_dict['eng_vocab_size'],
        fre_vocab_size=data_dict['fre_vocab_size'],
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        learning_rate=learning_rate,
        device=device
    )
    
    if verbose:
        print(f"Created model with {sum(p.data.numel() for p in trainer.model.parameters())} parameters")
    
    # Train model
    history = trainer.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    
    return trainer, data_dict


def translate_sentence(trainer, sentence, eng_tokenizer, fre_tokenizer, 
                      max_eng_length, device='cpu', use_beam_search=False, beam_width=3, debug=False):
    """
    Translate a single sentence using trained model
    
    Args:
        trainer: Trained model trainer
        sentence: English sentence to translate
        eng_tokenizer: English tokenizer
        fre_tokenizer: French tokenizer
        max_eng_length: Maximum English sequence length
        device: Device to run on
        use_beam_search: Whether to use beam search (slower but better quality)
        beam_width: Beam width for beam search
        debug: Print debug information
        
    Returns:
        translation: Translated French sentence
    """
    from data_utils import pad_sequences
    
    if debug:
        print(f"Input sentence: '{sentence}'")
    
    # Tokenize and pad input sentence
    sequence = eng_tokenizer.texts_to_sequences([sentence])
    if debug:
        print(f"Tokenized sequence: {sequence}")
    
    padded = pad_sequences(sequence, maxlen=max_eng_length, padding='post')
    if debug:
        print(f"Padded sequence: {padded}")
    
    # Convert to tensor
    encoder_inputs = Tensor(padded.float()).to(device)
    
    # Generate translation
    trainer.model.eval()
    with torch.no_grad():
        # Get start token ID (assuming 'sos' maps to 1)
        start_token_id = fre_tokenizer.word_index.get('sos', 1)
        end_token_id = fre_tokenizer.word_index.get('eos', 2)
        
        if debug:
            print(f"Start token ID: {start_token_id}, End token ID: {end_token_id}")
        
        if use_beam_search:
            generated_tokens = trainer.model.beam_search(
                encoder_inputs, 
                beam_width=beam_width,
                max_length=50,
                start_token_id=start_token_id,
                end_token_id=end_token_id
            )
        else:
            generated_tokens = trainer.model.generate(
                encoder_inputs, 
                max_length=50,
                start_token_id=start_token_id,
                end_token_id=end_token_id
            )
        
        if debug:
            print(f"Generated token IDs: {generated_tokens}")
    
    # Convert back to text
    generated_sequence = generated_tokens.cpu().numpy().tolist()
    if debug:
        print(f"Generated sequence: {generated_sequence}")
    
    translation = fre_tokenizer.sequences_to_texts(generated_sequence)[0]
    
    # Remove SOS and EOS tokens
    translation = translation.replace('sos ', '').replace(' eos', '').strip()
    
    if debug:
        print(f"Final translation: '{translation}'")
    
    return translation


def evaluate_translations(trainer, test_sentences, eng_tokenizer, fre_tokenizer, 
                         max_eng_length, device='cpu', use_beam_search=False):
    """
    Evaluate model on multiple test sentences
    
    Args:
        trainer: Trained model trainer
        test_sentences: List of English sentences to translate
        eng_tokenizer: English tokenizer
        fre_tokenizer: French tokenizer
        max_eng_length: Maximum English sequence length
        device: Device to run on
        use_beam_search: Whether to use beam search
        
    Returns:
        translations: List of translated sentences
    """
    translations = []
    
    print(f"Evaluating {len(test_sentences)} sentences...")
    print(f"Using {'beam search' if use_beam_search else 'greedy decoding'}")
    print("-" * 60)
    
    for i, sentence in enumerate(test_sentences):
        translation = translate_sentence(
            trainer, sentence, eng_tokenizer, fre_tokenizer, 
            max_eng_length, device, use_beam_search=use_beam_search
        )
        translations.append(translation)
        print(f"{i+1:2d}. EN: {sentence}")
        print(f"    FR: {translation}")
        print()
    
    return translations