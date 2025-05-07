import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import ( # type: ignore
    Input, Embedding, Conv1D, MaxPooling1D, LSTM,
    Bidirectional, Dense, Dropout, Concatenate, GlobalMaxPooling1D
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

class HybridConfig:
    def __init__(self):
        self.MAX_WORDS = 50000
        self.MAX_LENGTH = 200
        self.EMBEDDING_DIM = 128
        self.CNN_FILTERS = [64, 128]
        self.LSTM_UNITS = [64, 32]
        self.DENSE_UNITS = [256, 128]
        self.DROPOUT_RATES = [0.5, 0.3]


def build_hybrid_model(config):
    """Build and return a hybrid model combining CNN, BiLSTM, and Attention."""
    
    # Input layer
    input_layer = Input(shape=(config.MAX_LENGTH,))
    
    # Embedding layer
    embedding = Embedding(config.MAX_WORDS, config.EMBEDDING_DIM)(input_layer)
    
    # CNN branch
    conv1 = Conv1D(config.CNN_FILTERS[0], 5, activation='relu')(embedding)
    pool1 = MaxPooling1D(2)(conv1)
    conv2 = Conv1D(config.CNN_FILTERS[1], 5, activation='relu')(pool1)
    pool2 = GlobalMaxPooling1D()(conv2)
    
    # BiLSTM branch
    lstm1 = Bidirectional(LSTM(config.LSTM_UNITS[0], return_sequences=True))(embedding)
    lstm2 = Bidirectional(LSTM(config.LSTM_UNITS[1], return_sequences=True))(lstm1)
    
    # Attention mechanism
    attention = tf.keras.layers.Attention()([lstm2, lstm2])
    attention_pool = GlobalMaxPooling1D()(attention)
    
    # Combine branches
    concat = Concatenate()([pool2, attention_pool])
    
    # Dense layers with corrected dropout
    dense1 = Dense(config.DENSE_UNITS[0], activation='relu')(concat)
    dropout1 = Dropout(config.DROPOUT_RATES[0])(dense1)
    dense2 = Dense(config.DENSE_UNITS[1], activation='relu')(dropout1)
    dropout2_layer = Dropout(config.DROPOUT_RATES[1])(dense2)  # Fixed variable name
    output = Dense(1, activation='sigmoid')(dropout2_layer)
    
    # Create model
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
    return model

def prepare_data(texts, config):
    """Prepare text data for the model."""
    # Convert all texts to strings and handle NaN values
    texts = [str(text) if pd.notnull(text) else '' for text in texts]
    
    tokenizer = Tokenizer(num_words=config.MAX_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=config.MAX_LENGTH)
    return padded_sequences, tokenizer

def train_and_save_hybrid(X_train, y_train, X_val, y_val):
    """Train and save the hybrid model."""
    
    # Create config
    config = HybridConfig()
    
    # Prepare data
    X_train_pad, tokenizer = prepare_data(X_train, config)
    X_val_pad, _ = prepare_data(X_val, config)
    
    # Create model
    model = build_hybrid_model(config)
    
    # Create directory for saved models
    save_dir = os.path.join('saved_models', 'hybrid')
    os.makedirs(save_dir, exist_ok=True)
    
    # Define callbacks
    checkpoint_path = os.path.join(save_dir, 'best_model.h5')
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            mode='max',
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_val_pad, y_val),
        epochs=3,
        batch_size=32,
        callbacks=callbacks
    )
    
    # Save tokenizer
    tokenizer_path = os.path.join(save_dir, 'tokenizer.json')
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
    
    # Save config
    config_path = os.path.join(save_dir, 'config.json')
    config_dict = {
        'MAX_WORDS': config.MAX_WORDS,
        'MAX_LENGTH': config.MAX_LENGTH,
        'EMBEDDING_DIM': config.EMBEDDING_DIM,
        'CNN_FILTERS': config.CNN_FILTERS,
        'LSTM_UNITS': config.LSTM_UNITS,
        'DENSE_UNITS': config.DENSE_UNITS,
        'DROPOUT_RATES': config.DROPOUT_RATES
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f)
    
    return model, history, tokenizer

# In the main section, modify the data loading:
if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Load data with error handling
    try:
        # Load fake news data
        fake_df = pd.read_csv('Fake.csv')
        
        # Load and combine real news data
        real_df1 = pd.read_csv('Real.csv')
        real_df2 = pd.read_csv('Real_one.csv')
        real_df3 = pd.read_csv('Final_one.csv')
        real_df4 = pd.read_csv('Final_one_in.csv')
        
        # Print column names for debugging
        print("Fake.csv columns:", fake_df.columns)
        print("Real.csv columns:", real_df1.columns)
        print("Real_one.csv columns:", real_df2.columns)
        
        # Combine real news dataframes
        true_df = pd.concat([real_df1, real_df2, real_df3, real_df4], ignore_index=True)
        
        # Standardize column names - adjust these based on your actual column names
        required_columns = ['title', 'text']
        
        # Create text column from available columns
        for df in [fake_df, true_df]:
            if 'text' not in df.columns:
                if 'content' in df.columns:
                    df['text'] = df['content']
                elif 'description' in df.columns:
                    df['text'] = df['description']
                else:
                    df['text'] = df['title']  # fallback to title if no content/description
        
        # Add labels
        fake_df['label'] = 1
        true_df['label'] = 0
        
        # Combine datasets
        df = pd.concat([fake_df, true_df], ignore_index=True)
        
        # Ensure text column exists and handle missing values
        df['text'] = df['text'].fillna('')
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            df['text'].values,
            df['label'].values,
            test_size=0.2,
            random_state=42
        )
        
        # Train and save model
        model, history, tokenizer = train_and_save_hybrid(X_train, y_train, X_val, y_val)
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Please check your CSV files and column names")
        raise