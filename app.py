import streamlit as st
import tensorflow as tf
import os
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

class HybridModelLoader:
    def __init__(self):
        self.model_dir = os.path.join('saved_models', 'hybrid')
        
    def load_model(self):
        model_path = os.path.join(self.model_dir, 'best_model.h5')
        config_path = os.path.join(self.model_dir, 'config.json')
        tokenizer_path = os.path.join(self.model_dir, 'tokenizer.json')
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Load tokenizer
        with open(tokenizer_path, 'r') as f:
            tokenizer = tokenizer_from_json(f.read())
            
        return model, config, tokenizer

def predict_news(text, model, tokenizer, config):
    # Preprocess text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=config['MAX_LENGTH'])
    
    # Get prediction
    prediction = model.predict(padded)[0][0]
    return float(prediction)

def main():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="🔍",
        layout="centered"
    )
    
    # Header
    st.title("🔍 Fake News Detector")
    st.markdown("### Using Hybrid CNN-BiLSTM Model")
    
    try:
        # Load model
        loader = HybridModelLoader()
        model, config, tokenizer = loader.load_model()
        
        # Text input
        news_text = st.text_area(
            "📰 Enter news article to analyze:",
            height=200,
            placeholder="Paste your news article here..."
        )
        
        # Analyze button
        if st.button("🔍 Analyze Article"):
            if not news_text:
                st.warning("⚠️ Please enter some text to analyze")
                return
                
            with st.spinner('Analyzing...'):
                # Get prediction
                confidence = predict_news(news_text, model, tokenizer, config)
                
                # Results section
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                # Prediction result
                if confidence > 0.5:
                    st.error("🚫 This article appears to be FAKE")
                    reliability = "Low"
                else:
                    st.success("✅ This article appears to be REAL")
                    reliability = "High"
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Reliability Score", reliability)
                with col2:
                    st.metric("Fake Probability", f"{confidence*100:.1f}%")
                with col3:
                    st.metric("Real Probability", f"{(1-confidence)*100:.1f}%")
                
                # Confidence visualization
                st.markdown("### Confidence Level")
                st.progress(confidence)
        
        # Model information
        with st.expander("ℹ️ About the Model"):
            st.markdown("""
            This detector uses a hybrid architecture combining:
            - Convolutional Neural Networks (CNN)
            - Bidirectional LSTM
            - Attention mechanism
            
            The model analyzes both local and sequential patterns in text to identify potential fake news.
            """)
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model files are present in the saved_models/hybrid directory")

if __name__ == "__main__":
    main()