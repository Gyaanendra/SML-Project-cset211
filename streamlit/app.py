# Installation commands (run these first):
# pip install streamlit==1.39.0
# pip install tensorflow==2.16.1 transformers==4.44.2 tf-keras==2.16.0 tokenizers==0.19.1 sentencepiece==0.2.0 --no-deps
# pip install scikit-learn matplotlib pandas numpy

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: disable oneDNN messages

import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras import layers, models

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .fake {
        background-color: #ffebee;
        border: 2px solid #ef5350;
    }
    .real {
        background-color: #e8f5e9;
        border: 2px solid #66bb6a;
    }
    .probability {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Global variable to hold encoder (needed for custom layer)
_GLOBAL_ENCODER = None

# Custom Layer Definition
class HFEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_name: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self._encoder = None
        
    def build(self, input_shape):
        # Encoder will be set from global variable
        global _GLOBAL_ENCODER
        if _GLOBAL_ENCODER is not None:
            self._encoder = _GLOBAL_ENCODER
        super().build(input_shape)
        
    def call(self, inputs):
        if self._encoder is None:
            global _GLOBAL_ENCODER
            if _GLOBAL_ENCODER is None:
                raise RuntimeError("Encoder not initialized. Please load models first.")
            self._encoder = _GLOBAL_ENCODER
            
        output = self._encoder(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            training=False
        )
        return output.last_hidden_state
    
    def get_config(self):
        config = super().get_config()
        config.update({"model_name": self.model_name})
        return config
    
    @classmethod
    def from_config(cls, config):
        model_name = config.pop("model_name", None)
        return cls(model_name, **config)

# Initialize session state
@st.cache_resource
def load_models():
    """Load encoder, tokenizer, and model (cached)"""
    global _GLOBAL_ENCODER
    MODEL_NAME = 'roberta-base'
    
    with st.spinner('Loading encoder and tokenizer...'):
        encoder = TFAutoModel.from_pretrained(MODEL_NAME)
        encoder.trainable = False
        
        # SET GLOBAL ENCODER BEFORE LOADING MODEL
        _GLOBAL_ENCODER = encoder
        
        # Load tokenizer directly from Hugging Face (not from local folder)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    with st.spinner('Loading classification model...'):
        custom_objects = {"HFEncoderLayer": HFEncoderLayer}
        model_path = 'best_fake_news_model.h5'
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            st.stop()
        
        model = models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
    
    return encoder, tokenizer, model, MODEL_NAME

# Load models
try:
    encoder, tokenizer, model, MODEL_NAME = load_models()
    # Global encoder is already set inside load_models()
        
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.stop()

# Prediction function
MAX_LEN = 128

def predict(text: str):
    """Predict if news is fake or real"""
    enc = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_tensors='tf'
    )
    
    prob = model.predict(
        {
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask']
        },
        verbose=0
    ).ravel()[0]
    
    return {
        "probability_fake": float(prob),
        "probability_real": float(1 - prob),
        "label": "Fake" if prob >= 0.5 else "Real"
    }

# UI
st.markdown("<h1 class='main-header'>🔍 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("---")

# Info section
with st.expander("ℹ️ About this tool"):
    st.write(f"""
    This tool uses a fine-tuned **{MODEL_NAME}** model to classify news articles as real or fake.
    
    **How it works:**
    - Enter a news headline or article text
    - The model analyzes the text using transformer-based architecture
    - Get a prediction with confidence score
    
    **Model Details:**
    - Base Model: {MODEL_NAME}
    - Max Length: {MAX_LEN} tokens
    - Threshold: 0.5 (>=0.5 = Fake, <0.5 = Real)
    """)

st.markdown("### Enter news text to analyze:")

# Text input
text_input = st.text_area(
    label="News Text",
    placeholder="Enter a news headline or article text here...",
    height=150,
    label_visibility="collapsed"
)


# Predict button
if st.button("🔍 Analyze News", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            result = predict(text_input)
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        # Result box
        box_class = "fake" if result['label'] == "Fake" else "real"
        prob_display = result['probability_fake'] if result['label'] == "Fake" else result['probability_real']
        
        if result['label'] == 'Fake':
            st.markdown(f"""
                <div class='result-box fake'>
                    <h2 style='color: #c62828;'>🚫 FAKE NEWS</h2>
                    <div class='probability' style='color: #d32f2f;'>{prob_display:.1%}</div>
                    <p style='color: #333333;'>Confidence Level</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='result-box real'>
                    <h2 style='color: #2e7d32;'>✅ REAL NEWS</h2>
                    <div class='probability' style='color: #388e3c;'>{prob_display:.1%}</div>
                    <p style='color: #333333;'>Confidence Level</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Detailed metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fake Probability", f"{result['probability_fake']:.1%}")
        with col2:
            st.metric("Real Probability", f"{result['probability_real']:.1%}")
        
        # Progress bars
        st.markdown("**Confidence Breakdown:**")
        st.progress(result['probability_fake'], text=f"Fake: {result['probability_fake']:.1%}")
        st.progress(result['probability_real'], text=f"Real: {result['probability_real']:.1%}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Powered by RoBERTa & TensorFlow</p>",
    unsafe_allow_html=True
)