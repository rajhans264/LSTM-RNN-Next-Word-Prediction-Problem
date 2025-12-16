import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_model_safely():
    # Prefer modern format
    if os.path.exists("NextWordPredictionModelLSTM.keras"):
        try:
            model = tf.keras.models.load_model("NextWordPredictionModelLSTM.keras")
            return model
        except Exception as e:
            st.warning(f"Failed to load model.keras: {e}")

    # Legacy fallback
    if os.path.exists("NextWordPredictionModelLSTM.h5"):
        try:
            model = tf.keras.models.load_model("NextWordPredictionModelLSTM.h5", compile=False)
            model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
            return model
        except Exception as e:
            st.error(f"Failed to load NextWordPredictionModelLSTM.h5: {e}")

    raise FileNotFoundError("No model file found. Please include NextWordPredictionModelLSTM.keras or NextWordPredictionModelLSTM.h5 in the repository.")

@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
# -----------------------------
# LOAD ARTIFACTS
# -----------------------------

try:
    model = load_model_safely()
except Exception as e:
    st.stop()

try:
    tokenizer = load_pickle("WordTokenizer.pkl")
except FileNotFoundError as e:
    st.error(f"Missing artifact: {e}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

# #Load the LSTM Model
# model=load_model("NextWordPredictionModelLSTM.keras")

# #3 Laod the tokenizer
# with open('WordTokenizer.pkl','rb') as handle:
#     tokenizer=pickle.load(handle)


# Function for Prediction
def NextWordPrediction(model,tokenizer,text,max_seq_len):
  token_list = tokenizer.texts_to_sequences([text])[0]
  if len(token_list)>= max_seq_len:
    token_list = token_list[-(max_seq_len-1):]
  token_list = pad_sequences([token_list],maxlen=max_seq_len-1,padding='pre')
  prediction = model.predict(token_list,verbose=0)
  predicted_word_index = np.argmax(prediction,axis=1)
  for word,index in tokenizer.word_index.items():
    if index == predicted_word_index:
      return word
  return None

# streamlit app
st.title("LSTM-RNN Next Word Predictor")
input_text=st.text_input("Enter the sequence of Words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = NextWordPrediction(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')

# to run the app, use the command: streamlit run app4.py