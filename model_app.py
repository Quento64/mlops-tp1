import streamlit as st
from transformers import pipeline

@st.cache_data
def load_data():
    return pipeline("fill-mask", model="bert-base-uncased")

model = load_data()
text = st.text_input("input_text", "salut [MASK] ça va ?")

predictions = model(text)
output = [{"token": p["token_str"], "score": p["score"]} for p in predictions]

st.write("Predicted token:", output)