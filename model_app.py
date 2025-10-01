import streamlit as st
from transformers import pipeline

@st.cache_data
def load_data():
    return pipeline("fill-mask", model="bert-base-uncased")

# Our model takes texts as input
# In that text there must be a single [MASK] token
# The model then returns a list of words that could be placed at the [MASK]
# Each predicted word also has a matching score
model = load_data()
st.write("Enter a phrase with [MASK] where you want the predicted word")
text = st.text_input("Input text:", "salut [MASK] ça va ?")

predictions = model(text)
output = [{"token": p["token_str"], "score": p["score"]} for p in predictions]

st.write("Predicted token:", output)