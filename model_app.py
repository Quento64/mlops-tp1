import streamlit as st
import joblib

@st.cache_data
def load_data():
    return joblib.load("regression.joblib")

model = load_data()
size = st.number_input("size")
nb_rooms = st.number_input("nb_rooms", 0, value=0)
garden = st.number_input("garden", 0, 1, 0)

prediction = model.predict([[size, nb_rooms, garden]])
st.write("Predicted price:", prediction[0])