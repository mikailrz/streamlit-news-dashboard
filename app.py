import streamlit as st

st.title("🚀 My First Streamlit App")
st.write("Hello! This is my first Streamlit web app deployed on Streamlit Cloud.")

# Simple input and output
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}! Welcome to Streamlit.")
