import streamlit as st 
import pandas as pd
import numpy as np
import pickle 

# Load the model
clf = pickle.load(open("mymodel.pkl","rb"))

def predict(data):
    clf = pickle.load(open("mymodel.pkl","rb"))
    return clf.predict(data)

st.title("Advertig spends Prediction using machine learning")
st.markdown("This Model Identify total spends on advertising")

st.header("Advertising spend on various media")
col1,col2 = st.columns(2)

with col1:
    st.text("TV")
    tv = st.slider("Adver.Spends on Tv", 1.0,10000.0,0.5)
    st.text("Radio")
    rd = st.slider("adver. spends on radio",1.0,10000.0,0.5)
    st.text("NewsPaper")
    newspaper = st.slider("adver.spends on NewsPaper",1.0,10000.0,0.5)
st.text('')
if st.button("sales Prediction"):
    result = clf.predict(np.array([[tv,rd,newspaper]]))
    st.text(result[0])
st.markdown("Developed by Rex at Nielit daman")
