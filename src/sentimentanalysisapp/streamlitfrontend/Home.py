import streamlit as st
import json
import requests
import time 

labels_dict={
'0':'Negative 👎',
'1':'Neutral 😐',
'2':'Positive 👍'
}

st.set_page_config(
    page_title="Sentiment analysis on COVID-19 vaccine",
    layout="wide")

st.title("Welcome to COVID-19 vaccine sentiment analysis app ")
st.sidebar.header("Home")
st.subheader("What do you think about COVID-19 vaccine ? 🤔 ")


text_input = st.text_input(label="Comment",key="text_to_predict",placeholder="ex : I'm against Covid-19 vaccine.")
predict_button = st.button(label="Predict sentiment")

if predict_button:
    if text_input.strip() == "":
        st.write("There is no comment to predict sentiment on. Plese enter your text bellow.")
    else:

        prediction_url = "http://backend:8080/predict/"+text_input
        res = requests.get(prediction_url, timeout=3)
        st.write("Your sentiment is : **{}**.".format(labels_dict[res.json()['result']]))
    