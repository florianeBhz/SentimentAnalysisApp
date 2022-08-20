import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import os

st.set_page_config(page_title="Database", page_icon="📁",layout="wide")

st.markdown("# Database")
st.sidebar.header("Database")
st.write("Read and add new entries to the test database.")


# create form 
st.write("### Create a new entry")
create_text = st.text_input(label="Text",key="form_text",placeholder="ex : I'm against Covid-19 vaccine.")
create_label = st.selectbox("Choose a sentiment label",('0 : Negative', '1 : Neutral','2 : Positive'))

create_button = st.button(label="Create entry")

if create_button:
    if create_text.strip() == "":
        st.write("There is no text to save.")
    else:
        label = create_label.split(':')[0].strip()
        st.write("Text :**"+create_text+"**")
        st.write("Sentiment : **"+label+"**")
    


@st.cache
def get_data():
    #AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
    #df = pd.read_csv("test.csv",header=0, engine='python')#AWS_BUCKET_URL + "/agri.csv.gz")
    #df = df.drop(columns=['Unnamed: 0'])
    #df.labels = df.labels.astype('int')
    #df.dropna(inplace=True)

    return None


try:
    st.write("### Dataset overview")
    t1, t2 = st.tabs(['Filter','Dataset'])
    data = get_data().iloc [:10]
    
    with t1:
        filter = st.multiselect(
        "Filter label", [0, 1, 2]
    )
    with t2:
        if not filter:
            st.dataframe(data.sort_index())
        else:
            data = data[data['labels'].isin(filter)]
            st.dataframe(data.sort_index())
            
        
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )