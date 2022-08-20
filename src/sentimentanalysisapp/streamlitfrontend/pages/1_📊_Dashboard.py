import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard", page_icon="📊")
st.sidebar.header("Dashboard")
st.title("Model performances")

labels_dict={
0:'Negative',
1:'Neutral',
2:'Positive'
}

@st.cache
def get_data():
    #AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
    #df = pd.read_csv("test.csv",header=0, engine='python',usecols=['tweet_prep','labels'])#AWS_BUCKET_URL + "/agri.csv.gz")
    #df.dropna(inplace=True)
    #df.labels = df.labels.astype('int')
    return None


try:
    data = get_data()
    # Model Last metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Accuracy", "96%","")
    col2.metric("Model F1 score", "9", "")
    col3.metric("Inference time", "120 S", "")
    col4.metric("Last trained at", "2022:08:11", "")

    # Metrics historic
    #convert to subplots
    col5,col6 = st.columns(2)
    with col5:
        st.write("Accuracy")
        chart_data1 = pd.DataFrame([85, 92, 93, 96])
        st.line_chart(chart_data1,height=120)

        st.write("F1")
        chart_data2 = pd.DataFrame([4, 4, 6, 9])
        st.line_chart(chart_data2,height=120)
            
        st.write("Inference time")
        chart_data3 = pd.DataFrame([125, 126, 125, 1])
        st.line_chart(chart_data3,height=120)

    # Confusion matrix
    with col6:
        st.write("Confusion matrix")
        #plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
        #st.pyplot()

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s

        x = list(data.labels.value_counts().index.map(lambda v:labels_dict[v]))
        y = data.labels.value_counts()

        source = ColumnDataSource(data=dict(x=x, y=y, color=Spectral6))
        p = figure(x_range=x, y_range=(0,9), height=250, title="Sentiment Labels Counts",
           toolbar_location=None, tools="",tooltips=[("Number", "@y")])

        p.vbar(x='x', top='y', width=0.5, color='color', legend_field="x", source=source)

        p.xgrid.grid_line_color = None
        p.legend.orientation = "horizontal"
        #p.legend.location = "top_center"

        st.bokeh_chart(p, use_container_width=True)
    """
        % e.reason
    )