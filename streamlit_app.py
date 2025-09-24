import streamlit as st
from streamlit_gsheets import GSheetsConnection

# Set the page configuration to use a wide layout
st.set_page_config(layout="wide")

# Create a connection object to the Google Sheet
conn = st.connection("gsheets", type=GSheetsConnection)

# Read the data from the Google Sheet into a DataFrame with no caching
df = conn.read(ttl=0)

# Display the DataFrame in the Streamlit app with a specified height
st.dataframe(df, height=600)
