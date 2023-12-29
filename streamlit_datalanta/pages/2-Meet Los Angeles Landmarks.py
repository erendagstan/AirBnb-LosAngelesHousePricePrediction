import pandas as pd
import streamlit as st
import folium
import pydeck as pdk
from streamlit_folium import st_folium, folium_static
from folium.plugins import Draw

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# Page settings
st.set_page_config(
    page_title="Landmarks",
    page_icon="🎡",
    layout="wide"
)


# Load data
@st.cache_data
def load_data():
    df_landmark = pd.read_csv("DatalantaProject/streamlit_datalanta/datasets/la_landmarks.csv", low_memory=False)
    df_la = pd.read_csv("DatalantaProject/streamlit_datalanta/datasets/la_dataframe_v4.csv", low_memory=False)
    return df_landmark, df_la


# assigning data
df_landmarks, df_la = load_data()

st.header(":red[Meet Los Angeles Landmarks] : :blue[Closest Airbnb Houses]")

option2 = st.selectbox(
    'Which landmark would you like to see?',
    ('-', 'Hollywood Sign', 'Staples Center', 'Walt Disney Concert Hall',
     'Venice Canals Walkway', 'Hollywood Walk of Fame',
     'The Wizard World of Harry Potter', 'Dodger Stadium',
     'TCL Chinese Theatres', 'Universal CityWalk Hollywood',
     'Union Station', 'Olvera Street', 'Little Tokyo',
     'Venice Beach Boardwalk', 'Los Angeles County Museum of Art',
     'Griffith Observatory', 'Griffith Park', 'The Getty',
     'Santa Monica Pier', 'La Brea Tar Pits and Museum'))

# Recommendation
if (option2 != "-"):
    landmark_nums = st.number_input("How many houses would you like to see?", min_value=1, max_value=25, value=5)
    x = df_la[df_la["nearly_landmark_name"] == option2].sort_values("distance_landmark", ascending=True)[
            ["Latitude", "Longitude", "nearly_landmark_name", "distance_landmark", "Price", "Listing Url"]][
        :landmark_nums]
    lat_x = df_landmarks[df_landmarks["Name"] == option2]["Y"].values[0]
    long_x = df_landmarks[df_landmarks["Name"] == option2]["X"].values[0]
    tuple_x = (lat_x, long_x)
    m3 = folium.Map(location=[lat_x, long_x], zoom_start=16)
    folium.Marker(location=tuple_x, popup=option2, icon=folium.Icon(color='red')).add_to(m3)
    for i in range(landmark_nums):
        folium.Marker(
            location=[x['Latitude'].iloc[i], x['Longitude'].iloc[i]],
            popup=f"House {i + 1} - Price: ${x['Price'].iloc[i]} - URL: {x['Listing Url'].iloc[i]}",
            icon=folium.Icon(color='blue')
        ).add_to(m3)
    folium_static(m3, width=1200, height=600)
