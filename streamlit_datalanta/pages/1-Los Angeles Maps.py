import pandas as pd
import streamlit as st
import folium
import pydeck as pdk
from streamlit_folium import st_folium, folium_static
from folium.plugins import Draw
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# Map Page
st.set_page_config(
    page_title="Maps",
    page_icon="🗺",
    layout="wide"
)

# Haritalar ve datasetler yükleninceye kadar progress bar
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

for i in range(1, 101):
    # Haritalar ve datasetler yükleniyor gibi düşünerek simüle ediyoruz
    # Progress bar güncelleniyor
    progress_bar.progress(i)
    status_text.text("%i%% Complete" % i)


# datasetlerin yüklenmesi
@st.cache_data
def load_data():
    df_houses = pd.read_csv("streamlit_datalanta/datasets/airbnb-listings.csv", sep=";", low_memory=False)
    df_fire_stations = pd.read_csv("streamlit_datalanta/datasets/Fire_Stations-new.csv", low_memory=False)
    df_police_station = pd.read_csv("streamlit_datalanta/datasets/Sheriff_and_Police_Stations.csv", low_memory=False)
    df_hosp2 = pd.read_csv("streamlit_datalanta/datasets/Hospitals_and_Medical_Centers.csv", low_memory=False)
    df_school = pd.read_csv("streamlit_datalanta/datasets/Schools_Colleges_and_Universities.csv", low_memory=False)
    metro_df = pd.read_csv("streamlit_datalanta/datasets/Metro_Rail_Lines_Stops.csv", low_memory=False)
    df_arrest = pd.read_csv("streamlit_datalanta/datasets/arrests_2017.csv", parse_dates=["Arrest Date"])
    df_landmark = pd.read_csv("streamlit_datalanta/datasets/la_landmarks.csv", low_memory=False)
    df_coffee = pd.read_csv("streamlit_datalanta/datasets/la_coffees_filtered.csv", low_memory=False)
    return df_houses, df_fire_stations, df_police_station, df_hosp2, df_school, metro_df, df_arrest, df_landmark, df_coffee


df_la_houses, df_fire_stations, df_police_station, df_hospitals, df_schools, df_metros, df_arrests, df_landmarks, df_coffees = load_data()

title_text = "Los Angeles Mapping"
st.write(f"<h1 style='text-align: center; color:#D83e32;'>{title_text}</h1>", unsafe_allow_html=True)

st.markdown("""
    <style>
        .big-text {
            font-size: 24px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("<p class='big-text'>🏠 Air Bnb Houses 🏠</p>", unsafe_allow_html=True)

df_la_coord = df_la_houses[['Latitude', 'Longitude']]
st.map(df_la_coord, latitude="Latitude", longitude="Longitude")

# Airbnb Houses - PyDeck
st.markdown("<p class='big-text'>🏠 Air Bnb Houses - PyDeck 🏠</p>", unsafe_allow_html=True)
chart_data = pd.DataFrame(
    df_la_coord.values,
    columns=['lat', 'lon'])

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=34.054440,
        longitude=-118.252510,
        zoom=11,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            'HexagonLayer',
            data=chart_data,
            get_position='[lon, lat]',
            radius=200,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=chart_data,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=200,
        ),
    ],
))

st.write("\n")

# Sayfayı bölme
column1, column2 = st.columns(2)

# COLUMN 1
# Fire Stations
column1.markdown("<p style='color:#E85423' class='big-text'>🧯 Fire Stations 🧯</p>", unsafe_allow_html=True)
df_fire_stations = df_fire_stations[df_fire_stations['org_name'] == 'LA County']
column1.map(df_fire_stations, latitude="Y", longitude="X", color='#E85423')

# Metro Lines
column1.markdown("<p style='color:#D93025' class='big-text'>🚇 Metro Lines 🚇</p>",
                 unsafe_allow_html=True)

column1.markdown(
    "|<span style='color:#0000ff;  font-size:12px; '> Blue Line </span>|<span style='color:#ff0000; font-size:12px;'> Red Line </span>|<span style='color:#00ff00; font-size:12px;'> Green Line </span>|<span style='color:#800080; font-size:12px;'> Purple Line </span>|<span style='color:#ffd700; font-size:12px;'> Gold Line </span>|<span style='color:#ffffff; font-size:12px;'> Expo Line </span>|",
    unsafe_allow_html=True)
color_dict = {'Blue Line': '#0000ff',
              'Red Line': '#ff0000',
              'Green Line': '#00ff00',
              'Purple Line': '#800080',
              'Gold Line': '#ffd700',
              'Expo Line': '#ffffff'}
df_metros['Line'] = df_metros['MetroLine'].map(color_dict)
column1.map(df_metros, latitude="Y", longitude="X", color="Line")

# Hospital
column1.markdown("<p style='color:#D93025' class='big-text'>🏥 Hospitals 🏥</p>", unsafe_allow_html=True)
column1.map(df_hospitals, latitude="Y", longitude="X", color='#D93025')

# Police Stations
column2.markdown("<p style='color:#0000ff' class='big-text'>🚨 Police Stations 🚨</p>", unsafe_allow_html=True)
df_police_station = df_police_station[df_police_station['cat2'] == 'Sheriff and Police Stations']
column2.map(df_police_station, latitude="Y", longitude="X", color='#0000ff')

# Schools
column2.markdown(
    "<span style='color:#3ca4d8' class='big-text'>🏫 Schools 🏫</span>",
    unsafe_allow_html=True)

column2.markdown(
    "|<span style='color:#FF0050;  font-size:12px; '> Public Schools </span>|<span style='color:#00FF00; font-size:12px;'> Private Schools </span>|<span style='color:#0000FF; font-size:12px;'> Colleges and Universities </span>|",
    unsafe_allow_html=True)
df_schools = df_schools[df_schools['cat3'].isin(['Public Schools', 'Private Schools', 'Colleges and Universities'])]
# Map categories to colors
color_dict = {'Public Schools': (255, 0, 0, 80),
              'Private Schools': (0, 255, 0, 80),
              'Colleges and Universities': (0, 0, 255, 80)}
df_schools.reset_index(drop=True, inplace=True)
df_schools.loc[:, 'category'] = df_schools['cat3'].map(color_dict)
column2.map(df_schools, latitude="Y", longitude="X", color="category")

# Cafes
column2.markdown(
    "<span style='color:#AF856B' class='big-text'>☕ Cafes ☕</span>",
    unsafe_allow_html=True)
df_coffees.head()
# df_cafe = df_coffees[~df_coffees["CITY"].isin(
#     ["WHITTIER", "CANYON COUNTRY", "SANTA ANA", "SUNLAND", "WEST HOLLYWOOD", "OCEANSIDE", "MOSS LANDING", "MONROVIA",
#     "LAKE HUGHES"])]
# df_cafe2 = df_cafe[~df_cafe["CITY"].isin(
#    ["SOUTH BEND", "SANTA MONICA", "NORTHFIELD", "LA CRESCENTA", "FRANKLIN PARK", "LEES SUMMIT", "FLORHAM PARK",
#     "DOVER", "DELPHOS", "COMPTON", "CHAMPAIGN"])]
# df_cafe3 = df_cafe2[~df_cafe2["CITY"].isin(
#    ["ROCKLIN", "SALT LAKE CITY", "MARINA DEL REY", "PANORAMA CITY", "NORTH HOLLYWOOD", "SHERMAN OAKS", "WEST HILLS"])]
# df_cafe4 = df_cafe3[~df_cafe3["CITY"].isin(
#    ["WOODLAND HILLS", "VALLEY CENTER", "TWENTYNINE PALMS", "TARZANA", "SYLMAR", "SUN VALLEY", "LA QUINTA", "CARSON"])]
# df_cafe5 = df_cafe4[df_cafe4["Longitude"].astype(str).str.startswith("-118")]
# df_cafe5.to_csv("DatalantaProject/datasets/la_coffees_filtered.csv")
column2.map(df_coffees, latitude="Latitude", longitude="Longitude")

# Landmarks
m2 = folium.Map(location=[df_landmarks["Y"].mean(), df_landmarks["X"].mean()], zoom_start=11)
for index, row in df_landmarks.iterrows():
    folium.Marker([row["Y"], row["X"]], popup=f"Upvotes: {row['Upvotes']}", tooltip=row["Name"]).add_to(m2)

# Arrests Areas
df_arrests = df_arrests[df_arrests['Disposition Description'] == 'FELONY COMPLAINT FILED']
df_arrests.drop(df_arrests[(df_arrests['LAT'] == 0) | (df_arrests['LON'] == 0)].index, inplace=True)
df_areaArrest_group = df_arrests.groupby("Area Name").agg({"LAT": "mean",
                                                           "LON": 'mean'})
df_areaArrest_group["crimes"] = df_arrests["Area Name"].value_counts()
df_areaArrest_group.reset_index(inplace=True)

m = folium.Map(location=[df_areaArrest_group["LAT"].mean(), df_areaArrest_group["LON"].mean()], zoom_start=10)
# DataFrame'deki konumları haritaya ekleyin ve tooltip ekleyin
for index, row in df_areaArrest_group.iterrows():
    folium.Marker([row["LAT"], row["LON"]], popup=f"Crimes: {row['crimes']}", tooltip=row["Area Name"])
    radius = 20  # Çemberin yarıçapı (örneğin 500 metre)
    folium.CircleMarker(
        location=[row["LAT"], row["LON"]],
        radius=20,
        color="blue",  # Çember rengi
        fill=True,
        fill_color="blue",  # Dolgu rengi
        fill_opacity=0.2,  # Dolgu saydamlığı
        tooltip=f"Area Name: {row['Area Name']}, Crimes: {row['crimes']}"
    ).add_to(m)

# Output Arrest Areas & Landmarks
with column1:
    st.markdown(
        "<span style='color:#ACAFB3' class='big-text'>🐱‍👤 Arrests Areas 🐱‍👤</span>",
        unsafe_allow_html=True)
    output = st_folium(m, width=1000, height=500, returned_objects=[], key="new")
with column2:
    st.markdown(
        "<span style='color:#F77868' class='big-text'>🎡 Landmarks 🎡</span>",
        unsafe_allow_html=True)
    output2 = st_folium(m2, width=1000, height=500, returned_objects=[], key="new")
