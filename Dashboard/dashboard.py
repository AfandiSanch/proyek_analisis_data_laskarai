import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import os
from sklearn.preprocessing import KBinsDiscretizer

# URL dataset kualitas udara
urls = {
    "Aotizhongxin": "https://raw.githubusercontent.com/AfandiSanch/proyek_analisis_data_laskarai/6ecd78a38c9ce4a2101f98edffc5f43daa34e38c/Data%20Air%20Quality/PRSA_Data_Aotizhongxin_20130301-20170228.csv",
    "Changping": "https://raw.githubusercontent.com/AfandiSanch/proyek_analisis_data_laskarai/6ecd78a38c9ce4a2101f98edffc5f43daa34e38c/Data%20Air%20Quality/PRSA_Data_Changping_20130301-20170228.csv",
    "Wanliu": "https://raw.githubusercontent.com/AfandiSanch/proyek_analisis_data_laskarai/6ecd78a38c9ce4a2101f98edffc5f43daa34e38c/Data%20Air%20Quality/PRSA_Data_Wanliu_20130301-20170228.csv",
    "Wanshouxigong": "https://raw.githubusercontent.com/AfandiSanch/proyek_analisis_data_laskarai/6ecd78a38c9ce4a2101f98edffc5f43daa34e38c/Data%20Air%20Quality/PRSA_Data_Wanshouxigong_20130301-20170228.csv"
}

# Fungsi untuk memuat data
@st.cache_data
def load_data(url, station_name):
    df = pd.read_csv(url, usecols=['year', 'month', 'day', 'hour', 'PM2.5', 'PM10', 'RAIN', 'WSPM'])
    df.dropna(inplace=True)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df['station'] = station_name
    return df[['datetime', 'station', 'PM2.5', 'PM10', 'RAIN', 'WSPM']]

# Memuat semua dataset dan menggabungkan
data_frames = [load_data(url, station) for station, url in urls.items()]
data_combined = pd.concat(data_frames, ignore_index=True)

# Sidebar Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Home", "Statistik Deskriptif", "Visualisasi", "Analisis RFM", "Peta PM2.5"])

# Home
if page == "Home":
    st.title("📊 Analisis Kualitas Udara Beijing")
    st.write("### Contoh Data (5 baris per station)")
    
    # Ambil 5 baris per station
    sample_data = data_combined.groupby('station').head(5).reset_index(drop=True)
    
    st.dataframe(sample_data)


# Statistik Deskriptif
elif page == "Statistik Deskriptif":
    st.title("📊 Statistik Deskriptif")
    st.write(data_combined.describe())

# Visualisasi
elif page == "Visualisasi":
    tab1, tab2 = st.tabs(["📈 Tren Polusi", "☔ Scatter Plot Curah Hujan & Angin"])
    with tab1:
        st.write("### Tren Polusi Udara (PM2.5 & PM10)")
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.lineplot(data=data_combined, x='datetime', y='PM2.5', hue='station', marker='o', ax=ax)
        sns.lineplot(data=data_combined, x='datetime', y='PM10', hue='station', marker='x', linestyle='--', ax=ax)
        plt.xlabel("Tanggal")
        plt.ylabel("Konsentrasi (µg/m³)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with tab2:
        st.write("### Pengaruh Curah Hujan & Kecepatan Angin")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(data=data_combined, x='RAIN', y='PM2.5', hue='WSPM', palette='viridis', size='WSPM', sizes=(20, 200), alpha=0.6, ax=ax)
        plt.xlabel("Curah Hujan (mm)")
        plt.ylabel("PM2.5 (µg/m³)")
        st.pyplot(fig)

# Analisis RFM
elif page == "Analisis RFM":
    st.title("📊 Analisis Recency, Frequency, Monetary (RFM)")
    data_combined['date'] = data_combined['datetime'].dt.date
    rfm = data_combined.groupby('station').agg(
        Recency=('date', lambda x: (data_combined['date'].max() - x.max()).days),
        Frequency=('PM2.5', 'count'),
        Monetary=('PM2.5', 'mean')
    ).reset_index()
    st.dataframe(rfm)

# Peta PM2.5
elif page == "Peta PM2.5":
    st.title("🌍 Distribusi PM2.5 berdasarkan Lokasi")

    locations = {
        'station': list(urls.keys()),
        'latitude': [39.99, 40.00, 39.95, 39.93],
        'longitude': [116.31, 116.35, 116.30, 116.28]
    }
    locations_df = pd.DataFrame(locations)
    data_geo = data_combined.groupby('station').agg({'PM2.5': 'mean'}).reset_index()
    data_geo = data_geo.merge(locations_df, on='station')

    # Konversi ke GeoDataFrame
    gdf = gpd.GeoDataFrame(data_geo, geometry=gpd.points_from_xy(data_geo.longitude, data_geo.latitude))

    # Path ke shapefile
    shapefile_path = "110m_cultural/ne_110m_admin_0_countries.shp"

    # Pastikan file tersedia
    if os.path.exists(shapefile_path):
        world = gpd.read_file(shapefile_path)
    else:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Buat peta interaktif menggunakan Folium
    m = folium.Map(location=[39.95, 116.30], zoom_start=10)

    # Tambahkan marker dengan ukuran dan warna berdasarkan PM2.5
    for _, row in gdf.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=row['PM2.5'] / 5,  # Ukuran proporsional
            popup=f"{row['station']}: {row['PM2.5']:.2f} µg/m³",
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.6,
        ).add_to(m)

    folium_static(m)
