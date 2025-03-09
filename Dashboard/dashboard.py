import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
from sklearn.preprocessing import KBinsDiscretizer

# URL dataset
urls = {
    "Aotizhongxin": "https://raw.githubusercontent.com/AfandiSanch/proyek_analisis_data_laskarai/6ecd78a38c9ce4a2101f98edffc5f43daa34e38c/Data%20Air%20Quality/PRSA_Data_Aotizhongxin_20130301-20170228.csv",
    "Changping": "https://raw.githubusercontent.com/AfandiSanch/proyek_analisis_data_laskarai/6ecd78a38c9ce4a2101f98edffc5f43daa34e38c/Data%20Air%20Quality/PRSA_Data_Changping_20130301-20170228.csv",
    "Wanliu": "https://raw.githubusercontent.com/AfandiSanch/proyek_analisis_data_laskarai/6ecd78a38c9ce4a2101f98edffc5f43daa34e38c/Data%20Air%20Quality/PRSA_Data_Wanliu_20130301-20170228.csv",
    "Wanshouxigong": "https://raw.githubusercontent.com/AfandiSanch/proyek_analisis_data_laskarai/6ecd78a38c9ce4a2101f98edffc5f43daa34e38c/Data%20Air%20Quality/PRSA_Data_Wanshouxigong_20130301-20170228.csv"
}

# Fungsi untuk memuat data
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df.dropna(inplace=True)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    return df

# Memuat semua dataset
data_frames = {station: load_data(url) for station, url in urls.items()}

data_combined = pd.concat(data_frames.values(), ignore_index=True)
data_combined['station'] = np.repeat(list(data_frames.keys()), [len(df) for df in data_frames.values()])

# Sidebar Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Home", "Statistik Deskriptif", "Visualisasi", "Analisis RFM", "Peta PM2.5"])

# Home
if page == "Home":
    st.title("📊 Proyek Analisis Data: Air Quality Dataset")
    st.write("### Data")
    st.dataframe(data_combined.head())

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
    gdf = gpd.GeoDataFrame(data_geo, geometry=gpd.points_from_xy(data_geo.longitude, data_geo.latitude))

    shapefile_path = r"\\110m_cultural\\ne_110m_admin_0_countries.shp"
    if os.path.exists(shapefile_path):
        world = gpd.read_file(shapefile_path)
        fig, ax = plt.subplots(figsize=(10, 10))
        world.boundary.plot(ax=ax, linewidth=1, color="black")
        gdf.plot(column='PM2.5', ax=ax, legend=True, cmap='OrRd', markersize=100)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        st.pyplot(fig)
    else:
        st.error(f"File Shapefile tidak ditemukan: {shapefile_path}. Pastikan path benar!")
