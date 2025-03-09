import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

# URL dataset
urls = {
    "Aotizhongxin": "https://raw.githubusercontent.com/AfandiSanch/proyek_analisis_data_laskarai/6ecd78a38c9ce4a2101f98edffc5f43daa34e38c/Data%20Air%20Quality/PRSA_Data_Aotizhongxin_20130301-20170228.csv",
    "Changping": "https://raw.githubusercontent.com/AfandiSanch/proyek_analisis_data_laskarai/6ecd78a38c9ce4a2101f98edffc5f43daa34e38c/Data%20Air%20Quality/PRSA_Data_Changping_20130301-20170228.csv",
    "Wanliu": "https://raw.githubusercontent.com/AfandiSanch/proyek_analisis_data_laskarai/6ecd78a38c9ce4a2101f98edffc5f43daa34e38c/Data%20Air%20Quality/PRSA_Data_Wanliu_20130301-20170228.csv",
    "Wanshouxigong": "https://raw.githubusercontent.com/AfandiSanch/proyek_analisis_data_laskarai/6ecd78a38c9ce4a2101f98edffc5f43daa34e38c/Data%20Air%20Quality/PRSA_Data_Wanshouxigong_20130301-20170228.csv"
}

# Fungsi untuk memuat data
@st.cache_data
def load_data(url, station):
    df = pd.read_csv(url)

    # Hapus baris yang memiliki NaN di kolom tahun, bulan, hari, jam
    df.dropna(subset=['year', 'month', 'day', 'hour'], inplace=True)

    # Konversi ke datetime dengan error handling
    df['datetime'] = pd.to_datetime(
        df[['year', 'month', 'day', 'hour']].astype(str).agg('-'.join, axis=1), 
        errors='coerce'
    )

    # Hapus data yang gagal dikonversi ke datetime
    df.dropna(subset=['datetime'], inplace=True)

    df['station'] = station  # Tambahkan kolom station
    return df

# Memuat semua dataset dengan nama station masing-masing
data_frames = [load_data(url, station) for station, url in urls.items()]
data_combined = pd.concat(data_frames, ignore_index=True)

# Sidebar Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Home", "Statistik Deskriptif", "Visualisasi", "Peta PM2.5"])

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
    st.title("📊 Visualisasi Data")
    
    # Pastikan 'PM2.5' tersedia sebelum memvisualisasikan
    if 'PM2.5' in data_combined.columns:
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.lineplot(data=data_combined, x='datetime', y='PM2.5', hue='station', marker='o', ax=ax)
        plt.xticks(rotation=45)
        plt.xlabel("Tanggal")
        plt.ylabel("PM2.5 (µg/m³)")
        st.pyplot(fig)
    else:
        st.warning("Kolom 'PM2.5' tidak ditemukan dalam dataset!")

# Peta PM2.5
elif page == "Peta PM2.5":
    st.title("🌍 Distribusi PM2.5")

    # Koordinat lokasi
    locations = {
        'station': list(urls.keys()),
        'latitude': [39.99, 40.00, 39.95, 39.93],
        'longitude': [116.31, 116.35, 116.30, 116.28]
    }
    locations_df = pd.DataFrame(locations)

    # Hanya gunakan data yang memiliki 'PM2.5'
    if 'PM2.5' in data_combined.columns:
        data_geo = data_combined.dropna(subset=['PM2.5']).groupby('station').agg({'PM2.5': 'mean'}).reset_index()
        data_geo = data_geo.merge(locations_df, on='station')

        # Konversi ke GeoDataFrame
        gdf = gpd.GeoDataFrame(data_geo, geometry=gpd.points_from_xy(data_geo.longitude, data_geo.latitude))

        # Pastikan dataset peta tersedia
        if "naturalearth_lowres" in gpd.datasets.available:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

            # Plot peta
            fig, ax = plt.subplots(figsize=(10, 10))
            world.boundary.plot(ax=ax, linewidth=1, color="black")
            gdf.plot(column='PM2.5', ax=ax, legend=True, cmap='OrRd', markersize=100)

            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            st.pyplot(fig)
        else:
            st.error("Dataset peta tidak ditemukan. Pastikan GeoPandas terinstal dengan benar.")
    else:
        st.warning("Kolom 'PM2.5' tidak ditemukan dalam dataset!")
