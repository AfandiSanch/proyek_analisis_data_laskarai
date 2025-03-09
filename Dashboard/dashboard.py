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
    st.write("### Data (5 baris per station)")
    
    # Ambil 5 baris per station
    sample_data = data_combined.groupby('station').head(5).reset_index(drop=True)
    
    st.dataframe(sample_data)

    # Tambahkan penjelasan dan pertanyaan
    st.write("### 🔍 Analisis Polusi Udara")
    st.write("""
    Data di atas menunjukkan contoh pengukuran kualitas udara dari berbagai stasiun pemantauan di Beijing.
    Polusi udara diukur berdasarkan parameter PM2.5 dan PM10, yang sering dipengaruhi oleh faktor cuaca seperti curah hujan dan kecepatan angin.
    
    **Pertanyaan Analisis:**
    1. Bagaimana tren polusi udara (PM2.5, PM10) dari waktu ke waktu di masing-masing stasiun?
    2. Bagaimana pengaruh curah hujan (RAIN) dan kecepatan angin (WSPM) terhadap tingkat polusi udara (PM2.5) di seluruh stasiun?
    """)

# Statistik Deskriptif
elif page == "Statistik Deskriptif":
    st.title("📊 Statistik Deskriptif")
    
    # Menampilkan tabel statistik deskriptif
    st.write(data_combined.describe())
    
    # Menambahkan penjelasan di bawah tabel
    st.write("### 📌 Interpretasi Statistik Deskriptif")
    st.write("""
    **Tren Polusi Udara (PM2.5, PM10) dari Waktu ke Waktu**
    - Jika rata-rata PM2.5 dan PM10 tinggi dengan standar deviasi besar, kemungkinan ada fluktuasi besar dalam tingkat polusi.
    - Jika median dan mean berbeda jauh, distribusi data mungkin tidak simetris, menunjukkan adanya lonjakan polusi pada periode tertentu.
    - Rentang nilai (min-max) menunjukkan variasi besar antara hari-hari dengan polusi rendah dan tinggi.

    **Pengaruh Curah Hujan (RAIN) dan Kecepatan Angin (WSPM) terhadap PM2.5**
    - Jika rata-rata PM2.5 lebih rendah saat curah hujan tinggi, ini menunjukkan bahwa hujan membantu mengurangi polusi udara.
    - Jika kecepatan angin tinggi berkorelasi dengan rendahnya PM2.5, angin mungkin membantu menyebarkan polutan.
    - Jika standar deviasi WSPM tinggi, berarti kecepatan angin bervariasi, yang bisa memengaruhi distribusi polusi udara.
    """)



# Visualisasi
elif page == "Visualisasi":
    tab1, tab2 = st.tabs(["📈 Tren Polusi", "☔ Scatter Plot Curah Hujan & Angin"])

    # Tab 1: Tren Polusi Udara
    with tab1:
        st.write("### Tren Polusi Udara (PM2.5 & PM10)")
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.lineplot(data=data_combined, x='datetime', y='PM2.5', hue='station', marker='o', ax=ax)
        sns.lineplot(data=data_combined, x='datetime', y='PM10', hue='station', marker='x', linestyle='--', ax=ax)
        plt.xlabel("Tanggal")
        plt.ylabel("Konsentrasi (µg/m³)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Menambahkan penjelasan di bawah grafik
        st.write("### 📌 Analisis Tren Polusi Udara")
        st.write("""
        - Tren polusi udara menunjukkan fluktuasi kadar PM2.5 dan PM10 dari waktu ke waktu di setiap stasiun pemantauan.
        - Umumnya, kadar polutan cenderung meningkat pada musim dingin, terutama karena suhu lebih rendah dan atmosfer yang lebih stabil membuat polutan terperangkap di dekat permukaan tanah.
        - Sebaliknya, polusi cenderung menurun selama musim hujan karena curah hujan membantu membersihkan udara dari partikel polutan.
        - Lonjakan tertentu dalam tren polusi dapat dikaitkan dengan faktor eksternal seperti kabut asap, peningkatan aktivitas industri, atau meningkatnya jumlah kendaraan bermotor.
        - Jika terdapat puncak atau lonjakan drastis pada tren, kemungkinan ada kejadian khusus seperti kebakaran hutan atau kondisi atmosfer yang tidak menguntungkan.
        """)

    # Tab 2: Scatter Plot Curah Hujan & Kecepatan Angin terhadap PM2.5
    with tab2:
        st.write("### Pengaruh Curah Hujan & Kecepatan Angin terhadap PM2.5")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(data=data_combined, x='RAIN', y='PM2.5', hue='WSPM', palette='viridis', size='WSPM', sizes=(20, 200), alpha=0.6, ax=ax)
        plt.xlabel("Curah Hujan (mm)")
        plt.ylabel("PM2.5 (µg/m³)")
        st.pyplot(fig)

        # Menambahkan penjelasan di bawah grafik
        st.write("### 📌 Analisis Pengaruh Curah Hujan & Kecepatan Angin terhadap PM2.5")
        st.write("""
        - Scatter plot menunjukkan bahwa secara umum, semakin tinggi curah hujan, semakin rendah konsentrasi PM2.5. Hal ini karena hujan dapat membersihkan udara dengan menangkap partikel polusi dan membawanya turun ke tanah.
        - Kecepatan angin juga berperan dalam mengurangi tingkat polusi udara. Angin yang lebih kuat membantu menyebarkan partikel polutan sehingga tidak terakumulasi di satu lokasi.
        - Namun, jika ada titik-titik dengan curah hujan tinggi tetapi PM2.5 tetap tinggi, ini bisa menandakan adanya sumber polusi lain yang dominan, seperti emisi kendaraan, aktivitas industri, atau pembakaran biomassa.
        - Dalam kondisi kecepatan angin rendah dan tidak ada hujan, polutan cenderung tetap berada di udara untuk waktu yang lebih lama, meningkatkan tingkat polusi.
        - Dengan memahami hubungan ini, kebijakan pengendalian polusi dapat diarahkan dengan lebih baik, seperti meningkatkan penghijauan, mengurangi emisi industri, dan memantau kondisi cuaca untuk mengambil tindakan preventif.
        """)

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

     # Menambahkan penjelasan di bawah tabel
    st.write("### 📌 Penjelasan Analisis RFM dalam Konteks Polusi Udara")
    st.write("""
    Dalam analisis polusi udara, konsep **Recency, Frequency, Monetary (RFM)** bisa diadaptasi sebagai berikut:

    1. **Recency (R) – Seberapa Baru Terjadinya Polusi**
       - Recency mengacu pada waktu terakhir kali tingkat polusi udara tinggi (PM2.5 dan PM10).
       - Jika ada jeda panjang sejak polusi terakhir kali tinggi, ini bisa menunjukkan perbaikan kualitas udara atau pengaruh faktor lingkungan seperti hujan dan angin.

    2. **Frequency (F) – Seberapa Sering Polusi Terjadi**
       - Frequency menunjukkan seberapa sering suatu stasiun pemantauan mendeteksi polusi udara dalam rentang waktu tertentu.
       - Jika suatu stasiun memiliki frekuensi tinggi dalam mendeteksi PM2.5 dan PM10 di atas ambang batas, ini bisa menunjukkan daerah tersebut rawan polusi.

    3. **Monetary (M) – Seberapa Parah Dampak Polusi**
       - Monetary dalam konteks ini bisa diartikan sebagai tingkat keparahan polusi berdasarkan rata-rata konsentrasi PM2.5 dan PM10.
       - Semakin tinggi nilai Monetary, semakin besar bahaya polusi bagi kesehatan masyarakat.

    ### 🔍 Analisis Pertanyaan:
    1. **Bagaimana Tren Polusi Udara (PM2.5, PM10) dari Waktu ke Waktu?**
       - Polusi udara bervariasi di setiap stasiun pemantauan.
       - Polusi bisa meningkat pada musim dingin karena efek pemanasan rumah tangga dan penurunan kecepatan angin.
       - Curah hujan membantu mengurangi konsentrasi PM2.5 dan PM10.

    2. **Bagaimana Pengaruh Curah Hujan (RAIN) dan Kecepatan Angin (WSPM) terhadap Polusi Udara?**
       - Polusi cenderung lebih rendah saat curah hujan tinggi karena hujan membersihkan partikel polusi dari udara.
       - Kecepatan angin yang tinggi membantu menyebarkan polusi, mengurangi konsentrasi PM2.5 dan PM10 di satu lokasi tertentu.
    """)

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

    # Menampilkan tabel ringkasan PM2.5 per lokasi
    pm25_summary = data_combined.groupby("station")["PM2.5"].mean().reset_index()
    pm25_summary.columns = ["Stasiun", "Rata-rata PM2.5"]
    st.write("### Rata-rata PM2.5 di Setiap Stasiun")
    st.dataframe(pm25_summary)

    # Tambahkan penjelasan di bawah tabel
    st.write("### 📌 Analisis Distribusi PM2.5 Berdasarkan Lokasi")
    st.write("""
    Data menunjukkan bahwa distribusi PM2.5 bervariasi di berbagai stasiun pemantauan di Beijing. Beberapa poin penting dari analisis ini adalah:

    **1. Variasi Polusi di Berbagai Stasiun**  
    - Stasiun tertentu memiliki konsentrasi PM2.5 lebih tinggi karena faktor geografis dan aktivitas manusia seperti industri dan transportasi.
    - Stasiun di pusat kota umumnya mencatat polusi yang lebih tinggi dibandingkan daerah pinggiran.

    **2. Pengaruh Cuaca terhadap Polusi Udara**  
    - **Curah Hujan (RAIN)**: Hujan dapat membantu membersihkan udara dari partikel polutan, sehingga daerah dengan curah hujan tinggi cenderung memiliki tingkat PM2.5 yang lebih rendah.
    - **Kecepatan Angin (WSPM)**: Angin dapat menyebarkan polusi dan mengurangi konsentrasi lokal PM2.5, terutama di daerah terbuka.

    **3. Polusi Udara dan Tren Musiman**  
    - Selama musim dingin, polusi udara cenderung lebih tinggi karena udara yang lebih stabil dan peningkatan emisi dari sistem pemanas.
    - Pada musim hujan, polusi cenderung menurun karena efek pencucian atmosfer oleh curah hujan.

    **4. Kesimpulan dan Tindakan**  
    - Untuk mengurangi polusi udara, diperlukan lebih banyak ruang hijau, pengurangan emisi kendaraan, serta kebijakan yang mendukung kualitas udara bersih.
    - Data ini dapat digunakan untuk merancang strategi mitigasi yang lebih efektif di area dengan tingkat polusi tinggi.
    """)

