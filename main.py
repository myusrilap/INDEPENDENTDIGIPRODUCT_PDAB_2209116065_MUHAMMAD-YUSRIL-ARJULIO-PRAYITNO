import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

st.set_option('deprecation.showPyplotGlobalUse', False)

def load_data():
    return pd.read_csv('datacleaned.csv')
    
def load_data2():
    return pd.read_csv('Data_Cleaned.csv')



def kmeans(data):
    # Normalisasi data menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data)
    
    kmeans_model = KMeans(n_clusters=4, random_state=42)

    # Melatih model dengan data yang telah dinormalisasi
    kmeans_model.fit(data_norm)

    # Prediksi cluster untuk setiap data
    cluster_labels = kmeans_model.predict(data_norm)

    # Gabungkan data asli dengan label cluster
    data_with_clusters = pd.concat([data.reset_index(drop=True), pd.Series(cluster_labels, name='Cluster')], axis=1)

    st.write(data_with_clusters)

    return kmeans_model, data_with_clusters


def visualize_kmeans_clustering(x_final_norm, kmeans_model):
    # Normalisasi data menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    x_final_norm_scaled = scaler.fit_transform(x_final_norm)
    
    # Reduksi dimensi menggunakan PCA
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_final_norm_scaled)
    
    # Prediksi label kluster menggunakan model K-Means
    kmeans_labels = kmeans_model.predict(x_final_norm_scaled)

    # Visualisasi K-Means Clustering
    plt.figure(figsize=(8, 6))
    # Menggunakan palet warna yang berbeda-beda untuk setiap kluster
    sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=kmeans_labels, palette='tab10', s=100)
    plt.title('K-Means Clustering')
    plt.legend(title='Cluster')
    plt.axis('equal')  # Menyesuaikan skala sumbu x dan y
    st.pyplot()
    
def main():
    st.title('Analisis Segmentasi Pelanggan')
    st.sidebar.title("Selamat Datang!")
    df = load_data()
    df2 = load_data2()
    selected_features = ["Gender", "AgeGroup", "Spending Type", "PriceCategory","Product Category"]
    x_final = df[selected_features]
    selected_features2 = ["Gender", "AgeGroup", "Spending Type", "PriceCategory", "Product Category_Beauty", "Product Category_Clothing", "Product Category_Electronics"]
    x_final_norm = df2[selected_features2]
    with st.sidebar:
        page = option_menu("Pilih Halaman",
                           ["Informasi Dasar","Distribusi", "Visualisasi Berdasarkan Total Transaksi", "Visualisasi Berdasarkan Produk Yang Dibeli", "Clustering"])
    if page == "Informasi Dasar":
        st.subheader("Informasi Dasar Dataset yang digunakan")
        st.write(x_final.head())

        st.write("**Kategori Usia (AgeGroup)**:")
        st.write("- **Young Adult**: Usia 18-27 tahun.")
        st.write("- **Early Adult**: Usia 28-37 tahun.")
        st.write("- **Adult**: Usia 38-47 tahun.")
        st.write("- **Middle-Aged Adult**: Usia 48-57 tahun.")
        st.write("- **Elderly Adult**: Usia 58-64 tahun.")
        st.write("- **Uncategorized Age**: Usia yang tidak masuk ke dalam kategori yang ditentukan.")

        st.write("**Kategori Jumlah Belanjaan (Spending Type)**:")
        st.write("- **Low Spend**: Jumlah belanjaan antara 25 hingga 100.")
        st.write("- **Moderate Spend**: Jumlah belanjaan antara 101 hingga 500.")
        st.write("- **High Spend**: Jumlah belanjaan antara 501 hingga 2000.")
        st.write("- **Uncategorized**: Jumlah belanjaan yang tidak masuk ke dalam kategori yang ditentukan.")

        st.write("**Kategori Harga Produk (PriceCategory)**: ")
        st.write("- **Cheap**: Harga per unit produk antara 25 hingga 100.")
        st.write("- **Moderate**: Harga per unit produk antara 101 hingga 300.")
        st.write("- **Expensive**: Harga per unit produk antara 301 hingga 500.")
        st.write("- **Uncategorized**: Harga per unit produk yang tidak masuk ke dalam kategori yang ditentukan.")

    elif page == "Distribusi":
        st.header("Distribusi Data")
        st.subheader("Distribusi Gender")
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='Gender', palette='coolwarm')
        plt.xlabel('Gender')
        plt.ylabel('Jumlah')
        plt.title('Distribusi Gender')
        st.pyplot()
        st.write("Berdasarkan data di atas, jumlah pelanggan pria dan wanita hampir seimbang dan tidak memiliki perbedaan yang signifikan.")

        # Visualisasi untuk kolom AgeGroup
        st.subheader("Distribusi Kelompok Umur")
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='AgeGroup', palette='coolwarm')
        plt.xlabel('Kelompok Umur')
        plt.ylabel('Jumlah')
        plt.title('Distribusi Kelompok Umur')
        st.pyplot()
        agegroup_percentage = df['AgeGroup'].value_counts(normalize=True) * 100
        st.write("Persentase Distribusi Kelompok Umur:")
        st.write(agegroup_percentage)

        st.write('distribusi umur pelanggan dengan mayoritas berada dalam kelompok Middle-Aged Adult (22.7%), diikuti oleh Adult (22.2%) dan Young Adult (21.4%). Kelompok Early Adult juga signifikan dengan persentase sebesar 19.1%, sedangkan kelompok Elderly Adult memiliki persentase terendah (14.6%).')
        st.write("Dengan mayoritas pelanggan berada dalam kelompok Middle-Aged Adult, strategi pemasaran dan promosi dapat difokuskan pada produk atau layanan yang menarik bagi segmen usia ini. Namun, perlu juga memperhatikan kelompok Adult dan Young Adult yang memiliki persentase yang hampir sama, sehingga upaya pemasaran dapat disesuaikan untuk menarik perhatian dari segmen usia ini. ")

    elif page == "Visualisasi Berdasarkan Total Transaksi":
        gender_spending_percentage = df.groupby('Gender')['Spending Type'].value_counts(normalize=True) * 100
        agegroup_spending_percentage = df.groupby('AgeGroup')['Spending Type'].value_counts(normalize=True) * 100

        st.subheader("Total Transaksi berdasarkan Gender")
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='Spending Type', hue='Gender', palette='coolwarm')
        plt.xlabel('Total Transaksi')
        plt.ylabel('Jumlah')
        plt.title('Total Transaksi berdasarkan Gender')
        st.pyplot()
        st.write("Persentase Total Transaksi berdasarkan Gender:")
        st.write(gender_spending_percentage)
        st.write("Data menunjukkan mayoritas pengeluaran pelanggan, baik pria maupun wanita, berada pada kategori Low Spend dengan persentase sekitar 45.1% untuk wanita dan 46.3% untuk pria. Meskipun ada sedikit perbedaan, pola pengeluaran secara umum relatif serupa di antara kedua kelompok gender. Sedangkan untuk kategori High Spend dan Moderate Spend, persentase pengeluaran relatif serupa di antara kedua kelompok gender, dengan tidak ada perbedaan yang signifikan.")

        st.subheader("Total Transaksi berdasarkan Kelompok Umur")
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='Spending Type', hue='AgeGroup', palette='coolwarm')
        plt.xlabel('Total Transaksi')
        plt.ylabel('Jumlah')
        plt.title('Total Transaksi berdasarkan Kelompok Umur')
        st.pyplot()
        st.write("Persentase Total Transaksi berdasarkan Kelompok Umur:")
        st.write(agegroup_spending_percentage)
        st.write("Kelompok Early Adult memiliki proporsi tertinggi dalam kategori 'Moderate Spend' (28.27%), menunjukkan kecenderungan untuk melakukan pembelian dengan nilai sedang.")
        st.write("Kelompok Middle-Aged Adult memiliki pola pengeluaran yang lebih merata antara kategori 'Low Spend', 'Moderate Spend', dan 'High Spend', dengan persentase yang relatif serupa di antara ketiganya.")
        st.write("Kelompok Elderly Adult cenderung memiliki proporsi tertinggi dalam kategori 'Low Spend' (49.32%), menunjukkan kecenderungan untuk melakukan pembelian dengan nilai rendah.")
        st.write("Kelompok Young Adult memiliki proporsi tertinggi dalam kategori 'High Spend' (33.64%), menunjukkan kecenderungan untuk melakukan pembelian dengan nilai tinggi.")
        st.write("Kelompok Adult memiliki proporsi tertinggi dalam kategori 'Low Spend' (50.90%), menunjukkan kecenderungan untuk melakukan pembelian dengan nilai rendah.")
        
        st.write("- Fokuskan promosi pada produk dengan nilai sedang untuk segmen yang lebih muda dan produk dengan nilai rendah untuk segmen yang lebih tua.")
        st.write("- Tingkatkan promosi untuk produk dengan nilai tinggi untuk menarik perhatian segmen yang lebih muda.")
        st.write("- Diversifikasi strategi pemasaran untuk mencakup semua kategori pengeluaran bagi segmen usia menengah.")
        st.write("- Gunakan promosi khusus untuk mendorong pembelian di kategori yang kurang diminati dalam setiap kelompok usia.")

    elif page == "Visualisasi Berdasarkan Produk Yang Dibeli":
        gender_percentage = df.groupby('Gender')['Product Category'].value_counts(normalize=True) * 100
        agegroup_percentage = df.groupby('AgeGroup')['Product Category'].value_counts(normalize=True) * 100

        st.subheader("Pembelian Produk berdasarkan Gender")
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='Product Category', hue='Gender', palette='coolwarm')
        plt.xlabel('Product Category')
        plt.ylabel('Count')
        plt.title('Pembelian Produk berdasarkan Gender')
        st.pyplot()
        st.write("Persentase Pembelian Produk berdasarkan Gender:")
        st.write(gender_percentage)
        st.write("- **Female**: Lebih cenderung untuk membeli produk Clothing (34.12%) dan Electronics (33.33%) dibandingkan dengan produk Beauty (32.55%).")
        st.write("- **Male**: Lebih cenderung untuk membeli produk Clothing (36.12%) dan Electronics (35.10%) dibandingkan dengan produk Beauty (28.78%).")
        st.write("Terjadi perbedaan preferensi dalam pembelian produk antara kedua gender, dengan pola yang lebih seimbang antara pembelian produk Clothing dan Electronics, namun terdapat perbedaan yang lebih signifikan dalam pembelian produk Beauty.")
        st.write("Untuk strategi pemasaran, fokuskan promosi pada produk Beauty untuk female dan pertimbangkan peningkatan promosi produk Clothing dan Electronics untuk male guna menarik lebih banyak pembeli.")
        # Visualisasi pembelian produk berdasarkan Age Group
        st.subheader("Pembelian Produk berdasarkan Kelompok Umur")
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='Product Category', hue='AgeGroup', palette='coolwarm')
        plt.xlabel('Kategori Produk')
        plt.ylabel('Jumlah')
        plt.title('Pembelian Produk berdasarkan Kelompok Umur')
        st.pyplot()
        st.write("Persentase Pembelian Produk berdasarkan Kelompok Umur:")
        st.write(agegroup_percentage)
        st.write("- **Adult**: Lebih cenderung membeli produk Clothing (34.68%) dan Electronics (33.33%) daripada produk Beauty (31.98%).")
        st.write("- **Early Adult**: Preferensi pembelian lebih merata antara produk Clothing (36.13%), Electronics (33.51%), dan Beauty (30.37%).")
        st.write("- **Elderly Adult**: Lebih cenderung membeli produk Clothing (37.67%) dan Electronics (33.56%) daripada produk Beauty (28.77%).")
        st.write("- **Middle-Aged Adult**: Lebih cenderung membeli produk Electronics (37.89%) dan Clothing (35.24%) daripada produk Beauty (26.87%).")
        st.write("- **Young Adult**: Lebih cenderung membeli produk Beauty (35.05%) dibandingkan dengan produk Clothing (32.71%) dan Electronics (32.24%).")
        st.write("- Fokuskan promosi pada produk Clothing dan Electronics untuk kelompok usia dewasa dan lanjut usia, sementara produk Beauty lebih diminati oleh kelompok usia muda.")
        st.write("- Pertimbangkan penyesuaian strategi pemasaran untuk menyesuaikan preferensi pembelian dari masing-masing kelompok umur guna meningkatkan penjualan.")

    elif page == "Clustering":
        st.subheader("Data dengan hasil klustering")
        kmeans_model, data_with_clusters = kmeans(x_final_norm)
        st.subheader("Visualisasi Klustering")
        visualize_kmeans_clustering(x_final_norm, kmeans_model)
        
        
        st.subheader("Analisis Klaster Pelanggan")

        st.write("### Cluster 0 (Pembeli Hemat pada Produk Kecantikan dan Pakaian):")
        st.write("Klaster ini mewakili segmen pasar yang cenderung sensitif terhadap harga dan berbelanja dengan hemat. Mereka menunjukkan preferensi yang konsisten terhadap produk kecantikan dan pakaian dengan harga yang terjangkau, namun kurang tertarik pada produk elektronik. Bisnis dapat menargetkan segmen ini dengan menawarkan diskon atau penawaran khusus untuk produk kecantikan dan pakaian yang terjangkau.")

        st.write("### Cluster 1 (Pembeli dengan Penghasilan Tinggi dan Preferensi Produk Elektronik Mahal):")
        st.write("Klaster ini mencakup pelanggan dengan penghasilan tinggi yang cenderung memilih produk elektronik dengan harga yang lebih mahal. Mereka menjadi target utama untuk produk-produk inovatif dan canggih di industri elektronik. Strategi pemasaran yang cocok untuk segmen ini mungkin melibatkan promosi eksklusif atau penawaran bundel untuk produk elektronik premium.")

        st.write("### Cluster 2 (Pembeli Rata-rata dengan Minat Luas):")
        st.write("Klaster ini terdiri dari pelanggan dengan belanjaan rata-rata dan minat yang beragam pada produk pakaian dan elektronik dengan harga yang sedang. Mereka mungkin menjadi target yang ideal untuk kampanye pemasaran yang menyoroti berbagai pilihan produk dengan harga terjangkau. Penawaran paket atau diskon lintas kategori dapat menarik perhatian segmen pasar ini.")

        st.write("### Cluster 3 (Pembeli Beragam dengan Preferensi Produk Elektronik dan Kecantikan):")
        st.write("Klaster ini mencakup pelanggan yang memiliki preferensi beragam terhadap produk elektronik dan kecantikan dengan belanjaan yang bervariasi. Mereka cenderung membeli produk dengan harga sedang hingga mahal. Bisnis dapat menargetkan segmen ini dengan menawarkan produk-produk inovatif dan eksklusif dalam kedua kategori tersebut, serta memperkuat kehadiran merek melalui strategi pemasaran yang terintegrasi.")



main()
