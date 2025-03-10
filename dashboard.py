import streamlit as st
import pandas as pd
import os
import gdown
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

# Folder penyimpanan dataset
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Mapping nama file dengan ID Google Drive
FILE_ID_MAPPING = {
    "customers_dataset.csv": "14G37l0ArGjiDr2plpnW4UCP0ilshUIni",
    "geolocation_dataset.csv": "1KvKZoRMumHoPxxWRMPccT51kACo5ephB",
    "order_items_dataset.csv": "1Zz0gmCD0TVCkmgCf4j_U65lQBrYKsVh0",
    "order_payments_dataset.csv": "1Sc8mo0RClVDYtlMJVcsjIsuUBXVVGYQl",
    "order_reviews_dataset.csv": "1Ezzwn61F3obD487vYRj1ymqfTRjnBXtv",
    "orders_dataset.csv": "1vA_6o4QZYQvT1J-xKqNuRhRJCih98gTH",
    "product_category_name_translation.csv": "1de3buGiTGhj9DuAznOwN11jOqRm5uRJc",
    "products_dataset.csv": "1jv1Z9Ry4n4aJouVWYipW7xZlm-_RIcP6",
    "sellers_dataset.csv": "1DFNRi_4t_TomZwsEgMiVIEqRSq_f5okL"
}

# Fungsi untuk mengunduh dataset dari Google Drive
def download_from_drive(filename):
    file_path = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(file_path):
        url = f"https://drive.google.com/uc?id={FILE_ID_MAPPING[filename]}"
        gdown.download(url, file_path, quiet=False)
    return file_path

# Fungsi untuk memuat data dengan caching
@st.cache_data
def load_data():
    return {
        "customers": pd.read_csv(download_from_drive('customers_dataset.csv')),
        "geolocation": pd.read_csv(download_from_drive('geolocation_dataset.csv')),
        "order_items": pd.read_csv(download_from_drive('order_items_dataset.csv'), on_bad_lines='skip'),
        "order_payments": pd.read_csv(download_from_drive('order_payments_dataset.csv')),
        "order_reviews": pd.read_csv(download_from_drive('order_reviews_dataset.csv')),
        "orders": pd.read_csv(download_from_drive('orders_dataset.csv'), parse_dates=['order_purchase_timestamp', 'order_delivered_customer_date']),
        "product_translations": pd.read_csv(download_from_drive('product_category_name_translation.csv')),
        "products": pd.read_csv(download_from_drive('products_dataset.csv')),
        "sellers": pd.read_csv(download_from_drive('sellers_dataset.csv'))
    }

# Load data
data = load_data()

# Streamlit UI
st.title("E-Commerce Data Analysis Dashboard")

menu = st.sidebar.selectbox("Select Analysis", ["Overview", "Order & Customer Analysis", "Product & Sales Analysis", "Geographical Insights", "Customer Segmentation"])

if menu == "Overview":
    st.header("Dataset Overview")
    st.write("### Sample Data")
    st.write("Customers Dataset", data["customers"].head())
    st.write("Orders Dataset", data["orders"].head())
    st.write("Products Dataset", data["products"].head())

elif menu == "Order & Customer Analysis":
    st.header("Order & Customer Analysis")

    # Filter berdasarkan jumlah transaksi pelanggan
    order_threshold = st.slider("Select minimum number of orders", 1, 10, 1)
    customer_transactions = data["orders"].groupby('customer_id')['order_id'].count().reset_index()
    customer_transactions.columns = ['customer_id', 'total_orders']
    customer_transactions = customer_transactions[customer_transactions['total_orders'] >= order_threshold]

    bins = [0, 1, 3, 10, float('inf')]
    labels = ['Low-value', 'Medium-value', 'High-value', 'Premium']
    customer_transactions['customer_segment'] = pd.cut(customer_transactions['total_orders'], bins=bins, labels=labels, right=False)

    st.write("### Customer Segments")
    st.bar_chart(customer_transactions['customer_segment'].value_counts())

elif menu == "Product & Sales Analysis":
    st.header("Product & Sales Analysis")

    # Pilih jumlah top produk yang ingin ditampilkan
    top_n = st.slider("Select number of top-selling products", 5, 20, 10)
    top_products = data["order_items"].groupby('product_id').size().reset_index(name='count').sort_values(by='count', ascending=False).head(top_n)

    st.write("### Top Selling Products")
    st.bar_chart(top_products.set_index('product_id'))

    payment_counts = data["order_payments"]['payment_type'].value_counts()
    st.write("### Payment Methods Distribution")
    st.bar_chart(payment_counts)

elif menu == "Geographical Insights":
    st.header("Geographical Insights")

    geolocation_df = data["geolocation"].groupby(['geolocation_lat', 'geolocation_lng']).size().reset_index(name='count')
    world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    world.plot(ax=ax, color='black')
    ax.scatter(geolocation_df['geolocation_lng'], geolocation_df['geolocation_lat'], c=geolocation_df['count'], cmap='Reds', alpha=0.9)
    plt.title("Transaction Density by Location")
    
    st.pyplot(fig)

elif menu == "Customer Segmentation":
    st.header("Customer Segmentation using RFM Analysis")

    latest_date = data["orders"]['order_purchase_timestamp'].max()
    rfm = data["orders"].groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (latest_date - x.max()).days,
        'order_id': 'count'
    }).reset_index()

    monetary = data["order_items"].groupby('order_id')['price'].sum().reset_index()
    orders_with_monetary = data["orders"][['customer_id', 'order_id']].merge(monetary, on='order_id', how='left')
    monetary_per_customer = orders_with_monetary.groupby('customer_id')['price'].sum().reset_index()
    rfm = rfm.merge(monetary_per_customer, on='customer_id', how='left')
    rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(rfm['Recency'], bins=30, kde=True, color='blue', ax=ax[0])
    ax[0].set_title('Recency Distribution')
    sns.histplot(rfm['Frequency'], bins=30, kde=True, color='green', ax=ax[1])
    ax[1].set_title('Frequency Distribution')
    sns.histplot(rfm['Monetary'], bins=30, kde=True, color='red', ax=ax[2])
    ax[2].set_title('Monetary Distribution')

    st.pyplot(fig)
