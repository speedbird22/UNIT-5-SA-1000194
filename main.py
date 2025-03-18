import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from mlxtend.frequent_patterns import apriori, association_rules

# Load Data
file_path = 'amazon.csv'
df = pd.read_csv(file_path)
st.write(df.head())  # Debugging to check if data loads correctly

# Data Cleaning
original_size = df.shape[0]  # Store original dataset size before cleaning
df.dropna(subset=['product_id', 'actual_price', 'discounted_price', 'rating', 'rating_count'], inplace=True)
df.fillna(method='ffill', inplace=True)
df['actual_price'] = pd.to_numeric(df['actual_price'].replace('[^\d.]', '', regex=True), errors='coerce')
df['discounted_price'] = pd.to_numeric(df['discounted_price'].replace('[^\d.]', '', regex=True), errors='coerce')
df['actual_price'].replace(0, np.nan, inplace=True)
df['discounted_price'].replace(0, np.nan, inplace=True)
df['actual_price'].fillna(df['actual_price'].median(), inplace=True)
df['discounted_price'].fillna(df['discounted_price'].median(), inplace=True)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating'].replace(0, np.nan, inplace=True)
df['rating'].fillna(df['rating'].median(), inplace=True)
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
df['rating_count'].replace(0, np.nan, inplace=True)
df['rating_count'].fillna(df['rating_count'].median(), inplace=True)
df['rating_count'] = df['rating_count'].astype(int)

# Verify data cleaning did not remove excessive rows
final_size = df.shape[0]
st.write(f"Rows before cleaning: {original_size}, Rows after cleaning: {final_size}")

# Encoding Categorical Features
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'].astype(str))

# Fixing Graph Rendering Issue
def plot_histograms():
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df['actual_price'], bins=30, ax=ax[0], kde=True)
    ax[0].set_title("Actual Price Distribution")
    sns.histplot(df['discounted_price'], bins=30, ax=ax[1], kde=True)
    ax[1].set_title("Discounted Price Distribution")
    fig.tight_layout()  # Ensure proper spacing
    st.pyplot(fig)

# Fixing Association Rule Mining
def association_rule_mining():
    st.write("Applying Apriori Algorithm for Frequent Itemset Mining")
    
    # Convert product_id and category into a transactional format
    df_basket = df.pivot_table(index='product_id', columns='category', aggfunc='size', fill_value=0)
    df_basket = df_basket.applymap(lambda x: 1 if x > 0 else 0)  # Convert to binary format

    # Ensure there are valid transactions
    if df_basket.shape[0] == 0:
        st.write("No transactions available for Apriori. Try adjusting min_support.")
        return

    # Apply Apriori Algorithm
    frequent_itemsets = apriori(df_basket, min_support=0.01, use_colnames=True)
    if frequent_itemsets.empty:
        st.write("No frequent itemsets found. Try lowering the min_support value.")
        return

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    if rules.empty:
        st.write("No association rules generated. Adjust min_threshold or check dataset.")
    else:
        st.write(rules.head())

# Streamlit App
st.title("Amazon Data Analysis Dashboard")
menu = st.sidebar.selectbox("Choose Analysis", ["Exploratory Data Analysis", "Customer Segmentation", "Association Rule Mining", "User Behavior Analysis"])

if menu == "Exploratory Data Analysis":
    plot_histograms()
elif menu == "Association Rule Mining":
    association_rule_mining()
