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

# Data Cleaning
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

# Encoding Categorical Features
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'].astype(str))

# Association Rule Mining
def association_rule_mining():
    st.write("Applying Apriori Algorithm for Frequent Itemset Mining")
    
    # Convert product_id to string and create a transaction format
    df_basket = df.groupby(['product_id', 'category']).size().unstack(fill_value=0)
    df_basket = df_basket.applymap(lambda x: 1 if x > 0 else 0)  # Convert to binary format

    # Apply Apriori Algorithm
    frequent_itemsets = apriori(df_basket, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    # Display top 5 rules
    st.write(rules.head())

# Streamlit App
st.title("Amazon Data Analysis Dashboard")
menu = st.sidebar.selectbox("Choose Analysis", ["Exploratory Data Analysis", "Customer Segmentation", "Association Rule Mining", "User Behavior Analysis"])

if menu == "Association Rule Mining":
    association_rule_mining()
