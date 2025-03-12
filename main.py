import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
def load_data():
    df = pd.read_csv("amazon.csv")
    df.dropna(subset=["discounted_price", "rating", "rating_count"], inplace=True)
    df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors='coerce')
    df["rating"] = pd.to_numeric(df["rating"], errors='coerce')
    df["rating_count"] = pd.to_numeric(df["rating_count"], errors='coerce')
    df.dropna(inplace=True)
    return df

df = load_data()

# Sidebar Menu
st.sidebar.title("Amazon E-commerce Analysis")
menu = st.sidebar.radio("Select Analysis", ["Data Overview", "Exploratory Data Analysis", "Customer Segmentation", "Frequent Itemsets", "User Behavior Analysis"])

# Data Overview
if menu == "Data Overview":
    st.title("Data Overview")
    st.dataframe(df.head())
    st.write("Basic Statistics:")
    st.write(df.describe())

# EDA
elif menu == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    if not df.empty:
        fig, axes = plt.subplots(3, 1, figsize=(8, 15))
        sns.histplot(df["discounted_price"], bins=30, kde=True, ax=axes[0])
        axes[0].set_title("Discounted Price Distribution")
        
        sns.histplot(df["rating"], bins=30, kde=True, ax=axes[1])
        axes[1].set_title("Rating Distribution")
        
        sns.histplot(df["rating_count"], bins=30, kde=True, ax=axes[2])
        axes[2].set_title("Rating Count Distribution")
        
        st.pyplot(fig)
    else:
        st.warning("No data available for visualization.")

# Customer Segmentation
elif menu == "Customer Segmentation":
    st.title("Customer Segmentation")
    df_clean = df.dropna(subset=["discounted_price", "rating", "rating_count"])
    if not df_clean.empty:
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_clean["cluster"] = kmeans.fit_predict(df_clean[["discounted_price", "rating", "rating_count"]])
        st.write("Clustered Data:")
        st.dataframe(df_clean.head())
    else:
        st.warning("Not enough data for clustering.")

# Frequent Itemset Mining
elif menu == "Frequent Itemsets":
    st.title("Frequent Itemset Mining")
    if "user_id" in df.columns and "product_id" in df.columns:
        basket = df.groupby(["user_id", "product_id"]).size().unstack().fillna(0)
        frequent_items = apriori(basket, min_support=0.05, use_colnames=True)
        rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)
        st.write("Top Association Rules:")
        st.dataframe(rules.head())
    else:
        st.warning("Required columns for frequent itemset mining are missing.")

# User Behavior Analysis
elif menu == "User Behavior Analysis":
    st.title("User Behavior Analysis")
    if "user_id" in df.columns and "review_title" in df.columns and "review_content" in df.columns:
        st.write("Top Customer Reviews")
        st.dataframe(df[["user_id", "review_title", "review_content"]].dropna().head(10))
    else:
        st.warning("Required review data is missing.")
