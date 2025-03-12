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
    st.write("### Raw Data Preview")
    st.write(df.head(10))
    
    required_columns = ["discounted_price", "rating", "rating_count"]
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            return pd.DataFrame()
    
    df[required_columns] = df[required_columns].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=required_columns, inplace=True)
    return df

df = load_data()

# Sidebar Menu
st.sidebar.title("Amazon E-commerce Analysis")
menu = st.sidebar.radio("Select Analysis", ["Data Overview", "Exploratory Data Analysis", "Customer Segmentation", "Frequent Itemsets", "User Behavior Analysis"])

# Data Overview
if menu == "Data Overview":
    st.title("Data Overview")
    if df.empty:
        st.warning("No data available.")
    else:
        st.dataframe(df.head())
        st.write("Basic Statistics:")
        st.write(df.describe())

# EDA
elif menu == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    if df.empty:
        st.warning("No data available for visualization.")
    else:
        fig, axes = plt.subplots(3, 1, figsize=(8, 15))
        sns.histplot(df["discounted_price"].dropna(), bins=30, kde=True, ax=axes[0])
        axes[0].set_title("Discounted Price Distribution")
        
        sns.histplot(df["rating"].dropna(), bins=30, kde=True, ax=axes[1])
        axes[1].set_title("Rating Distribution")
        
        sns.histplot(df["rating_count"].dropna(), bins=30, kde=True, ax=axes[2])
        axes[2].set_title("Rating Count Distribution")
        
        st.pyplot(fig)

# Customer Segmentation
elif menu == "Customer Segmentation":
    st.title("Customer Segmentation")
    if df.empty:
        st.warning("No data available for clustering.")
    else:
        try:
            kmeans = KMeans(n_clusters=3, random_state=42)
            df = df.dropna(subset=["discounted_price", "rating", "rating_count"])
            df["cluster"] = kmeans.fit_predict(df[["discounted_price", "rating", "rating_count"]])
            st.write("Clustered Data:")
            st.dataframe(df[["discounted_price", "rating", "rating_count", "cluster"].head()])
        except Exception as e:
            st.error(f"Error in clustering: {e}")

# Frequent Itemset Mining
elif menu == "Frequent Itemsets":
    st.title("Frequent Itemset Mining")
    if "user_id" in df.columns and "product_id" in df.columns:
        basket = df.groupby(["user_id", "product_id"]).size().unstack().fillna(0)
        if not basket.empty:
            frequent_items = apriori(basket, min_support=0.05, use_colnames=True)
            rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)
            if not rules.empty:
                st.write("Top Association Rules:")
                st.dataframe(rules.head())
            else:
                st.warning("No significant association rules found.")
        else:
            st.warning("No frequent itemsets found.")
    else:
        st.warning("Required columns for frequent itemset mining are missing.")

# User Behavior Analysis
elif menu == "User Behavior Analysis":
    st.title("User Behavior Analysis")
    if "user_id" in df.columns and "review_title" in df.columns and "review_content" in df.columns:
        reviews = df[["user_id", "review_title", "review_content"]].dropna()
        if not reviews.empty:
            st.write("Top Customer Reviews")
            st.dataframe(reviews.head(10))
        else:
            st.warning("No review data available.")
    else:
        st.warning("Required review data is missing.")
