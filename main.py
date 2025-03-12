import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset with proper cleaning
@st.cache_data
def load_data():
    df = pd.read_csv("amazon.csv")

    # Ensure numerical columns are cleaned
    for col in ["discounted_price", "rating", "rating_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["discounted_price", "rating", "rating_count"], inplace=True)
    return df

df = load_data()

# Sidebar Menu
st.sidebar.title("Amazon E-commerce Analysis")
menu = st.sidebar.radio("Select Analysis", [
    "Data Overview", "Exploratory Data Analysis", "Customer Segmentation",
    "Frequent Itemsets", "User Behavior Analysis"
])

# Data Overview
if menu == "Data Overview":
    st.title("Data Overview")
    st.dataframe(df.head())
    st.write("Basic Statistics:")
    st.write(df.describe())

# EDA
elif menu == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")

    # Price Distribution
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["discounted_price"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    # Ratings Distribution
    st.subheader("Ratings Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["rating"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

# Customer Segmentation
elif menu == "Customer Segmentation":
    st.title("Customer Segmentation")

    # Ensure valid clustering
    df_clean = df[["discounted_price", "rating", "rating_count"]].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clean["cluster"] = kmeans.fit_predict(df_clean)

    st.write("Clustered Data:")
    st.dataframe(df_clean.head())

# Frequent Itemset Mining
elif menu == "Frequent Itemsets":
    st.title("Frequent Itemset Mining")

    # Convert user_id to categorical if available
    if "user_id" in df.columns:
        basket = df.groupby(["user_id", "product_id"]).size().unstack().fillna(0)
        frequent_items = apriori(basket, min_support=0.05, use_colnames=True)
        rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)
        st.write("Top Association Rules:")
        st.dataframe(rules.head())
    else:
        st.warning("User IDs are missing. Unable to perform itemset mining.")

# User Behavior Analysis
elif menu == "User Behavior Analysis":
    st.title("User Behavior Analysis")

    if "review_title" in df.columns and "review_content" in df.columns:
        st.write("Top Customer Reviews")
        st.dataframe(df[["user_id", "review_title", "review_content"]].dropna().head(10))
    else:
        st.warning("Review data is missing in the dataset.")
