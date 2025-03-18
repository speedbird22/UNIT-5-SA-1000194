import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D

# Function to clean price and discount columns
def clean_currency(value):
    """Remove ₹ and % symbols and convert to float."""
    if isinstance(value, str):
        return float(value.replace("\u20b9", "").replace(",", "").replace("%", "").strip())
    return np.nan

# Load raw dataset
df = pd.read_csv("amazon.csv")

# Clean data
df["discounted_price"] = df["discounted_price"].apply(clean_currency)
df["actual_price"] = df["actual_price"].apply(clean_currency)
df["discount_percentage"] = df["discount_percentage"].apply(clean_currency)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["rating_count"] = df["rating_count"].replace(",", "", regex=True).astype(float)

# Fill missing values with median
for col in ["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]:
    df[col].fillna(df[col].median(), inplace=True)

# Ensure user_id and product_id are strings
df["user_id"] = df["user_id"].astype(str)
df["product_id"] = df["product_id"].astype(str)

# Streamlit App
st.title("Amazon E-Commerce Data Analysis")

# Dropdown menu for different analyses
option = st.sidebar.selectbox("Select Analysis", [
    "Exploratory Data Analysis (EDA)",
    "Customer Segmentation",
    "Association Rule Mining",
    "User Behavior Analysis"
])

if option == "Exploratory Data Analysis (EDA)":
    st.header("Exploratory Data Analysis")
    
    # Histograms for Prices
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["discounted_price"], bins=30, kde=True, ax=ax[0])
    ax[0].set_title("Discounted Price Distribution")
    sns.histplot(df["actual_price"], bins=30, kde=True, ax=ax[1])
    ax[1].set_title("Actual Price Distribution")
    st.pyplot(fig)
    
    # Box plots for prices
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=df["discounted_price"], ax=ax[0])
    ax[0].set_title("Discounted Price Box Plot")
    sns.boxplot(y=df["actual_price"], ax=ax[1])
    ax[1].set_title("Actual Price Box Plot")
    st.pyplot(fig)
    
    # Scatter plot for price relationships
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df["actual_price"], y=df["discounted_price"], hue=df["discount_percentage"], palette="coolwarm")
    ax.set_title("Actual vs Discounted Price")
    st.pyplot(fig)
    
    # 3D Scatter plot for Price and Discount Percentage
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df["actual_price"], df["discounted_price"], df["discount_percentage"], c=df["discount_percentage"], cmap="coolwarm")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Discounted Price")
    ax.set_zlabel("Discount Percentage")
    ax.set_title("3D Price Analysis")
    st.pyplot(fig)
    
    # Heatmap for correlations
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[["discounted_price", "actual_price", "rating", "rating_count"]].corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig)
    
elif option == "Customer Segmentation":
    st.header("Customer Segmentation")
    
    # Select features for clustering
    features = df[["discounted_price", "actual_price", "rating", "rating_count"]]
    kmeans = KMeans(n_clusters=3, random_state=42).fit(features)
    df["Cluster"] = kmeans.labels_
    
    # Scatter plot for clusters
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df["actual_price"], y=df["discounted_price"], hue=df["Cluster"], palette="viridis")
    ax.set_title("Customer Segments")
    st.pyplot(fig)
    
    # 3D Clustering visualization
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df["actual_price"], df["discounted_price"], df["rating"], c=df["Cluster"], cmap="viridis")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Discounted Price")
    ax.set_zlabel("Rating")
    ax.set_title("3D Customer Segmentation")
    st.pyplot(fig)
    
elif option == "Association Rule Mining":
    st.header("Association Rule Mining")
    
    # Prepare data for Apriori
    basket = df.pivot_table(index="user_id", columns="product_id", values="discounted_price", aggfunc="count").fillna(0)
    basket = (basket > 0).astype(int)  # Convert to 1/0 format
    
    frequent_items = apriori(basket, min_support=0.005, use_colnames=True)
    
    if frequent_items.empty:
        st.warning("No frequent itemsets found. Lowering min_support to 0.001")
        frequent_items = apriori(basket, min_support=0.001, use_colnames=True)
    
    if not frequent_items.empty:
        rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)
        st.write("Frequent Itemsets:")
        st.dataframe(frequent_items.sort_values(by="support", ascending=False).head(10))
        st.write("Top Association Rules:")
        st.dataframe(rules.sort_values(by="lift", ascending=False).head(10))
    else:
        st.warning("No frequent itemsets found. Try lowering the min_support value further.")
    
elif option == "User Behavior Analysis":
    st.header("User Behavior Analysis")
    
   
    
    # Table of most reviewed products
    st.write("Most Reviewed Products:")
    st.dataframe(df.sort_values(by="rating_count", ascending=False)[["product_id", "rating_count"]].head(10))
