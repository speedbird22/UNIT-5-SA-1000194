import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from itertools import combinations

# Load dataset
def load_data():
    df = pd.read_csv("amazon.csv")
    return df

df = load_data()

# Data Preprocessing
st.title("Amazon E-Commerce Data Analysis Dashboard")
st.sidebar.header("Navigation")
option = st.sidebar.radio("Choose Analysis", [
    "Data Overview", "EDA", "Customer Segmentation", "Association Rule Mining", 
    "Price Prediction", "Product Recommendation", "User Behavior Analysis"
])

# Encode categorical variables
le = LabelEncoder()
df["category_encoded"] = le.fit_transform(df["category"].astype(str))

def plot_eda():
    st.subheader("Exploratory Data Analysis (EDA)")
    
    # Histograms
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["actual_price"], bins=30, kde=True, ax=ax[0])
    ax[0].set_title("Distribution of Actual Prices")
    sns.histplot(df["discounted_price"], bins=30, kde=True, ax=ax[1])
    ax[1].set_title("Distribution of Discounted Prices")
    st.pyplot(fig)
    
    # Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df["actual_price"], y=df["discounted_price"], hue=df["discount_percentage"], palette="coolwarm")
    ax.set_title("Price Comparison")
    st.pyplot(fig)
    
if option == "EDA":
    plot_eda()

# Customer Segmentation
if option == "Customer Segmentation":
    st.subheader("Customer Segmentation")
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[["discounted_price", "rating", "rating_count"]].dropna())
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df["discounted_price"], y=df["rating"], hue=df["cluster"], palette="viridis")
    ax.set_title("Customer Segments")
    st.pyplot(fig)

# Association Rule Mining
if option == "Association Rule Mining":
    st.subheader("Frequent Itemsets")
    df_exploded = df.assign(category=df["category"].str.split("|")).explode("category")
    transactions = df_exploded.groupby("product_id")["category"].apply(list)
    pair_counts = Counter()
    for items in transactions:
        for pair in combinations(set(items), 2):
            pair_counts[pair] += 1
    pair_df = pd.DataFrame(pair_counts.items(), columns=["Item Pair", "Count"]).sort_values(by="Count", ascending=False)
    st.dataframe(pair_df.head(10))

# Price Prediction Model
if option == "Price Prediction":
    st.subheader("Price Prediction Model")
    X = df[["actual_price", "discount_percentage", "rating", "rating_count", "category_encoded"]].dropna()
    y = df["discounted_price"].loc[X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Recommendation System
if option == "Product Recommendation":
    st.subheader("Product Recommendation")
    ratings_matrix = df.pivot_table(index="product_id", columns="category", values="rating").fillna(0)
    similarity_matrix = cosine_similarity(ratings_matrix)
    product_similarity_df = pd.DataFrame(similarity_matrix, index=ratings_matrix.index, columns=ratings_matrix.index)
    
    def get_recommendations(product_id, n=5):
        if product_id in product_similarity_df:
            similar_products = product_similarity_df[product_id].sort_values(ascending=False).iloc[1 : n + 1]
            return df[df["product_id"].isin(similar_products.index)][["product_id", "product_name"]]
        else:
            return "Product ID not found."
    
    product_id = st.text_input("Enter Product ID:")
    if product_id:
        st.dataframe(get_recommendations(product_id))

# User Behavior Analysis
if option == "User Behavior Analysis":
    st.subheader("User Behavior Analysis")
    st.write("Understanding customer reviews and ratings trends.")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df["rating"])
    ax.set_title("Rating Distribution")
    st.pyplot(fig)
