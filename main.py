import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

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

# EDA Functions
def plot_histograms():
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df['actual_price'], bins=30, ax=ax[0], kde=True)
    ax[0].set_title("Actual Price Distribution")
    sns.histplot(df['discounted_price'], bins=30, ax=ax[1], kde=True)
    ax[1].set_title("Discounted Price Distribution")
    st.pyplot(fig)

def plot_scatter():
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df['actual_price'], y=df['discounted_price'], hue=df['discounted_price'] / df['actual_price'])
    plt.title("Actual Price vs Discounted Price")
    st.pyplot(fig)

def plot_correlation():
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df[['actual_price', 'discounted_price', 'rating', 'rating_count', 'category']].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    st.pyplot(fig)

# Customer Segmentation
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['actual_price', 'discounted_price', 'rating', 'rating_count']])
kmeans = KMeans(n_clusters=3, random_state=42)
df['customer_segment'] = kmeans.fit_predict(df_scaled)

def plot_segments():
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df['actual_price'], y=df['discounted_price'], hue=df['customer_segment'], palette='viridis')
    plt.title("Customer Segments")
    st.pyplot(fig)

def plot_3d_graph():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['actual_price'], df['discounted_price'], df['rating'], c=df['customer_segment'], cmap='viridis')
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Discounted Price')
    ax.set_zlabel('Rating')
    ax.set_title("3D Visualization of Product Data")
    st.pyplot(fig)

# Streamlit App
st.title("Amazon Data Analysis Dashboard")
option = st.sidebar.selectbox("Choose Analysis", ["Histograms", "Scatter Plot", "Correlation Heatmap", "Customer Segmentation", "3D Visualization"])

if option == "Histograms":
    plot_histograms()
elif option == "Scatter Plot":
    plot_scatter()
elif option == "Correlation Heatmap":
    plot_correlation()
elif option == "Customer Segmentation":
    plot_segments()
elif option == "3D Visualization":
    plot_3d_graph()
