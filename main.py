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

# Data Cleaning - Ensure no data is deleted
df.fillna(df.select_dtypes(include=[np.number]).median(), inplace=True)
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'].astype(str))

# Exploratory Data Analysis (EDA)
def plot_histograms_boxplots():
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(df['actual_price'], bins=30, ax=ax[0, 0], kde=True)
    ax[0, 0].set_title("Actual Price Distribution")
    sns.histplot(df['discounted_price'], bins=30, ax=ax[0, 1], kde=True)
    ax[0, 1].set_title("Discounted Price Distribution")
    sns.boxplot(x=df['actual_price'], ax=ax[1, 0])
    ax[1, 0].set_title("Actual Price Box Plot")
    sns.boxplot(x=df['discounted_price'], ax=ax[1, 1])
    ax[1, 1].set_title("Discounted Price Box Plot")
    st.pyplot(fig)

def plot_scatter():
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df['actual_price'], y=df['discounted_price'], hue=df['discounted_price'] / df['actual_price'])
    plt.title("Actual Price vs Discounted Price")
    st.pyplot(fig)

def plot_rating_analysis():
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x=df['rating'].value_counts().index, y=df['rating'].value_counts(), ax=ax[0])
    ax[0].set_title("Rating Distribution")
    sns.histplot(df['rating_count'], bins=30, ax=ax[1], kde=True)
    ax[1].set_title("Rating Count Distribution")
    st.pyplot(fig)

def plot_category_analysis():
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x=df['category'].value_counts().index, y=df['category'].value_counts(), ax=ax[0])
    ax[0].set_title("Category Distribution")
    df['category'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1])
    ax[1].set_title("Category Distribution (Pie Chart)")
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
option = st.sidebar.selectbox("Choose Analysis", [
    "Exploratory Data Analysis (EDA)",
    "Customer Segmentation",
    "Association Rule Mining",
    "User Behavior Analysis",
])

if option == "Exploratory Data Analysis (EDA)":
    analysis = st.selectbox("Select EDA Analysis", [
        "Histograms & Box Plots", "Scatter Plots", "Rating Analysis", "Category Analysis", "Correlation Heatmap"
    ])
    if analysis == "Histograms & Box Plots":
        plot_histograms_boxplots()
    elif analysis == "Scatter Plots":
        plot_scatter()
    elif analysis == "Rating Analysis":
        plot_rating_analysis()
    elif analysis == "Category Analysis":
        plot_category_analysis()
    elif analysis == "Correlation Heatmap":
        plot_correlation()

elif option == "Customer Segmentation":
    segmentation_analysis = st.selectbox("Select Segmentation Analysis", ["Customer Segments", "3D Visualization"])
    if segmentation_analysis == "Customer Segments":
        plot_segments()
    elif segmentation_analysis == "3D Visualization":
        plot_3d_graph()

# Placeholder for Association Rule Mining and User Behavior Analysis
elif option == "Association Rule Mining":
    st.write("Coming soon: Association Rule Mining using Apriori Algorithm!")
elif option == "User Behavior Analysis":
    st.write("Coming soon: User behavior insights from reviews and ratings!")
