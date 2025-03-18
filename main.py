import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mpl_toolkits.mplot3d import Axes3D

# Load Data
file_path = 'amazon.csv'
df = pd.read_csv(file_path)

# Data Cleaning (Ensuring no data is lost)
df.fillna(method='ffill', inplace=True)
df['actual_price'] = pd.to_numeric(df['actual_price'].replace('[^\d.]', '', regex=True), errors='coerce')
df['discounted_price'] = pd.to_numeric(df['discounted_price'].replace('[^\d.]', '', regex=True), errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
df.fillna(df.median(), inplace=True)
df['rating_count'] = df['rating_count'].astype(int)
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'].astype(str))

# EDA Functions
def plot_price_distribution():
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df['actual_price'], bins=30, ax=ax[0], kde=True)
    ax[0].set_title("Actual Price Distribution")
    sns.histplot(df['discounted_price'], bins=30, ax=ax[1], kde=True)
    ax[1].set_title("Discounted Price Distribution")
    st.pyplot(fig)

def plot_box_plots():
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=df['actual_price'], ax=ax[0])
    ax[0].set_title("Actual Price Boxplot")
    sns.boxplot(y=df['discounted_price'], ax=ax[1])
    ax[1].set_title("Discounted Price Boxplot")
    st.pyplot(fig)

def plot_scatter():
    df['discount_percentage'] = (df['actual_price'] - df['discounted_price']) / df['actual_price'] * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df['actual_price'], y=df['discounted_price'], hue=df['discount_percentage'])
    plt.title("Actual Price vs Discounted Price")
    st.pyplot(fig)

def plot_rating_distribution():
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=df['rating'].value_counts().index, y=df['rating'].value_counts().values)
    plt.title("Product Rating Distribution")
    st.pyplot(fig)

def plot_category_distribution():
    fig, ax = plt.subplots(figsize=(8, 5))
    df['category'].value_counts().plot(kind='bar')
    plt.title("Product Category Distribution")
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

# Association Rule Mining
def run_association_rule_mining():
    basket = df.groupby(['product_id', 'category']).size().unstack().fillna(0)
    frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
    st.write("Association Rules:")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Streamlit App
st.title("Amazon Data Analysis Dashboard")
menu = st.sidebar.selectbox("Choose Analysis", ["Exploratory Data Analysis", "Customer Segmentation", "Association Rule Mining", "User Behavior Analysis", "Correlation Analysis"])

if menu == "Exploratory Data Analysis":
    option = st.selectbox("EDA Options", ["Price Distribution", "Box Plots", "Scatter Plot", "Rating Distribution", "Category Distribution"])
    if option == "Price Distribution":
        plot_price_distribution()
    elif option == "Box Plots":
        plot_box_plots()
    elif option == "Scatter Plot":
        plot_scatter()
    elif option == "Rating Distribution":
        plot_rating_distribution()
    elif option == "Category Distribution":
        plot_category_distribution()

elif menu == "Customer Segmentation":
    option = st.selectbox("Segmentation Options", ["Customer Segments", "3D Visualization"])
    if option == "Customer Segments":
        plot_segments()
    elif option == "3D Visualization":
        plot_3d_graph()

elif menu == "Association Rule Mining":
    run_association_rule_mining()

elif menu == "Correlation Analysis":
    plot_correlation()
