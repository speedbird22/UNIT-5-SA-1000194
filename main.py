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

# EDA Functions
def eda_plots():
    option = st.selectbox("Choose EDA Analysis", ["Histograms & Box Plots", "Scatter Plots", "Bar Charts", "Pie Chart & Heatmap"])
    
    if option == "Histograms & Box Plots":
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        sns.histplot(df['actual_price'], bins=30, ax=ax[0, 0], kde=True)
        ax[0, 0].set_title("Actual Price Distribution")
        sns.histplot(df['discounted_price'], bins=30, ax=ax[0, 1], kde=True)
        ax[0, 1].set_title("Discounted Price Distribution")
        sns.boxplot(y=df['actual_price'], ax=ax[1, 0])
        ax[1, 0].set_title("Actual Price Boxplot")
        sns.boxplot(y=df['discounted_price'], ax=ax[1, 1])
        ax[1, 1].set_title("Discounted Price Boxplot")
        st.pyplot(fig)
    
    elif option == "Scatter Plots":
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=df['actual_price'], y=df['discounted_price'], hue=df['discounted_price'] / df['actual_price'])
        plt.title("Actual Price vs Discounted Price")
        st.pyplot(fig)
    
    elif option == "Bar Charts":
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.barplot(x=df['category'].value_counts().index, y=df['category'].value_counts().values, ax=ax[0])
        ax[0].set_title("Product Category Distribution")
        sns.barplot(x=df['rating'].value_counts().index, y=df['rating'].value_counts().values, ax=ax[1])
        ax[1].set_title("Product Rating Distribution")
        st.pyplot(fig)
    
    elif option == "Pie Chart & Heatmap":
        fig, ax = plt.subplots(figsize=(6, 6))
        df['category'].value_counts().plot.pie(autopct='%1.1f%%', cmap='viridis')
        plt.title("Category Distribution")
        st.pyplot(fig)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[['actual_price', 'discounted_price', 'rating', 'rating_count', 'category']].corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        st.pyplot(fig)

# Customer Segmentation
def customer_segmentation():
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[['actual_price', 'discounted_price', 'rating', 'rating_count']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['customer_segment'] = kmeans.fit_predict(df_scaled)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df['actual_price'], y=df['discounted_price'], hue=df['customer_segment'], palette='viridis')
    plt.title("Customer Segments")
    st.pyplot(fig)

# Association Rule Mining
def association_rule_mining():
    st.write("Applying Apriori Algorithm for Frequent Itemset Mining")
    df_encoded = pd.get_dummies(df[['product_id', 'category']])
    frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    st.write(rules.head())

# User Behavior Analysis
def user_behavior_analysis():
    st.write("User Behavior Analysis based on Reviews and Ratings")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['rating_count'], bins=30, kde=True)
    plt.title("User Engagement Distribution")
    st.pyplot(fig)

# Streamlit App
st.title("Amazon Data Analysis Dashboard")
menu = st.sidebar.selectbox("Choose Analysis", ["Exploratory Data Analysis", "Customer Segmentation", "Association Rule Mining", "User Behavior Analysis"])

if menu == "Exploratory Data Analysis":
    eda_plots()
elif menu == "Customer Segmentation":
    customer_segmentation()
elif menu == "Association Rule Mining":
    association_rule_mining()
elif menu == "User Behavior Analysis":
    user_behavior_analysis()
