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
def run_association_rule_mining():
    df_basket = df[['product_id', 'category']]
    df_basket['category'] = df_basket['category'].astype(str)
    df_basket = pd.crosstab(df_basket['product_id'], df_basket['category'])
    
    if df_basket.shape[0] == 0 or df_basket.shape[1] == 0:
        st.write("No transactions available for Association Rule Mining.")
        return
    
    frequent_itemsets = apriori(df_basket, min_support=0.01, use_colnames=True)
    
    if frequent_itemsets.empty:
        st.write("No frequent itemsets found. Try reducing min_support further.")
    else:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
        
        if rules.empty:
            st.write("No strong association rules found. Try lowering the lift threshold.")
        else:
            st.write("Top 5 Association Rules")
            st.write(rules.head())
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x=rules['support'], y=rules['confidence'], hue=rules['lift'], size=rules['lift'], palette='coolwarm')
            plt.title("Support vs Confidence with Lift")
            plt.xlabel("Support")
            plt.ylabel("Confidence")
            st.pyplot(fig)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(rules['lift'], bins=20, kde=True, color='blue')
            plt.title("Distribution of Lift Values")
            plt.xlabel("Lift")
            st.pyplot(fig)

# Streamlit App
st.title("Amazon Data Analysis Dashboard")
analysis_type = st.sidebar.selectbox("Choose Analysis", ["Association Rule Mining"])

if analysis_type == "Association Rule Mining":
    run_association_rule_mining()
