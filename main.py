import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans

# Load your dataset
@st.cache_data
def load_data():
    df = pd.read_csv("amazon.csv")  # Ensure the correct file path
    return df

df = load_data()

st.title("Amazon Market Basket Analysis")

# Dropdown menu
option = st.selectbox("Select an analysis section", [
    "EDA", "Clustering", "Association", "Behavior Analysis", "Customer Segments", "Frequently Bought Together Items", "Insights from Customer Behavior"
])

if option == "EDA":
    st.header("Exploratory Data Analysis (EDA)")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df['discounted_price'], bins=30, kde=True, ax=ax[0])
    ax[0].set_title("Distribution of Discounted Prices")
    
    if 'product_category' in df.columns:
        sns.boxplot(x=df['product_category'], y=df['discounted_price'], ax=ax[1])
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
        ax[1].set_title("Price Distribution Across Categories")
    else:
        st.error("Column 'product_category' not found in the dataset!")
    
    corr = df[['discounted_price', 'actual_price', 'rating', 'rating_count']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax[2])
    ax[2].set_title("Correlation Heatmap")
    
    st.pyplot(fig)
    st.dataframe(df.describe())

if option == "Clustering":
    st.header("Customer Segmentation with Clustering")
    num_clusters = 4
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['discounted_price', 'rating_count']])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df['discounted_price'], y=df['rating_count'], hue=df['cluster'], palette='viridis')
    ax.set_title("Customer Segmentation Based on Price & Rating Count")
    st.pyplot(fig)
    
    st.dataframe(df.groupby('cluster').mean())

if option == "Association":
    st.header("Association Rule Mining")
    df_encoded = pd.get_dummies(df[['product_category', 'product_name']])
    frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

if option == "Behavior Analysis":
    st.header("User Behavior Analysis")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x='rating', y='rating_count', data=df, ci=None, ax=ax[0])
    ax[0].set_title("Ratings vs. Review Count")
    df['rating'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1])
    ax[1].set_title("Rating Distribution Pie Chart")
    
    st.pyplot(fig)
    st.dataframe(df[['user_id', 'review_title', 'review_content']].head(10))

if option == "Customer Segments":
    st.header("Customer Segmentation")
    if 'product_category' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=df['product_category'], y=df['discounted_price'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title("Product Categories vs. Discounted Price")
        st.pyplot(fig)
        st.dataframe(df[['product_category', 'discounted_price']].groupby('product_category').mean())
    else:
        st.error("Column 'product_category' not found in the dataset!")

if option == "Frequently Bought Together Items":
    st.header("Frequently Bought Together Items")
    frequent_pairs = rules[['antecedents', 'consequents', 'lift']].sort_values(by='lift', ascending=False).head(10)
    st.dataframe(frequent_pairs)

if option == "Insights from Customer Behavior":
    st.header("Insights from Customer Behavior")
    top_reviewed = df[['product_name', 'rating_count']].sort_values(by='rating_count', ascending=False).head(10)
    st.dataframe(top_reviewed)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['rating'], bins=5, kde=True)
    ax.set_title("Distribution of Ratings")
    st.pyplot(fig)

# Run with: streamlit run script_name.py
