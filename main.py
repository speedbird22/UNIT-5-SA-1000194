import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# Load and clean dataset
def load_and_clean_data():
    file_path = "amazon.csv"
    df = pd.read_csv(file_path)
    
    # Convert price and rating columns to numeric, forcing errors to NaN
    df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors='coerce')
    df["actual_price"] = pd.to_numeric(df["actual_price"], errors='coerce')
    df["rating"] = pd.to_numeric(df["rating"], errors='coerce')
    df["rating_count"] = pd.to_numeric(df["rating_count"], errors='coerce')
    
    # Data Cleaning
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["discounted_price", "actual_price", "rating", "rating_count", "category", "product_id"], inplace=True)
    df = df[df["discounted_price"] > 0]
    df = df[df["actual_price"] >= df["discounted_price"]]
    
    return df

df = load_and_clean_data()

# Streamlit App
st.title("Amazon E-Commerce Data Analysis")

# Sidebar Menu
menu = st.sidebar.selectbox("Select Analysis Type", [
    "Exploratory Data Analysis (EDA)", "Customer Segmentation",
    "Association Rule Mining", "User Behavior Analysis"
])

if menu == "Exploratory Data Analysis (EDA)":
    st.subheader("Exploratory Data Analysis")
    
    # Histograms and Boxplots
    st.write("### Distribution of Product Prices")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["discounted_price"].dropna(), bins=30, kde=True, ax=ax[0])
    ax[0].set_title("Discounted Price Distribution")
    sns.boxplot(x=df["actual_price"].dropna(), ax=ax[1])
    ax[1].set_title("Actual Price Boxplot")
    st.pyplot(fig)
    
    # Scatter Plots
    st.write("### Price Comparison")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df["actual_price"], y=df["discounted_price"], hue=df["discounted_price"] - df["actual_price"], palette="coolwarm")
    plt.title("Actual Price vs Discounted Price")
    st.pyplot(fig)
    
    # Rating Distribution
    st.write("### Product Ratings Distribution")
    rating_counts = df["rating"].value_counts().reset_index()
    rating_counts.columns = ["Rating", "Count"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=rating_counts["Rating"], y=rating_counts["Count"])
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.title("Ratings Count Distribution")
    st.pyplot(fig)
    
    # Heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df[["discounted_price", "actual_price", "rating", "rating_count"]].corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig)

elif menu == "Customer Segmentation":
    st.subheader("Customer Segmentation Using K-Means")
    
    # Select relevant features and drop NaN
    cluster_df = df[["discounted_price", "actual_price", "rating", "rating_count"]].dropna().copy()
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_df["Cluster"] = kmeans.fit_predict(cluster_df)
    
    # Visualization
    st.write("### Clusters based on Pricing and Ratings")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=cluster_df["actual_price"], y=cluster_df["discounted_price"], hue=cluster_df["Cluster"], palette="Set1")
    plt.xlabel("Actual Price")
    plt.ylabel("Discounted Price")
    plt.title("Customer Segmentation Clusters")
    st.pyplot(fig)

elif menu == "Association Rule Mining":
    st.subheader("Frequent Itemset Mining with Apriori")
    
    # Transform data for Apriori
    basket = df.pivot_table(index='product_id', columns='category', values='discounted_price', aggfunc='sum').fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # Apply Apriori
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    # Display Rules
    st.write("### Association Rules")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

elif menu == "User Behavior Analysis":
    st.subheader("User Behavior Insights")
    
    # Analyze Reviews
    if "user_id" in df.columns:
        review_counts = df["user_id"].value_counts().head(10)
        st.write("### Top 10 Users by Review Count")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=review_counts.index, y=review_counts.values)
        plt.xticks(rotation=90)
        plt.title("Most Active Reviewers")
        st.pyplot(fig)
    else:
        st.write("User data not available in this dataset.")
