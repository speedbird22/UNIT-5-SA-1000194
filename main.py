import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans

# Ensure the CSV file exists before loading
def load_data():
    file_path = "amazon.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found. Please check the file path.")
    df = pd.read_csv(file_path)
    df = df.dropna()  # Remove missing values to prevent errors
    return df

df = load_data()

print("Amazon Market Basket Analysis")

# Exploratory Data Analysis (EDA)
print("\nExploratory Data Analysis (EDA)")
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

if 'discounted_price' in df.columns:
    sns.histplot(df['discounted_price'], bins=30, kde=True, ax=ax[0])
    ax[0].set_title("Distribution of Discounted Prices")
else:
    print("Column 'discounted_price' not found in the dataset!")

if 'product_category' in df.columns and 'discounted_price' in df.columns:
    sns.boxplot(x=df['product_category'], y=df['discounted_price'], ax=ax[1])
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
    ax[1].set_title("Price Distribution Across Categories")
else:
    print("Required columns for box plot are missing!")

numeric_cols = ['discounted_price', 'actual_price', 'rating', 'rating_count']
available_numeric_cols = [col for col in numeric_cols if col in df.columns]

if available_numeric_cols:
    corr = df[available_numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax[2])
    ax[2].set_title("Correlation Heatmap")
else:
    print("No numeric columns available for correlation analysis!")

plt.show()
print(df.describe())

# Customer Segmentation with Clustering
print("\nCustomer Segmentation with Clustering")
num_clusters = 4
if 'discounted_price' in df.columns and 'rating_count' in df.columns:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['discounted_price', 'rating_count']])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df['discounted_price'], y=df['rating_count'], hue=df['cluster'], palette='viridis')
    ax.set_title("Customer Segmentation Based on Price & Rating Count")
    plt.show()
    
    print(df.groupby('cluster').mean())
else:
    print("Required columns for clustering are missing!")

# Association Rule Mining
print("\nAssociation Rule Mining")
if 'product_category' in df.columns and 'product_name' in df.columns:
    df_encoded = pd.get_dummies(df[['product_category', 'product_name']])
    frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
else:
    print("Required columns for association analysis are missing!")

# User Behavior Analysis
print("\nUser Behavior Analysis")
if 'rating' in df.columns and 'rating_count' in df.columns:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x='rating', y='rating_count', data=df, ci=None, ax=ax[0])
    ax[0].set_title("Ratings vs. Review Count")
    df['rating'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[1])
    ax[1].set_title("Rating Distribution Pie Chart")
    plt.show()
else:
    print("Required columns for behavior analysis are missing!")

if 'user_id' in df.columns and 'review_title' in df.columns and 'review_content' in df.columns:
    print(df[['user_id', 'review_title', 'review_content']].head(10))
else:
    print("Review data columns are missing!")

# Customer Segmentation
print("\nCustomer Segmentation")
if 'product_category' in df.columns and 'discounted_price' in df.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=df['product_category'], y=df['discounted_price'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title("Product Categories vs. Discounted Price")
    plt.show()
    print(df[['product_category', 'discounted_price']].groupby('product_category').mean())
else:
    print("Required columns for customer segmentation are missing!")

# Frequently Bought Together Items
print("\nFrequently Bought Together Items")
if 'antecedents' in rules.columns and 'consequents' in rules.columns:
    frequent_pairs = rules[['antecedents', 'consequents', 'lift']].sort_values(by='lift', ascending=False).head(10)
    print(frequent_pairs)
else:
    print("No frequent item pairs found!")

# Insights from Customer Behavior
print("\nInsights from Customer Behavior")
if 'product_name' in df.columns and 'rating_count' in df.columns:
    top_reviewed = df[['product_name', 'rating_count']].sort_values(by='rating_count', ascending=False).head(10)
    print(top_reviewed)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['rating'], bins=5, kde=True)
    ax.set_title("Distribution of Ratings")
    plt.show()
else:
    print("Required columns for insights analysis are missing!")
