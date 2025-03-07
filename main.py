import os
import subprocess
import pandas as pd
import streamlit as st

# Ensure Plotly is installed
try:
    import plotly.express as px
except ImportError:
    st.warning("Plotly is not installed. Installing now...")
    subprocess.run(["pip", "install", "plotly"], check=True)
    import plotly.express as px

# Load dataset
file_path = "amazon.csv"
df = pd.read_csv(file_path)

# Data Cleaning
for col in ["discounted_price", "actual_price", "discount_percentage"]:
    df[col] = df[col].astype(str).str.replace(r'[₹,%]', '', regex=True).str.replace(',', '').astype(float)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["rating_count"] = df["rating_count"].astype(str).str.replace(',', '').astype(float)
df["main_category"] = df["category"].apply(lambda x: x.split('|')[0] if isinstance(x, str) else x)

df.fillna(0, inplace=True)

# Streamlit UI
st.title("Amazon E-Commerce Data Analysis")

# Sidebar for selecting graphs
option = st.sidebar.selectbox("Select a graph", [
    "Price Distribution",
    "Discount vs Actual Price",
    "Rating Distribution",
    "Category Distribution",
    "Popular Products",
    "3D Price vs Discount vs Rating",
    "3D Rating vs Rating Count vs Discount"
])

# Function to plot Price Distribution
def plot_price_distribution():
    fig = px.histogram(df, x="actual_price", nbins=50, title="Price Distribution", labels={'actual_price': 'Actual Price'})
    st.plotly_chart(fig)

# Function to plot Discount vs Actual Price
def plot_discount_vs_price():
    fig = px.scatter(df, x="actual_price", y="discount_percentage", title="Discount vs Actual Price", labels={'actual_price': 'Actual Price', 'discount_percentage': 'Discount Percentage'})
    st.plotly_chart(fig)

# Function to plot Rating Distribution
def plot_rating_distribution():
    fig = px.histogram(df, x="rating", nbins=10, title="Rating Distribution", labels={'rating': 'Rating'})
    st.plotly_chart(fig)

# Function to plot Category Distribution
def plot_category_distribution():
    category_counts = df["main_category"].value_counts().head(10)
    fig = px.bar(x=category_counts.index, y=category_counts.values, title="Category Distribution (Top 10)", labels={'x': 'Category', 'y': 'Count'})
    st.plotly_chart(fig)

# Function to plot Top 10 Popular Products
def plot_popular_products():
    top_products = df.nlargest(10, "rating_count")
    fig = px.bar(x=top_products["product_name"], y=top_products["rating_count"], title="Top 10 Popular Products", labels={'x': 'Product Name', 'y': 'Rating Count'})
    st.plotly_chart(fig)

# Function to plot 3D Scatter Plot of Price, Discount, and Rating
def plot_3d_price_discount_rating():
    fig = px.scatter_3d(df, x="actual_price", y="discount_percentage", z="rating", title="3D Scatter: Price, Discount, and Rating")
    st.plotly_chart(fig)

# Function to plot 3D Scatter Plot of Rating, Rating Count, and Discount
def plot_3d_rating_vs_count_vs_discount():
    fig = px.scatter_3d(df, x="rating", y="rating_count", z="discount_percentage", title="3D Scatter: Rating, Count, Discount")
    st.plotly_chart(fig)

# Show selected graph
if option == "Price Distribution":
    plot_price_distribution()
elif option == "Discount vs Actual Price":
    plot_discount_vs_price()
elif option == "Rating Distribution":
    plot_rating_distribution()
elif option == "Category Distribution":
    plot_category_distribution()
elif option == "Popular Products":
    plot_popular_products()
elif option == "3D Price vs Discount vs Rating":
    plot_3d_price_discount_rating()
elif option == "3D Rating vs Rating Count vs Discount":
    plot_3d_rating_vs_count_vs_discount()
