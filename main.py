import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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

# Function to plot Price Distribution
def plot_price_distribution():
    plt.figure(figsize=(10, 6))
    sns.histplot(df["actual_price"], bins=50, kde=True, color='blue')
    plt.title("Price Distribution")
    plt.xlabel("Actual Price")
    plt.ylabel("Frequency")
    plt.show()
    input("Press Enter to continue...")

# Function to plot Discount vs Actual Price
def plot_discount_vs_price():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["actual_price"], y=df["discount_percentage"], color='orange')
    plt.title("Discount vs Actual Price")
    plt.xlabel("Actual Price")
    plt.ylabel("Discount Percentage")
    plt.show()
    input("Press Enter to continue...")

# Function to plot Rating Distribution
def plot_rating_distribution():
    plt.figure(figsize=(10, 6))
    sns.histplot(df["rating"], bins=10, kde=True, color='red')
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.show()
    input("Press Enter to continue...")

# Function to plot Category Distribution
def plot_category_distribution():
    plt.figure(figsize=(12, 6))
    category_counts = df["main_category"].value_counts().head(10)  # Top 10 categories
    sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
    plt.title("Category Distribution (Top 10)")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()
    input("Press Enter to continue...")

# Function to plot Top 10 Popular Products
def plot_popular_products():
    plt.figure(figsize=(12, 6))
    top_products = df.nlargest(10, "rating_count")
    sns.barplot(x=top_products["product_name"], y=top_products["rating_count"], palette='magma')
    plt.title("Top 10 Popular Products")
    plt.xlabel("Product Name")
    plt.ylabel("Rating Count")
    plt.xticks(rotation=45, ha='right')
    plt.show()
    input("Press Enter to continue...")

# Function to plot 3D Scatter Plot of Price, Discount, and Rating
def plot_3d_price_discount_rating():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df["actual_price"], df["discount_percentage"], df["rating"], c=df["rating"], cmap='coolwarm')
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Discount Percentage")
    ax.set_zlabel("Rating")
    ax.set_title("3D Scatter Plot: Price, Discount, and Rating")
    plt.show()
    input("Press Enter to continue...")

# Function to plot 3D Scatter Plot of Rating, Rating Count, and Discount
def plot_3d_rating_vs_count_vs_discount():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df["rating"], df["rating_count"], df["discount_percentage"], c=df["discount_percentage"], cmap='viridis')
    ax.set_xlabel("Rating")
    ax.set_ylabel("Rating Count")
    ax.set_zlabel("Discount Percentage")
    ax.set_title("3D Scatter Plot: Rating vs Rating Count vs Discount")
    plt.show()
    input("Press Enter to continue...")

# Main menu to select graphs
def main():
    while True:
        print("\nAmazon E-Commerce Data Analysis")
        print("1. Price Distribution")
        print("2. Discount vs Actual Price")
        print("3. Rating Distribution")
        print("4. Category Distribution")
        print("5. Popular Products")
        print("6. 3D Price vs Discount vs Rating")
        print("7. 3D Rating vs Rating Count vs Discount")
        print("8. Exit")
        choice = input("Select an option (1-8): ")
        
        if choice == '1':
            plot_price_distribution()
        elif choice == '2':
            plot_discount_vs_price()
        elif choice == '3':
            plot_rating_distribution()
        elif choice == '4':
            plot_category_distribution()
        elif choice == '5':
            plot_popular_products()
        elif choice == '6':
            plot_3d_price_discount_rating()
        elif choice == '7':
            plot_3d_rating_vs_count_vs_discount()
        elif choice == '8':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select again.")

if __name__ == "__main__":
    main()
