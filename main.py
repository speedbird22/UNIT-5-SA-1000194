import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Function to plot Discount vs Actual Price
def plot_discount_vs_price():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["actual_price"], y=df["discount_percentage"], color='orange')
    plt.title("Discount vs Actual Price")
    plt.xlabel("Actual Price")
    plt.ylabel("Discount Percentage")
    plt.show()

# Function to plot Rating Distribution
def plot_rating_distribution():
    plt.figure(figsize=(10, 6))
    sns.histplot(df["rating"], bins=10, kde=True, color='red')
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.show()

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

# Main menu to select graphs
def main():
    while True:
        print("\nAmazon E-Commerce Data Analysis")
        print("1. Price Distribution")
        print("2. Discount vs Actual Price")
        print("3. Rating Distribution")
        print("4. Category Distribution")
        print("5. Popular Products")
        print("6. Exit")
        choice = input("Select an option (1-6): ")
        
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
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select again.")

if __name__ == "__main__":
    main()
