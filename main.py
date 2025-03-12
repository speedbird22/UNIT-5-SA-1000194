import streamlit as st

# creating streamlit ui
st.set_page_config(page_title="Amazon Market Basket Analysis", layout="wide")
st.title("Amazon Market Basket Analysis")
analysis_option = st.sidebar.radio("Select Analysis", ["EDA", "Clustering", "Association Rules", "User Behavior"])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import re
from wordcloud import WordCloud
from collections import Counter
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ["matplotlib", "seaborn", "scikit-learn", "mlxtend", "wordcloud"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

# data loading & preprocessing

@st.cache_data
def load_data():
    df = pd.read_csv("amazon.csv")

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Remove currency symbols, commas, and empty strings, then convert to float
    price_columns = ['discounted_price', 'actual_price', 'rating_count']
    for col in price_columns:
        df[col] = df[col].astype(str).str.strip()  # Remove leading/trailing spaces
        df[col] = df[col].replace(['', ' '], np.nan)  # Convert empty strings and spaces to NaN
        df[col] = df[col].apply(
            lambda x: re.sub(r'[^\d.]', '', str(x)) if pd.notna(x) else x)  # Remove non-numeric chars
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert cleaned values to float

    # Drop any rows where price columns are still NaN after cleaning
    df.dropna(subset=price_columns, inplace=True)

    # Handle missing values (fill with median for numerical, mode for categorical)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical features
    if 'category' in df.columns:
        df['category_code'] = df['category'].astype('category').cat.codes

    # Normalize numerical columns
    num_cols = ['discounted_price', 'actual_price', 'rating_count']
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Ensure discount_percentage exists
    if 'discounted_price' in df.columns and 'actual_price' in df.columns:
        df['discount_percentage'] = ((df['actual_price'] - df['discounted_price']) / df['actual_price']) * 100

    return df


df = load_data()

# exploratory data analysis
if analysis_option == "EDA":
    st.header("Exploratory Data Analysis")

    # plotting histogram and blox plot for discounted_price and actual_price
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # histogram for discounted price
    sns.histplot(df['discounted_price'], bins=20, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Histogram: Discounted Price")
    # box plot for discounted price
    sns.boxplot(y=df['discounted_price'], ax=axes[0, 1])
    axes[0, 1].set_title("Boxplot: Discounted Price")

    # histogram for actual price
    sns.histplot(df['actual_price'], bins=20, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Histogram: Actual Price")
    # boxplot for actual price
    sns.boxplot(y=df['actual_price'], ax=axes[1, 1])
    axes[1, 1].set_title("Boxplot: Actual Price")

    st.pyplot(fig)

    # # scatter plot for discounted_price, actual_price, discount_percentage with discount percentage as color
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=df['discounted_price'], y=df['actual_price'], hue=df['discount_percentage'], palette="coolwarm",
                    ax=ax)
    ax.set_title("Discounted Price vs Actual Price (Colored by Discount Percentage)")
    ax.set_xlabel("Discounted Price")
    ax.set_ylabel("Actual Price")
    st.pyplot(fig)

    # bar chart for product rating distributions and rating_count
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=df['rating'], y=df['rating_count'], palette="viridis", ax=ax)
    ax.set_title("Average Rating Count per Product Rating")
    ax.set_xlabel("Product Rating")
    ax.set_ylabel("Average Rating Count")
    st.pyplot(fig)

    # bar chart for category distribution - FIXED: make it readable
    if 'category' in df.columns:
        # Get the top 10 categories to prevent overcrowding
        top_categories = df['category'].value_counts().nlargest(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        top_categories.plot(kind='bar', color='teal', ax=ax)
        ax.set_title("Top 10 Product Categories")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')  # Rotate labels and align them
        plt.tight_layout()  # Adjust layout to fit all labels
        st.pyplot(fig)

        # Add an option to see full category distribution
        if st.checkbox("Show full category distribution"):
            st.dataframe(df['category'].value_counts())

    # bar chart for top 10 most popular products
    if 'product_id' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        df['product_id'].value_counts().nlargest(10).plot(kind='bar', color='orange', ax=ax)
        ax.set_title("Top 10 Most Popular Products")
        ax.set_xlabel("Product ID")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # heatmap for correlation between discounted_price, actual_price, rating, rating_count, category_code
    available_cols = ['discounted_price', 'actual_price', 'rating', 'rating_count']
    if 'category_code' in df.columns:
        df['category_code'] = pd.to_numeric(df['category_code'], errors='coerce')  # Ensure numeric
        available_cols.append('category_code')

    # Select only numeric columns for correlation
    numeric_cols = df[available_cols].select_dtypes(include=[np.number])

    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap_data = numeric_cols.corr()
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Feature Correlation Heatmap with Category")
    st.pyplot(fig)


# K means clustering (customer segmentation)
elif analysis_option == "Clustering":
    st.header("Customer Segmentation")

    features = ['discounted_price', 'actual_price', 'rating_count']
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[features])

    # scatter plot for clustering
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=df['discounted_price'], y=df['actual_price'], hue=df['cluster'], palette='viridis', ax=ax)
    ax.set_title("Customer Clustering: Discounted Price vs Actual Price")
    st.pyplot(fig)

    st.write("Cluster Distribution")
    st.dataframe(df['cluster'].value_counts())


# association rule mining / apriori algorithm - COMPLETELY REVISED
elif analysis_option == "Association Rules":
    st.header("Frequent Itemset & Association Rules")

    # Define default thresholds
    default_min_support = 0.01
    default_min_confidence = 0.1

    # Allow user to adjust parameters
    st.info("Use the sliders below to adjust parameters if needed.")
    col1, col2 = st.columns(2)
    with col1:
        min_support = st.slider("Min Support", 0.001, 0.1, default_min_support, 0.001,
                                format="%.3f", help="Lower values find more rules but may be less significant")
    with col2:
        min_confidence = st.slider("Min Confidence", 0.05, 0.5, default_min_confidence, 0.01,
                                   format="%.2f", help="Lower values find more rules but may be less reliable")

    if 'category' in df.columns:
        st.subheader("Category-based Association Rules")

        # Prepare transaction data
        # Group by product_id and get categories
        if 'product_id' in df.columns:
            # Method 1: One-hot encode categories by product_id (more likely to find meaningful associations)
            transactions = df.groupby('product_id')['category'].apply(list).reset_index()

            # Convert to one-hot encoded format for apriori
            # Create a list of all unique categories
            all_categories = df['category'].unique()

            # Create binary encoded dataframe
            encoded_data = pd.DataFrame({
                cat: transactions['category'].apply(lambda x: 1 if cat in x else 0) for cat in all_categories
            })

            # Run Apriori algorithm
            with st.spinner("Finding frequent itemsets..."):
                frequent_itemsets = apriori(encoded_data, min_support=min_support, use_colnames=True)

            if not frequent_itemsets.empty:
                st.write(f"Found {len(frequent_itemsets)} frequent itemsets")
                st.dataframe(frequent_itemsets.sort_values('support', ascending=False).head(10))

                # Generate association rules
                with st.spinner("Generating association rules..."):
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

                if not rules.empty:
                    st.write(f"Found {len(rules)} association rules")
                    # Display the top rules sorted by lift
                    st.subheader("Top Association Rules (sorted by lift)")

                    # Make rules more readable by converting frozensets to strings
                    readable_rules = rules.copy()
                    readable_rules['antecedents'] = readable_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    readable_rules['consequents'] = readable_rules['consequents'].apply(lambda x: ', '.join(list(x)))

                    st.dataframe(readable_rules.sort_values('lift', ascending=False)[
                                     ['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

                    # Visualize top rules
                    if len(rules) >= 5:
                        top_rules = rules.sort_values('lift', ascending=False).head(5)

                        # Create bar chart of top rules by lift
                        fig, ax = plt.subplots(figsize=(10, 6))

                        # Convert antecedents and consequents to strings for display
                        rule_labels = [
                            f"{', '.join(map(str, list(r.antecedents)))} → {', '.join(map(str, list(r.consequents)))}"
                            for i, r in top_rules.iterrows()]

                        # Plot the rules
                        sns.barplot(x=top_rules['lift'], y=rule_labels, palette='viridis', ax=ax)
                        ax.set_title("Top 5 Association Rules by Lift")
                        ax.set_xlabel("Lift")
                        ax.set_ylabel("Rule")
                        plt.tight_layout()
                        st.pyplot(fig)
                else:
                    st.warning("No rules found with current thresholds. Try lowering the support or confidence values.")
            else:
                st.warning("No frequent itemsets found. Try lowering the minimum support value.")
        else:
            # Method 2: Create a simpler basket if product_id is not available
            # Use one-hot encoding directly on the category column
            basket = pd.get_dummies(df['category'])

            with st.spinner("Finding frequent itemsets..."):
                frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)

            if not frequent_itemsets.empty:
                st.write(f"Found {len(frequent_itemsets)} frequent itemsets")
                st.dataframe(frequent_itemsets.sort_values('support', ascending=False).head(10))

                with st.spinner("Generating association rules..."):
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

                if not rules.empty:
                    st.write(f"Found {len(rules)} association rules")

                    # Make rules more readable
                    readable_rules = rules.copy()
                    readable_rules['antecedents'] = readable_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    readable_rules['consequents'] = readable_rules['consequents'].apply(lambda x: ', '.join(list(x)))

                    st.dataframe(readable_rules.sort_values('lift', ascending=False)[
                                     ['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
                else:
                    st.warning("No rules found with current thresholds. Try lowering the support or confidence values.")
            else:
                st.warning("No frequent itemsets found. Try lowering the minimum support value.")
    else:
        st.error("Category data unavailable for Association Rule Mining.")


# User Behavior Analysis - ADDED
elif analysis_option == "User Behavior":
    st.header("User Behavior Analysis")

    # Check if required columns exist
    user_columns = ['user_id', 'user_name', 'review_title', 'review_content', 'rating']
    missing_columns = [col for col in user_columns if col not in df.columns]

    if missing_columns:
        st.warning(f"Missing columns for analysis: {', '.join(missing_columns)}")
        st.info("Using available columns for analysis.")

    # Create tabs for different analysis sections
    tabs = st.tabs(["User Profiles", "Review Analysis", "Rating Distribution"])

    # Tab 1: User Profiles
    with tabs[0]:
        st.subheader("User Activity Profiles")

        if 'user_id' in df.columns:
            # Get top active users
            user_activity = df['user_id'].value_counts().reset_index()
            user_activity.columns = ['User ID', 'Activity Count']

            st.write("Top Active Users")
            st.dataframe(user_activity.head(10))

            # User activity distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(user_activity['Activity Count'], bins=20, kde=True, ax=ax)
            ax.set_title("Distribution of User Activity")
            ax.set_xlabel("Number of Activities per User")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("User ID information not available in the dataset.")

    # Tab 2: Review Analysis
    with tabs[1]:
        st.subheader("Customer Review Analysis")

        if 'review_content' in df.columns and df['review_content'].notna().any():
            # Sample reviews
            st.write("Sample Reviews:")
            st.dataframe(df[['review_title', 'review_content']].dropna().head())

            # Word cloud of review content if wordcloud is available
            all_reviews = " ".join(df['review_content'].dropna().astype(str))

            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white',
                                      max_words=100, contour_width=3, contour_color='steelblue')
                wordcloud.generate(all_reviews)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title("Common Words in Reviews")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate word cloud: {e}")

                # Show common words as alternative
                words = re.findall(r'\b\w+\b', all_reviews.lower())
                word_counts = Counter(words)
                common_words = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Count'])

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Count', y='Word', data=common_words, ax=ax)
                ax.set_title("Most Common Words in Reviews")
                st.pyplot(fig)

        else:
            st.info("Review content not available in the dataset.")

    # Tab 3: Rating Distribution
    with tabs[2]:
        st.subheader("Rating Analysis")

        if 'rating' in df.columns:
            # Rating distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            rating_counts = df['rating'].value_counts().sort_index()
            sns.barplot(x=rating_counts.index, y=rating_counts.values, palette='viridis', ax=ax)
            ax.set_title("Distribution of Ratings")
            ax.set_xlabel("Rating")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # Rating over time if date column exists
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            if date_columns:
                date_col = date_columns[0]

                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df['year_month'] = df[date_col].dt.to_period('M')

                    ratings_by_time = df.groupby('year_month')['rating'].mean().reset_index()
                    ratings_by_time['year_month'] = ratings_by_time['year_month'].astype(str)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.lineplot(x='year_month', y='rating', data=ratings_by_time, ax=ax)
                    ax.set_title("Average Rating Over Time")
                    ax.set_xlabel("Time Period")
                    ax.set_ylabel("Average Rating")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not analyze ratings over time: {e}")
        else:
            st.info("Rating information not available in the dataset.")
