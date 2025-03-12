import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Sample Data
data = {
    'Product': ['A', 'B', 'C', 'D', 'E'],
    'Sales': [120, 90, 75, 110, 95],
    'Profit': [30, 25, 15, 35, 20],
    'Quantity': [10, 8, 5, 12, 7]
}
df = pd.DataFrame(data)

# Function to generate a bar chart
def show_bar_chart():
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Product', y='Sales', data=df, palette='coolwarm')
    plt.title('Sales by Product', fontsize=16)
    plt.xlabel('Product', fontsize=14)
    plt.ylabel('Sales', fontsize=14)
    plt.grid(True)
    plt.show()

# Function to generate a scatter plot
def show_scatter_plot():
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x='Sales', y='Profit', data=df, hue='Product', s=250)
    plt.title('Sales vs Profit', fontsize=16)
    plt.xlabel('Sales', fontsize=14)
    plt.ylabel('Profit', fontsize=14)
    plt.grid(True)
    plt.show()

# Function to generate a pie chart
def show_pie_chart():
    plt.figure(figsize=(9, 9))
    plt.pie(df['Quantity'], labels=df['Product'], autopct='%1.1f%%', colors=sns.color_palette('coolwarm'), startangle=140)
    plt.title('Quantity Distribution', fontsize=16)
    plt.show()

# Function for 3D Visualization
def show_3d_visualization():
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(df['Sales'], df['Profit'], df['Quantity'], c='r', marker='o', s=150)
    ax.set_xlabel('Sales', fontsize=12)
    ax.set_ylabel('Profit', fontsize=12)
    ax.set_zlabel('Quantity', fontsize=12)
    ax.set_title('3D Visualization of Sales, Profit, and Quantity', fontsize=16)
    
    plt.show()

# Console-based menu
def menu():
    while True:
        print("\nChoose a visualization:")
        print("1. Bar Chart")
        print("2. Scatter Plot")
        print("3. Pie Chart")
        print("4. 3D Visualization")
        print("5. Exit")
        
        choice = input("Enter your choice: ")
        if choice == '1':
            show_bar_chart()
        elif choice == '2':
            show_scatter_plot()
        elif choice == '3':
            show_pie_chart()
        elif choice == '4':
            show_3d_visualization()
        elif choice == '5':
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    menu()
