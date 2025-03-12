import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk

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
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Product', y='Sales', data=df, palette='coolwarm')
    plt.title('Sales by Product')
    plt.xlabel('Product')
    plt.ylabel('Sales')
    plt.show()

# Function to generate a scatter plot
def show_scatter_plot():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Sales', y='Profit', data=df, hue='Product', s=200)
    plt.title('Sales vs Profit')
    plt.xlabel('Sales')
    plt.ylabel('Profit')
    plt.show()

# Function to generate a pie chart
def show_pie_chart():
    plt.figure(figsize=(8, 8))
    plt.pie(df['Quantity'], labels=df['Product'], autopct='%1.1f%%', colors=sns.color_palette('coolwarm'))
    plt.title('Quantity Distribution')
    plt.show()

# Function for 3D Visualization
def show_3d_visualization():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(df['Sales'], df['Profit'], df['Quantity'], c='r', marker='o', s=100)
    ax.set_xlabel('Sales')
    ax.set_ylabel('Profit')
    ax.set_zlabel('Quantity')
    ax.set_title('3D Visualization of Sales, Profit, and Quantity')
    
    plt.show()

# GUI Setup
root = tk.Tk()
root.title("Data Visualization")
root.geometry("400x400")

frame = tk.Frame(root)
frame.pack(pady=20)

# Styled Buttons
style = ttk.Style()
style.configure("TButton", font=("Arial", 14, "bold"), padding=10)

btn_bar = ttk.Button(frame, text="Bar Chart", command=show_bar_chart, style="TButton")
btn_bar.pack(pady=10)

btn_scatter = ttk.Button(frame, text="Scatter Plot", command=show_scatter_plot, style="TButton")
btn_scatter.pack(pady=10)

btn_pie = ttk.Button(frame, text="Pie Chart", command=show_pie_chart, style="TButton")
btn_pie.pack(pady=10)

btn_3d = ttk.Button(frame, text="3D Visualization", command=show_3d_visualization, style="TButton")
btn_3d.pack(pady=10)

root.mainloop()
