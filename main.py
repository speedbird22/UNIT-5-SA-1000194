# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SzRMA3gKqQzA6iNBhmsd_8i9OAHm3DUS
"""
import os
os.system("pip install dash")
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import dash_table

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

# Initialize Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Amazon E-Commerce Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Button("Price Distribution", id='btn-price', n_clicks=0, style={'margin': '5px'}),
        html.Button("Discount vs Actual Price", id='btn-discount', n_clicks=0, style={'margin': '5px'}),
        html.Button("Rating Distribution", id='btn-rating', n_clicks=0, style={'margin': '5px'}),
        html.Button("Category Distribution", id='btn-category', n_clicks=0, style={'margin': '5px'}),
        html.Button("Popular Products", id='btn-popular', n_clicks=0, style={'margin': '5px'}),
    ], style={'textAlign': 'center', 'margin': '20px'}),

    dcc.Graph(id='graph-output'),
])

@app.callback(
    Output('graph-output', 'figure'),
    [Input('btn-price', 'n_clicks'),
     Input('btn-discount', 'n_clicks'),
     Input('btn-rating', 'n_clicks'),
     Input('btn-category', 'n_clicks'),
     Input('btn-popular', 'n_clicks')]
)
def display_graph(n_price, n_discount, n_rating, n_category, n_popular):
    ctx = dash.callback_context
    if not ctx.triggered:
        return px.histogram(df, x="actual_price", nbins=50, title="Price Distribution", color_discrete_sequence=['blue'])

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "btn-price":
        return px.histogram(df, x="actual_price", nbins=50, title="Price Distribution", color_discrete_sequence=['blue'])
    elif button_id == "btn-discount":
        return px.scatter(df, x="actual_price", y="discount_percentage", title="Discount vs Actual Price", color_discrete_sequence=['orange'])
    elif button_id == "btn-rating":
        return px.histogram(df, x="rating", nbins=10, title="Rating Distribution", color_discrete_sequence=['red'])
    elif button_id == "btn-category":
        return px.bar(df["main_category"].value_counts().reset_index(), x="index", y="main_category", labels={"index": "Category", "main_category": "Count"}, title="Category Distribution", color_discrete_sequence=['green'])
    elif button_id == "btn-popular":
        return px.bar(df.nlargest(10, "rating_count"), x="product_name", y="rating_count", title="Top 10 Popular Products", color_discrete_sequence=['purple'])

    return px.histogram(df, x="actual_price", nbins=50, title="Price Distribution", color_discrete_sequence=['blue'])

if __name__ == '__main__':
    app.run_server(debug=True)
