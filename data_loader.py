import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_data():
    """Load and transform dataset into unified format"""
    try:
        os.makedirs('data', exist_ok=True)
        df = pd.read_csv("data/mobiles.csv")
        
       
        data = []
        for _, row in df.iterrows():
           
            if pd.notna(row['Amazon_Price']):
                data.append({
                    'model': row['Model'],
                    'retailer': 'Amazon',
                    'price': float(row['Amazon_Price']),
                    'rating': float(row['Amazon_Rating']),
                    'reviews': int(row['Amazon_Number_of_Reviews']),
                    'url': row['Amazon_URL']
                })
            
         
            if pd.notna(row['Flipkart_Price']):
                data.append({
                    'model': row['Model'],
                    'retailer': 'Flipkart',
                    'price': float(row['Flipkart_Price']),
                    'rating': float(row['Flipkart_Rating']),
                    'reviews': int(row['Flipkart_Number_of_Reviews']),
                    'url': row['Flipkart_URL']
                })
            
         
            if pd.notna(row['Croma_Price']):
                data.append({
                    'model': row['Model'],
                    'retailer': 'Croma',
                    'price': float(row['Croma_Price']),
                    'rating': float(row['Croma_Rating']),
                    'reviews': int(row['Croma_Number_of_Reviews']),
                    'url': row['Croma_URL']
                })

        return pd.DataFrame(data)
    
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return pd.DataFrame()

def filter_data(df, price_range, min_rating):
    """Filter dataset based on price and rating"""
    return df[
        (df['price'].between(price_range[0], price_range[1])) &
        (df['rating'] >= min_rating)
    ].sort_values(by='price')