import streamlit as st
import pandas as pd
import pickle

st.title("📊 Sales Dashboard & Prediction")

df = pd.read_csv("sales.csv")

df_grouped = df.groupby('month')['sales'].sum().reset_index()

st.subheader("Monthly Sales")
st.line_chart(df_grouped.set_index('month'))

st.subheader("Sales by Product")
product_sales = df.groupby('product')['sales'].sum()
st.bar_chart(product_sales)

st.subheader("Sales by Region")
region_sales = df.groupby('region')['sales'].sum()
st.bar_chart(region_sales)

model = pickle.load(open("model.pkl", "rb"))

future_month = df_grouped['month'].max() + 1
prediction = model.predict([[future_month]])

st.subheader("📈 Next Month Prediction")
st.write(f"Predicted Sales: {int(prediction[0])}")

st.subheader("💡 Business Insight")

if df_grouped['sales'].iloc[-1] > df_grouped['sales'].iloc[0]:
    st.write("Sales trend is increasing 📈")

top_product = product_sales.idxmax()
st.write(f"Top product: {top_product}")