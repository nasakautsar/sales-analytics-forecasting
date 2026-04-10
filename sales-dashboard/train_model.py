import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("sales.csv")

df_grouped = df.groupby('month')['sales'].sum().reset_index()

X = df_grouped[['month']]
y = df_grouped['sales']

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model saved!")
