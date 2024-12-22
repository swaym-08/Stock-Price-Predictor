import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model(r'C:\Users\shahs\Downloads\Stock_Market_Prediction_ML\price_predictor_model.keras')


st.header('Stock Market Predictor')

stock =st.text_input('Enter Stock Symnbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.70)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.70): len(data)])

def createPatternSet(data_train_scaled,steps):
  x_pattern=[]
  y_price=[]
  for day in range(steps,len(data_train_scaled)):
    row=data_train_scaled[day-steps:day,0]
    x_pattern.append(row)
    y_price.append(data_train_scaled[day])
  x_pattern,y_price=np.array(x_pattern),np.array(y_price)
  x_pattern=x_pattern.reshape(x_pattern.shape[0],x_pattern.shape[1],1)
  return x_pattern,y_price

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_test_scale = scaler.fit_transform(data_test)


x_test_final=createPatternSet(data_test_scale,25)[0]
y_test_final=createPatternSet(data_test_scale,25)[1]



st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
ma_50_days=data.Close.rolling(50).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r',label='50-Day-MA')
plt.plot(ma_100_days, 'b',label='100-Day-MA')
plt.plot(data.Close, 'g')
plt.legend() 
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r',label='100-Day-MA')
plt.plot(ma_200_days, 'b',label='200-Day-MA')
plt.plot(data.Close, 'g')
plt.legend() 
plt.show()
st.pyplot(fig3)


y_pred_test=scaler.inverse_transform(model.predict(x_test_final))

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(y_pred_test, 'r', label='Predicted Price')
plt.plot(scaler.inverse_transform(y_test_final), 'g', label = 'Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend() 
plt.show()
st.pyplot(fig4)