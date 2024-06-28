import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Streamlit setup
st.title('Stock Trend Prediction')

# Allow user to select a CSV file from the list
csv_files = ['yahoo.csv', 'amtd.csv', 'amazon.csv', 'disney.csv', 'sbux.csv', 'twtr.csv']
csv_file = st.selectbox('Select CSV File', csv_files)

# Load data from CSV file
df = pd.read_csv(csv_file, header=None)
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Trend']  # Assign appropriate column names

# Ensure the dataframe has the correct structure
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Describing data
st.subheader(f'Data from {csv_file}')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Close')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.index, ma100, 'r', label='100MA')
plt.plot(df.index, ma200, 'g', label='200MA')
plt.plot(df.index, df['Close'], 'b', label='Close')
plt.legend()
st.pyplot(fig)

# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Prepare training data
x_train = []
y_train = []

for i in range(100, len(data_training_array)):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape(x_train.shape[0], -1)  # Flatten the input for Linear Regression

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Prepare test data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = x_test.reshape(x_test.shape[0], -1)  # Flatten the input for Linear Regression

# Make predictions based on previous actual prices
predictions = []

for i in range(len(x_test)):
    pred_input = x_test[i].reshape(1, -1)
    next_pred = model.predict(pred_input)[0]
    predictions.append(next_pred)
    x_test = np.append(x_test, pred_input).reshape(-1, x_test.shape[1])  # Append the prediction to x_test