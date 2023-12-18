import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from sklearn.metrics import mean_squared_error

import requests
from datetime import datetime
import stock_model  # Assuming you have a custom stock model module

def getData(code, target_date):
    rapidapi_key = "6e0cf14825msh8abcf2141469b32p1ec411jsnd5d8d80dd79c"

    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/hi/history/{code}/1d"

    headers = {
        "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com",
        "X-RapidAPI-Key": rapidapi_key,
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        stock_data = stock_model.Stock.from_dict(data)

        filtered_data = {
            'closePrice': [],
            'date': []
        }

        for key, value in stock_data.items.items():
            data_date = datetime.strptime(value.date, '%d-%m-%Y')
            if data_date >= target_date and value.close != 0.0:
                filtered_data['closePrice'].append(float(value.close))
                filtered_data['date'].append(value.date)

        return filtered_data
    else:
        return 'error'

# membatasi tanggal pengambilan data 
target_date = datetime.strptime('01-01-2018', '%d-%m-%Y')

trainingPercentage = 0.7
layers = 5
learningRate = 0.01
time_steps = 30
epoch = 60
batchSize = 32


# mengambil data dari api dan mengubah format data menjadi dataframe    
data = getData("ASII.JK", target_date)
saham = pd.DataFrame(data)

data2 = getData("BZ=F", target_date)
minyak = pd.DataFrame(data2)


# melakukan filter untuk mencocokan tanggal yang akan digunakan untuk pelatihan 
minyak = minyak[minyak['date'].isin(saham['date'])]
saham = saham[saham['date'].isin(minyak['date'])]

# melakukan normalisasi agar rentang data 0-1
scaler = MinMaxScaler()

close_price = minyak[['closePrice']].values.reshape(-1, 1) #membuat array 1D menjadi 2D
scaler.fit(close_price)
minyak['closePrice'] = scaler.fit_transform(close_price)

close_price = saham[['closePrice']].values.reshape(-1, 1) #membuat array 1D menjadi 2D
scaler.fit(close_price)
saham['closePrice'] = scaler.fit_transform(close_price)


# menggabungkan data yang akan digunakan menjadi 1 dataframe
combined_df = minyak.merge(saham, on='date', suffixes=('_minyak', '_saham'))
combined_df = combined_df.drop('date', axis=1)

# Split the data into training and testing sets
train_size = int(len(combined_df) * trainingPercentage)
train_data, test_data = combined_df[:train_size], combined_df[train_size:]

# Prepare data for LSTM (timestep)
def create_dataset(dataset, time_steps=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_steps):
        a = dataset[i:(i + time_steps), :]
        dataX.append(a)
        dataY.append(dataset[i + time_steps, 1])  # Assuming 'closePrice_saham' is at index 1
    return np.array(dataX), np.array(dataY)

# time_steps = 10

# Combine both columns 'closePrice_minyak' and 'closePrice_saham' into a single NumPy array
combined_data = train_data[['closePrice_minyak', 'closePrice_saham']].values
X_train, y_train = create_dataset(combined_data, time_steps)

combined_data = test_data[['closePrice_minyak', 'closePrice_saham']].values
X_test, y_test = create_dataset(combined_data, time_steps)

# Build an LSTM model
legacy_adam_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learningRate)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(layers, input_shape=(time_steps, 2))) # 2 features in X
model.add(tf.keras.layers.Dropout(0.001))
model.add(tf.keras.layers.Dense(1)) # 1 output (y)
model.compile(optimizer=legacy_adam_optimizer, loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=epoch, batch_size=batchSize, validation_data=(X_test, y_test))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions to the original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# mengembalikan data saham (denormalisasi) 
close_price = saham[['closePrice']].values.reshape(-1, 1) #membuat array 1D menjadi 2D
saham['closePrice'] = scaler.inverse_transform(close_price)

# Create arrays for the original, training, and testing data
original_data = saham['closePrice'].values
training_data = np.full_like(original_data, fill_value=np.nan)
testing_data = np.full_like(original_data, fill_value=np.nan)
training_data[time_steps:len(train_predict) + time_steps] = train_predict.squeeze()
testing_data[len(train_predict) + 2 * time_steps:] = test_predict.squeeze()

# Plot the results
plt.figure(figsize=(14, 6))
plt.plot(original_data, label="Original Data", color='blue')
plt.plot(training_data, label="Training Data", color='green')
plt.plot(testing_data, label="Testing Data", color='red')
plt.legend()

# Calculate and display error percentage
train_error = np.abs(original_data[:len(train_predict)] - train_predict.squeeze())
test_error = np.abs(original_data[len(train_predict) + 2 * time_steps:] - test_predict.squeeze())

# mape
train_error_percentage = np.mean(train_error / original_data[:len(train_predict)]) 
test_error_percentage = np.mean(test_error / original_data[len(train_predict) + 2 * time_steps:]) 

#MSE
train_start = time_steps
train_end = train_start + len(train_predict)
test_start = train_end + time_steps
test_end = test_start + len(test_predict)
testing_mse = mean_squared_error(original_data[test_start:test_end], test_predict.squeeze())


# Calculate and display R-squared error
train_r2_score = r2_score(original_data[:len(train_predict)], train_predict.squeeze())
test_r2_score = r2_score(original_data[len(train_predict) + 2 * time_steps:], test_predict.squeeze())

plt.title(f"Comparison of Original Data, Training, and Testing\n"
          f"Test MAPE: {test_error_percentage:.2f}% "
          f"Test MSE: {testing_mse:.2f} "
          f"Test R-squared: {test_r2_score:.2f}")
plt.xlabel("Time Steps")
plt.ylabel("Close Price")
plt.show()

# print(combined_df.head())
# print(X_train[1:6])
# print(y_train[1:6])

# print(f"saham shape : {saham.shape}")
# print(f"minyak shape : {minyak.shape}")
    
# learning rate
# timestep
# batch size gaperlu
# optimizer
# R squared error
