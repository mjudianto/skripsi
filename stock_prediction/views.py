import json
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .databaseConnect import *
import math


def login(request):
  connection = connect()
  cursor = connection.cursor(dictionary=True)
  
  # print(user)
  alert = False
  
  if request.method == 'POST':
    username = request.POST.get('username')
    password = request.POST.get('password')
    
    query = f'select * from user where username = "{username}" and password = "{password}"'
    cursor.execute(query)
    user = cursor.fetchone()
    
    if (user == None):
      alert = True
    
      context = {
        'alert' : alert
      }
    else:
      if (user['admin'] != 0):
        return redirect('adminDashboard')
      else:
        return redirect('userDashboard')
  
  context = {
    'alert' : alert
  }
  
  return render(request, 'login.html', context)

def signup(request):
  connection = connect()
  cursor = connection.cursor(dictionary=True)
  
  # print(user)
  alert = False
  
  if request.method == 'POST':
    username = request.POST.get('username')
    password = request.POST.get('password')
    
    query = f'select * from user where username = "{username}"'
    cursor.execute(query)
    user = cursor.fetchone()
    
    if (user != None):
      alert = True
    else:
      query = f'insert into user values(null, "{username}", "{password}", default)'
      cursor.execute(query)
      connection.commit()
      
    context = {
      'alert' : alert
    }
  
  context = {
    'alert' : alert
  }
  
  return render(request, 'signup.html', context)

def userDashboard(request):
  connection = connect()
  cursor = connection.cursor(dictionary=True)
  
  query = f'select * from saham'
  cursor.execute(query)
  saham = cursor.fetchall()
  
  kodeSaham_values = [f"{item['nama']} - {item['kodeSaham']}" for item in saham]
  kodeSaham_json = json.dumps(kodeSaham_values)
  
  context = {
    'kodeSaham_json' : kodeSaham_json,
    'price_list' : [],
    'date_list' : [],
    'kodeSaham': '',
    'perusahaan': '',
    'predicted': None,
    'first_date': '',
    'second_date': '',
    "suggestion": ''
  }

  
  closePrice_list = []
  date_list = []
  
  if request.method == 'POST':
    searchedItem = request.POST.get('searchBarInput')
    parts = searchedItem.split(" - ")
    
    target_date = datetime.strptime('01-01-2018', '%d-%m-%Y')
    filtered_data = [item for item in saham if item['kodeSaham'] == parts[1]]
    
    # print(filtered_data[0]['kodeSaham'])
    minyak = getData("BZ=F", target_date);
    
    data = getData(filtered_data[0]['kodeSaham'], target_date)
    
    query = f'select * from timeStep where timeStepId="1"'
    cursor.execute(query)
    timeStep = cursor.fetchone()
    # print(timeStep)
    
    predicted = predict(minyak, data, timeStep['timestep'])
    
    max_gap = 0  # Initialize the maximum gap to 0
    max_gap_combination = None  # Initialize the combination with the maximum gap

    # Iterate through the numbers and calculate the gap between each pair
    max_gap = 0  # Initialize with negative infinity to ensure any gap is greater
    max_gap_combination = None  # Initialize with None to track the combination

    for i in range(len(predicted)):
        for j in range(i + 1, len(predicted)):
            value1 = predicted[i][1]
            value2 = predicted[j][1]
            gap = value1 - value2  # Calculate the gap without taking the absolute value
            if (gap*-1) > abs(max_gap):
                max_gap = abs(gap)
                max_gap_combination = (predicted[i][0], predicted[j][0])

    # Determine if the gap is positive or negative
    if max_gap_combination is None:
        suggestion = "harga terprediksi menurun, lebih baik tidak membeli dahulu"
    elif max_gap > 0:
        suggestion = "harga terprediksi menaik, waktunya membeli"
    else:
        suggestion = "zero"

    print("Maximum Gap:", abs(max_gap))
    print("Gap is", suggestion)
    print("Combination with Maximum Gap:", max_gap_combination)
    
    closePrice_list = [int(x) for x in data['closePrice'][-7:]]
    date_list = data['date'][-7:]
    
    kodeSaham = filtered_data[0]['kodeSaham']
    perusahaan = filtered_data[0]['nama']
    
    
    first_date = ""
    second_date = ""
    if (max_gap_combination is not None):
      first_date = max_gap_combination[0]
      second_date = max_gap_combination[1]

    print(first_date)
    print(second_date)
    
    context = {
      'kodeSaham_json' : kodeSaham_json,
      'price_list' : json.dumps(closePrice_list),
      'date_list' : json.dumps(date_list),
      'kodeSaham': kodeSaham,
      'perusahaan': perusahaan,
      'predicted': predicted,
      'first_date': first_date,
      'second_date': second_date,
      "suggestion": suggestion
    }

  return render(request, 'index.html', context)

def adminDashboard(request):
  connection = connect()
  cursor = connection.cursor(dictionary=True)
  
  graphContext = {}
  
  if request.method == 'POST':
    training = float(request.POST.get('training'))
    # testing = request.POST.get('testing')
    epoch = int(request.POST.get('epoch'))
    learningRate = float(request.POST.get('learningRate'))
    batchSize = int(request.POST.get('batchSize'))
    timeStep = int(request.POST.get('timeStep'))
    layers = int(request.POST.get('layers'))

    
    graphContext = lstm(training, layers, learningRate, timeStep, epoch, batchSize)
    
    query = f'update timestep set timestep={timeStep} where timeStepId="1"'
    cursor.execute(query)
    connection.commit()
    
    
    # print(graphContext)
  
  return render(request, 'admin.html', graphContext)

def adminCRUD(request):
  connection = connect()
  cursor = connection.cursor(dictionary=True)
  
  query = f'select * from saham'
  cursor.execute(query)
  saham = cursor.fetchall();
  
  message = None  
  
  context = {
    'saham' : saham,
  }
  
  
  if request.method == 'POST':
    sahamId = request.POST.get('sahamId')
    kodeSaham = request.POST.get('kodeSaham')
    namaPerusahaan = request.POST.get('namaPerusahaan')
        
    kodeSahamNew = request.POST.get('kodeSahamNew')
    namaPerusahaanNew = request.POST.get('namaPerusahaanNew')
    
    button = request.POST.get('buttonUpdateDelete')
    print(button)
    
    if (sahamId != '' and sahamId != None and button == 'update' and button != None):
      sahamExist = validateData(kodeSaham)
      if (sahamExist == True):
        query = f'update saham set kodeSaham="{kodeSaham}", nama="{namaPerusahaan}" where sahamId={sahamId}'
        cursor.execute(query)
        connection.commit()
        message = "Saham Berhasil Di Update"
    
    if (sahamId != '' and sahamId != None and button == 'delete' and button != None):
      query = f'delete from saham where sahamId={sahamId}'
      cursor.execute(query)
      connection.commit()
      message = "Saham Berhasil Di Delete"
    
    if (kodeSahamNew != '' and kodeSahamNew != None):
      sahamExist = validateData(kodeSahamNew)
      if (sahamExist == True):
        query = f'insert into saham values(null, "{kodeSahamNew}", "{namaPerusahaanNew}")'
        cursor.execute(query)
        connection.commit()
        message = "Saham Berhasil Di Tambahkan"
    
    query = f'select * from saham'
    cursor.execute(query)
    saham = cursor.fetchall();
  
    # print(message)
    context = {
      'saham' : saham,
      'message' : message
    }
    
  return render(request, 'admin_crud.html', context)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from sklearn.metrics import mean_squared_error

import requests
from datetime import datetime
from . import stock_model  # Assuming you have a custom stock model module

def getData(code, target_date):
    rapidapi_key = "798542c715msh1d2f1e14e67e2c4p17d2b7jsnd318540143ee"

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

def validateData(code):
  rapidapi_key = "798542c715msh1d2f1e14e67e2c4p17d2b7jsnd318540143ee"

  url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/hi/history/{code}/1d"
  
  headers = {
      "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com",
      "X-RapidAPI-Key": rapidapi_key,
  }
  
  response = requests.get(url, headers=headers)
  print(response)
  if response.status_code == 200:
    return True
  else:
    return False
  
def lstm(trainingPercentage, layers, learningRate, time_steps, epoch, batchSize):
  # membatasi tanggal pengambilan data 
  target_date = datetime.strptime('01-01-2018', '%d-%m-%Y')

  # trainingPercentage = 0.7
  # layers = 5
  # learningRate = 0.01
  # time_steps = 30
  # epoch = 60
  # batchSize = 32


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
  
  checkpoint_filepath = "best_model.h5" #simpan bobot model terbaik
  model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
      checkpoint_filepath,
      save_best_only=True, #memastikan hanya bobot terbaik yang akan disimpan, jdi apabila bobot terbaru lebih baik dari sbelumnya, bobot sebelumnya akan diganti
      monitor='val_loss', #memonitor kerugian validasi, callback akan menyimpan model ketika kerugian validasi berkurang
      mode='min', #menyimpan model ketika kerugian validasi diminimalkan 
      verbose=1
  )

  # Train the model
  history = model.fit(X_train, y_train, epochs=epoch, batch_size=batchSize, validation_data=(X_test, y_test), callbacks=[model_checkpoint])
  loss_list = np.array(history.history['loss'])
  val_lost_list = np.array(history.history['val_loss'])
  # plt.plot(history.history['loss'], label='Training loss')
  # plt.plot(history.history['val_loss'], label='Validation loss')
  # plt.legend()

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
  # plt.figure(figsize=(14, 6))
  # plt.plot(original_data, label="Original Data", color='blue')
  # plt.plot(training_data, label="Training Data", color='green')
  # plt.plot(testing_data, label="Testing Data", color='red')
  # plt.legend()

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

  # plt.title(f"Comparison of Original Data, Training, and Testing\n"
  #           f"Test MAPE: {test_error_percentage:.2f}% "
  #           f"Test MSE: {testing_mse:.2f} "
  #           f"Test R-squared: {test_r2_score:.2f}")
  # plt.xlabel("Time Steps")
  # plt.ylabel("Close Price")
  # plt.show()
  
  def process_data(data):
    return [int(x) if not math.isnan(x) else None for x in data.tolist()]
  
  training_data_list = process_data(training_data)
  testing_data_list = process_data(testing_data)

  context = {
    # akurasi 
    'testing_mse' : json.dumps(int(testing_mse)),
    'test_r2_score': json.dumps("{:.2f}".format(test_r2_score)),
    'test_error_percentage' :  json.dumps("{:.2f}".format(test_error_percentage * 100)),
    
    'original_data' : json.dumps([int(x) for x in original_data.tolist()]),
    'training_data' : json.dumps(training_data_list),
    'testing_data': json.dumps(testing_data_list),
    'graph_length' : json.dumps(len(original_data.tolist())),

    # 'loss_list' : json.dumps(loss_list),
    # 'val_lost_list' : json.dumps(val_lost_list)
  }
  
  return context

def predict(minyak, saham, time_steps):
  # Initialize a list to store predictions
  predictions = []
  checkpoint_filepath = "best_model.h5"
  model = tf.keras.models.load_model(checkpoint_filepath)
  
  saham = pd.DataFrame(saham)
  minyak = pd.DataFrame(minyak)
  
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
  
  def create_dataset(dataset, time_steps=1):
      dataX, dataY = [], []
      for i in range(len(dataset) - time_steps):
          a = dataset[i:(i + time_steps), :]
          dataX.append(a)
          dataY.append(dataset[i + time_steps, 1])  # Assuming 'closePrice_saham' is at index 1
      return np.array(dataX), np.array(dataY)

  combined_data = combined_df[['closePrice_minyak', 'closePrice_saham']].values
  X_train, y_train = create_dataset(combined_data, time_steps)
  
  # Use the last day in the testing dataset as the initial input
  initial_input = X_train[-1]

  # Create a loop to make predictions for the next 7 days
  for _ in range(7):
      # Reshape the initial input to match the model's input shape
      initial_input = initial_input.reshape(1, time_steps, 2)

      # Use the model to predict the next day's stock price
      next_day_prediction = model.predict(initial_input)

      # Append the predicted value to the predictions list
      predictions.append(next_day_prediction[0][0])  # Extract the predicted value from the array

      # Update the input data for the next day by shifting it and replacing the earliest data with the prediction
      initial_input = np.roll(initial_input, shift=-1, axis=1)
      initial_input[0, -1, 0] = next_day_prediction  # Assuming 'closePrice_minyak' is at index 0
      initial_input[0, -1, 1] = next_day_prediction  # Assuming 'closePrice_saham' is at index 1

  # predictions = predictions.reshape(-1, 1)
  from datetime import datetime, timedelta

  # Get today's date
  today = datetime.today()

  # Initialize a variable to count business days
  business_day_count = 0

  # Define the number of business days to retrieve
  target_business_days = 7

  # List to store the business days
  business_days = []

  # Iterate to find the next 7 business days
  while business_day_count < target_business_days:
      # Move to the next day
      today += timedelta(days=1)
      
      # Check if it's a weekend (Saturday or Sunday)
      if today.weekday() >= 5:
          continue
      
      # It's a business day, add it to the list
      business_days.append(today.strftime("%d-%m-%Y"))
      
      # Increment the count
      business_day_count += 1

  # Print the list of business days

  test_predict = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
  flat_list = [item for sublist in test_predict.tolist() for item in sublist]
  int_list = [int(x) for x in flat_list]

  # The 'predictions' list now contains the predicted stock prices for the next 7 days
  # print(flat_list)
  # print(business_days)
  
  # context = {
  #   'date' : business_days,
  #   'predicted_price' : flat_list
  # }
  context = list(zip(business_days, flat_list))
  
  return list(zip(business_days, int_list))