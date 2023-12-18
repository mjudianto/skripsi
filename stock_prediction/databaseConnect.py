import mysql.connector
from mysql.connector import Error

def connect():
  try:
      connection = mysql.connector.connect(
          host= "127.0.0.1",
          port= '3306',
          user= "root",
          password= "",
          database= "stock_prediction",
          auth_plugin='mysql_native_password')
      if connection.is_connected():
        return connection

  except Error as e:
      print("Error while connecting to MySQL", e)