# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 15:09:10 2020

@author: Rodolphe Johansson
"""

import csv
try :
    dataBase = open('dataBase.csv', 'a')
except :
    dataBase = open('dataBase.csv', 'w', newline='')

T = 0.01            # sampling time in seconds (1/Frequency)
Duration = 2        # Experience duration in seconds
N = int(Duration/T) # Number of data points

import serial
import time

# set up the serial line
ser = serial.Serial('COM5', 9600)
time.sleep(2)
print("Starting trainsmission")

# Read and record the data
data = []                        # empty list to store the data
for i in range(N):
    
    # To understand line 36 :
    # b = ser.readline()         # read a byte string
    # string_n = b.decode()      # decode byte string into Unicode  
    # string = string_n.rstrip() # remove \n and \r
    # flt = float(string)        # convert string to float
    # data.append(flt)           # add to the end of data list
    
    data.append(ser.readline().decode().rstrip())
    time.sleep(T)                # wait (sleep)

ser.close()

# Ask for equivalent fingers
order = []
print("Corresponding fingers (0 or 1) :")
order.append(int(input("Finger 1 : ")))
order.append(int(input("Finger 2 : ")))
order.append(int(input("Finger 3 : ")))
order.append(int(input("Finger 4 : ")))

# Writing on the csv file
writer = csv.writer(dataBase, delimiter=',', quotechar='|', 
                    quoting=csv.QUOTE_MINIMAL)
writer.writerow(data + order) 

dataBase.close()

