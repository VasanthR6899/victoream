# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 11:14:06 2020

@author: gauth
"""



# Multiple Linear Regression

# Importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#import RPi.GPIO as gpio
import argparse
import numpy as np
#import time
#import imutils
import cv2
#import serial
#import string
#import pynmea2
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\Vasanth\Desktop\project files/data.csv')
dataset.head()
X = dataset.iloc[:, :3].values
y = dataset.iloc[:, 5].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train1 = sc_X.fit_transform(X_train1)
X_test1 = sc_X.transform(X_test1)
sc_y = StandardScaler()

# Fitting Multiple Linear Regression to the Training set
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train1, y_train1)
y_pred_svr = regressor.predict(X_test1)


def region_of_interest(image):
	height = image.shape[0]
	triangle = np.array([
	[(-30,height),(224,106),(580,height)]
	])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask,triangle,255)
	masked_image = cv2.bitwise_and(image,mask)
	return masked_image

def displaylines(image,lines):
	lineimage = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			m,n,o,p = line.reshape(4)
			cv2.line(lineimage,(m,n),(o,p),(255,0,0),10)
	return lineimage #print(line)

# Read the frames
cap = cv2.VideoCapture(r'D:\project\challenge_video.mp4')
car_casade = cv2.CascadeClassifier(r'C:\Users\Vasanth\project files\cars.xml')

# Predicting the Test set results
while True:
	_,image =cap.read()
	#latitude,longitude=gps.get_data()
	#print(latitude,longitude) #height = 240, width = 320
	frame = cv2.resize(image,(int(image.shape[1]*1.4),int(image.shape[0]*1.4)),interpolation = cv2.INTER_AREA)
#	print("{} {}".format(int(frame.shape[0]),int(frame.shape[1])))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)
	canny = cv2.Canny(blur,50,150)
	car = car_casade.detectMultiScale(gray)
	for (x,y,w,h) in car :
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
		print("{} {} {} {}".format(x,y,w,h))
		df=pd.DataFrame([[x,x+w,y]])
		#print(df)
		cv2.putText(frame, "{}in".format(w),(int(x),int(y-2)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1)
        #from sklearn.svm import SVR
        #regressor = SVR(kernel='rbf')
        #regressor.fit(X_train1, y_train1)
		#i=SVR.regressor.predict(df)
        #print(i)
	masked = region_of_interest(canny)
	lines = cv2.HoughLinesP(masked,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap = 5)
	line_image = displaylines(frame,lines)
	final_image = cv2.addWeighted(frame,0.8, line_image,1,1)
	key = cv2.waitKey(1) & 0xFF
	#cv2.imshow("gau",frame)
	cv2.imshow("Final",final_image)
#	motormovement.forward(5)
	#if(w>80):
	#	motormovement.halt() 
	if(key == ord("q")):
		#gpio.cleanup()
		break
cap.release()
cv2.destroyAllWindows()
