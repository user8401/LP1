import urllib.request
import requests
import threading
import json
import sys
import RPi.GPIO as GPIO
import os
import time
import random
import Adafruit_DHT
import smbus
from ctypes import c_short
DHTpin = 21
key= 'YI6MFL2AWEESFM4W'
GPIO.setmode(GPIO.BCM)

def readDHT():
   humi,temp=Adafruit_DHT.read_retry(Adafruit_DHT.DHT11,DHTpin)
   return (str(int(humi)),str(int(temp)))

def main():
   print('system ready...')
   URL='https://api.thingspeak.com/update?api_key=%s' %key
   print ('wait')
   while True:
       (humi,temp)=readDHT()
       finalURL=URL+'&field1=%s&fifield2=%s'%(humi,temp)
       print (finalURL)
       s=urllib.request.urlopen(finalURL)
       print (humi+' '+ temp+' ')
       s.close()
       time.sleep(5)

if _name_ == '_main_':
    main()