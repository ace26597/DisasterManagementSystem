#!/usr/bin/python
import os
import paho.mqtt.client as paho
import gspread
import RPi.GPIO as GPIO
import time
import sys
import Adafruit_DHT
import datetime
import pandas as pd
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42

def collect(i):
    x=pd.read_csv('data.csv',usecols=[2,3,4,5])
    return x.rename(columns={'Activity_Label':'Activity'})
	
#	columns = ['T', 'H', 'W', 'F']
	# The one we will really focus on
	df2=collect(2)
	df3=collect(3)
	df4=collect(4)
	df5=collect(5)

	df2['Activity'].value_counts().plot(kind='bar', title='Plotting records by activity type', figsize=(10, 4),align='center');
	df3['Activity'].value_counts().plot(kind='bar', title='Plotting records by activity type', figsize=(10, 4),align='center');

	
client = paho.Client()
client.connect("broker.hivemq.com", 1883)
client.loop_start()

#GPIO SETUP
GPIO.setmode(GPIO.BCM)
water = 19
fire = 21
fan = 16
firex = 20
alarm = 26
fireon = False
fanon = False
alarmon = False
wateroff = False
GPIO.setup(fire, GPIO.IN)
GPIO.setup(water, GPIO.IN)
#GPIO.setup(fan, GPIO.OUT)
#GPIO.setup(firex, GPIO.OUT)
#GPIO.setup(alarm, GPIO.OUT)

def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    defaultdict = 'data.csv'
    # The final value function
    V = defaultdict(float)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
           # print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Find all states the we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        states_in_episode = set([tuple(x[0]) for x in episode])
        for state in states_in_episode:
            # Find the first occurance of the state in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]
    return V
	
def sample_policy(observation):
    """
    A policy that sticks if the score is >= 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1
	
while True:
	humidity, temperature = Adafruit_DHT.read_retry(11, 4)
	timestamp = datetime.datetime.now()
	print(timestamp)
	print 'Temp: {0:0.1f} C  Humidity: {1:0.1f} %'.format(temperature, humidity)

	digFire = GPIO.input(fire) # Flame sensor digital output
	watersense = GPIO.input(water)
#	print(watersense)
#	print(digFire)

	if(digFire) == 0 and fireon == False:
		fireon = True
		print 'Fire detected'
		print 'Starting Fire extinguisher'
#		GPIO.OUTPUT(firex, GPIO.HIGH)
	else:
		print 'No Fire Sensed'
		fireon = False
#		GPIO.OUTPUT(firex, GPIO.LOW)

	if(temperature) > 28 and fanon == False:
		fanon = True
		print 'Temperature Exceeded'
		print 'Starting Fan'
#		GPIO.OUTPUT(fan, GPIO.HIGH)
	else:
		fanon = False
#		GPIO.OUTPUT(fan, GPIO.LOW)

	if(humidity) > 50 and alarmon == False:
		alarmon = True
		print 'humidity Exceeded'
		print 'Starting Alarm'
#		GPIO.OUTPUT(alarm, GPIO.HIGH)
	else:
		alarmon = False
#		GPIO.OUTPUT(alarm, GPIO.LOW)

	if(watersense) == 1 and wateroff == False:
		wateroff = True
		print 'Water Level Exceeded'
		print 'Water Motor Stopped'
	else:
		wateroff = False
		print 'No Water sensed'

	csvresult = open("/home/pi/logs.csv","a")
	csvresult.write(str(timestamp) + " , " + str(temperature) + " , " + str(humidity) + " , " + str(wateroff) + " , " + str(fireon) + "\n")
	csvresult.close

	data=str(humidity)+str(temperature)+str(watersense)+str("/")+str(digFire)+str("/")+str(timestamp)
	client.publish("/disaster",data)

	os.system("./gdrive-linux-rpi update 1nxuwuLjToPighxsR3Zk9yCNIChcrFSeb logs.csv")
