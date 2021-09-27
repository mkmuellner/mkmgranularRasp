import os
import pygame
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
import struct
from guizero import App, Picture, Drawing
import random

def grain(rev=False, playhead_pos = 0, grainsize = 50):
    #rev: should the sample be reversed
    #playhead_pos: position of the playhead in the sample in fraction. 1 = end of file
    #grainsize: size of the grain in milliseconds

    global data_s16
    global Fs #the sampling rate
    global channels #mono (1) or stereo (2)

    grainsmp = data_s16[playhead_pos*len()]

    if(rev): #should the sample be reversed
        data_s16 = np.flip(data_s16)  # flip the sample data (reverse)


def GUI():
    app = App(width=800, height=400)
    drawing = Drawing(app, width=800, height=400)
    drawing.oval(30, 30, 60, 60, color="red")
    drawing.oval(50, 50, 80, 80, color="red")

    newd = Drawing(app, width=800, height=400)
    newd.image(0,0,image = "GUI2.PNG")
    app.display()

def speed_up(dta, shift): #make the sound play faster (and higher in pitch)
    return(np.delete(dta, np.arange(0, dta.size, shift)))
    #delete every "shift"th element in the array.

def speed_down(dta, shift): #slows down the audio by just duplicating data. could anti-alias by shifting a copy and then averaging?
    dta = (np.repeat(dta, 3)) #triple the datapoints
    return(np.delete(dta, np.arange(0, dta.size, shift))) #and now remove some again to get the pitch.

def env_hann(dta):
    return(dta*np.hanning(len(dta))) #this creates a hanning envelope. an array between 0 and 1.0

def reverse(dta): #reverses the sample data
    return(np.flip(dta))

def play_int16(dta): # all actions needed to play dta (as int16)
    # copy left channel onto right channel
    print(".")
    dta = env_hann(dta)
    dta = np.vstack((dta, dta)).T  # duplicate processed channels to create a "pseudo stereo" one to play"
    dta = dta.astype('i2')  # convert data to 16 bit int format
    sounddata = dta.tobytes()  # convert to buffer (sound data)
    grain = pygame.mixer.Sound(sounddata)
    pygame.mixer.Sound.play(grain)

def next_grain(data,playhead_position, playhead_jitter): #extract the next grain from full sample "data".
    global grain_length_samples
    sample_length = len(data)
    jitter = int(sample_length * 0.1 * random.randint(-1*playhead_jitter,playhead_jitter))
    ex_position = playhead_position - jitter
    if ex_position < 0:
        ex_position = -1 * ex_position
    if ex_position > (sample_length - grain_length_samples -1):
        ex_position = sample_length + (sample_length-ex_position)
    extracted = data[ex_position:(ex_position+grain_length_samples)]
    return(extracted)


sourceFileDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(sourceFileDir)


#read the wave file and give some stats about it
Fs, data = read('tori_amos_god_3.wav') #read the wave file

##initialize sound output via pygame
channels = 2
pygame.mixer.pre_init(buffer = 2048*16, frequency = Fs, channels = channels) #lower buffer gives more clicks
pygame.mixer.init()
###describe the wave file
## make sure we transform stereo to mono here.
print(f"Samplerate = {Fs}")
print(f"number of channels = {data.shape[1]}")
length = data.shape[0] / Fs
print(f"length = {length}s")
#data = data[:,0] #keep only the left channel
#print(Fs) #Fs contains the sample rate now

data = (data[:,0]) #only process the left channel

#data = speed_up(data, 10) #larger number less uptuning
#data = speed_down(data, 2) #larger number more downtuning
#data = reverse(data)







#data = env_hann(data) #create a hanning envelope



# #plot the waveform
# plt.figure()
# plt.plot(data_s16)
# plt.plot(data_s16, color = "red")
# plt.show()

#sound = pygame.mixer.Sound('Ashlight_Sample-29.wav')
#sbuffer = sound.get_raw() #get the raw data of the larger wave file
#data_s16 = np.frombuffer(sbuffer, dtype=np.int16, count=len(sbuffer)//2, offset=0)
#data_s16 = np.frombuffer(sbuffer, dtype=np.int16, count=-1, offset=0)


# hann = np.hanning(len(data_s16)) #this creates a hanning envelope. an array between 0 and 1.0
#
# #data_s16 = data_s16*hann
# data_s16 = data_s16.astype('i2') #need to move to int again i2 is integer made up of 2 bytes (aka int16)
#
# sbuffer = data_s16.tobytes()
#
# print(len(data_s16))
# print(len(sbuffer))
#
# grain1 = pygame.mixer.Sound(sbuffer)
# s = pygame.mixer.Sound.play(grain1)

#t1 = round(time.time()*100)

grain_length_ms = 50.0  #in milliseconds (global)
grains_per_second = 4.0 # how many grains are triggered per second
number_of_grains = 4 # how many grain channels are there (for pygame)
playhead_speed = 2000 # playhead movement in samples per second
playhead_jitter = 2 # jitter around the playhead as a factor. 1,0 = 10% of full sample size 0 = no jitter.
playhead_reversed = False # initial direction the playhead takes to trigger the samples.

## calculate additional information needed
grain_length_samples = round(grain_length_ms * (int(Fs)/1000)) #grain length now in samples
grain_samples = np.zeros(4) #contains the audio data
grain_waittime_ms = 1000.0 / grains_per_second  # how long to wait after one grain is triggered
currentgrain = 0 # which grain is currently supposed to be triggered
playhead_position = 0 #position of the playhead in samples

while True: #run forever
    dta = next_grain(data,playhead_position, playhead_jitter)
    play_int16(dta)
    if not(playhead_reversed):
        playhead_position = playhead_position + playhead_speed
    else:
        playhead_position = playhead_position - playhead_speed

    if playhead_position > len(data):
        playhead_position = len(data)-grain_length_samples
        playhead_reversed = True
        print("playhead reverse")
    if playhead_position < 1:
        playhead_position = grain_length_samples
        playhead_reversed = False
        print("playhead forward")

    #print(str(playhead_position)+" "+str(len(data)))
    #if(pygame.time.get_ticks() - )

    time.sleep(0.2)
    #time.sleep(grain_waittime_ms/1000)

#data = sbuffer[:,0] #keeps only channel 0



#0.836s actual sample time
#48000Hz
#16 bit float

#t2 = round(time.time()*100)
#print(str((t2-t1)/100)+"s")

#print(round(time.time()*1000.0)-t)


# Also make sure to include this to make sure it doesn't quit immediately


#grainsize = 300 #in ms
#fadetime = int(grainsize / 2)

