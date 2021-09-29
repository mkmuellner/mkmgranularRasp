import os
import pygame
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
import struct
from guizero import App, Picture, Drawing
import random
import datetime
import math

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

def inverse(x):
    y = 1/x
    return y

def env_hann(dta):
    return(dta*np.hanning(len(dta))) #this creates a hanning envelope. an array between 0 and 1.0

def env_decay(dta):
    q = np.empty(len(dta))
    for i in range(len(dta)):
        q = np.append(q, i)
    q = inverse(q)
    print(q)
    print(dta*q)
    return(dta*q) #this creates a hanning envelope. an array between 0 and 1.0

def env_exp(dta):
    q = np.empty(len(dta))
    for i in range(len(dta)):
        q = np.append(q,i)
    q = inverse(q).flip()
    return(dta*q) #this creates a hanning envelope. an array between 0 and 1.0

def reverse(dta): #reverses the sample data
    return(np.flip(dta))

def play_ready(dta, envtype): # all actions needed to play dta (as int16)
    # copy left channel onto right channel
    global currentgrain
    global grain_length_ms
    global soundloop_times
    #print(".")

    if envtype == 1: #hann envelope
        dta = env_hann(dta)
    elif envtype == 2: #decay 1/x envelope
        dta = env_decay(dta)
    elif envtype == 3:  # exp envelope
        dta = env_exp(dta)

    #print(dta)
    dta = np.vstack((dta, dta)).T  # duplicate processed channels to create a "pseudo stereo" one to play"
    dta = dta.astype('i2')  # convert data to 16 bit int format
    sounddata = dta.tobytes()  # convert to buffer (sound data)
    # doesn't actually seem to make much of a difference
    # compared to just running a single call to pygame.Mixer.Sound.play with a single sound.
    return(sounddata)

def next_grain(data,playhead_position, playhead_jitter, length_jitter): #extract the next grain from full sample "data".
    global grain_length_samples
    sample_length = len(data)
    jitter = int(sample_length * 0.01 * (playhead_jitter * (0.5-random.random())))
    ex_position = playhead_position - jitter
    if ex_position < 0:
        ex_position = -1 * ex_position
    if ex_position > (sample_length - grain_length_samples -1):
        ex_position = sample_length + (sample_length-ex_position)
    endposition = ex_position+grain_length_samples+round(grain_length_samples*length_jitter*(0.5-float(random.random())))
    extracted = data[ex_position:endposition]
    return(extracted)

def updateLFO():
    global LastLFOcall1
    global LastLFOcall2
    global LFO1
    global LFO2
    global LFO1_parameter1
    global LFO2_parameter1
    global LFO1_parameter2
    global LFO2_parameter2
    delta1 = (datetime.datetime.now() - LastLFOcall1) #the difference from last period to now
    delta2 = (datetime.datetime.now() - LastLFOcall2)

    delta1 = delta1.total_seconds()+0000.1
    delta2 = delta2.total_seconds()+0000.1

    #this doesn't work yet

    LFO1 = LFO1_parameter2* math.sin(delta1 * (2*math.pi/(1/LFO1_parameter1)))   #para1 is frequency para2 amplitude

    #print(f'LFO1 value: {LFO1}')

    LFO2 = LFO2_parameter2 * math.sin(delta2 * (2*math.pi/(1/LFO2_parameter1)))

    if delta1 > (1/LFO1_parameter1):
        LastLFOcall1 = datetime.datetime.now() #set a new timepoint when one full period is over.
    if delta2 > (1/LFO2_parameter1):
        LastLFOcall2 = datetime.datetime.now()

sourceFileDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(sourceFileDir)

sample1 = 'airplane engine start-old1.wav'
sample2 = 'NI_Fairlight_Samples-34.wav'

#read the wave file and give some stats about it
Fs, data = read(sample1) #read the wave file

##initialize sound output via pygame
channels = 12
#pygame.mixer.pre_init(buffer = 2048*16, frequency = Fs, channels = channels) #lower buffer gives more clicks
pygame.mixer.pre_init(buffer = 2048, frequency = Fs, channels = channels) #lower buffer gives more clicks but more lag
pygame.init()
pygame.mixer.init()

## apparently the below can also work
#pygame.mixer.quit()
#pygame.init(buffer = 2048*2, frequency = Fs, channels = channels)
## commented out because it was not necessary at least on PC

###describe the wave file
## make sure we transform stereo to mono here.
print(f"Samplerate = {Fs}")
print(f"number of channels = {data.shape[1]}")
length = data.shape[0] / Fs
print(f"length = {length}s")
#data = data[:,0] #keep only the left channel
#print(Fs) #Fs contains the sample rate now

data = (data[:,0]) #only process the left channel

# #plot the waveform
# plt.figure()
# plt.plot(data_s16)
# plt.plot(data_s16, color = "red")
# plt.show()

###global effects like pitch
#data = speed_up(data, 10) #larger number less uptuning
#data = speed_down(data, 4) #larger number more downtuning
#data = reverse(data)

grain_length_ms = 90.0  #in milliseconds (global)
grains_per_second = 4.0 # how many grains are triggered per second
number_of_grains = 4 # how many grain channels are there (for pygame)
playhead_speed = 50 # playhead movement in samples per second
playhead_jitter = 0.2 # jitter around the playhead as a factor. 1,0 = 10% of full sample size 0 = no jitter.
length_jitter = 0.1 #fold of original grain length
playhead_reversed = False # initial direction the playhead takes to trigger the samples.
soundloop_times = 0 #this repeats a given grain exactly after it is played for n times. 1 means repeated once.

## calculate additional information needed
grain_length_samples = round(grain_length_ms * (int(Fs)/1000)) #grain length now in samples
grain_samples = np.zeros(4) #contains the audio data
grain_waittime_ms = 1000.0 / grains_per_second  # how long to wait after one grain is triggered
currentgrain = 1 # which grain is currently supposed to be triggered
playhead_position = 0 #position of the playhead in samples

## the constant sample (played as two channels to overlap a bit)

### SAMPLE PLAYER
if False: # currently deactivated
    constant_sample = pygame.mixer.Sound(sample2) #this needs to be a sample that endlessly loopable
    constant_sample.set_volume(0.05)
    pygame.mixer.Channel(0).play(constant_sample, loops=-1, fade_ms=300)
    pygame.time.wait(300)
    constant_sample2 = pygame.mixer.Sound(sample2) #this needs to be a sample that endlessly loopable
    constant_sample2.set_volume(0.05)
    pygame.mixer.Channel(1).play(constant_sample2, loops=-1, fade_ms=300)

## initialize the three LFOs
LFO1 = 0 #this stores the LFO value (ie the multiplier)
LFO2 = 0
LFO1_type = 1 #sine
LFO2_type = 2 #sine
LFO1_parameter1 = 0.1 #for sine this will be frequency in Hz
LFO2_parameter1 = 0.2
LFO1_parameter2 = 0.2 #for sine this will be amplitude factor (multiplier)
LFO2_parameter2 = 0.3
##

LastLFOcall1 = datetime.datetime.now()
LastLFOcall2 = datetime.datetime.now()


while True: #Grain generation
    updateLFO()
    #begin_time = datetime.datetime.now()
    dta = next_grain(data,playhead_position, playhead_jitter, length_jitter)
    dta = speed_down(data, 12+round(LFO1*10)) #get some pitch variation with the LFO (just a test)
    #dta = cube_softclip(dta, 1)
    grain1 = pygame.mixer.Sound(play_ready(dta,2))
    pygame.mixer.Sound.play(grain1, loops = soundloop_times)
    pygame.time.wait(10)

    dta = next_grain(data, playhead_position, playhead_jitter, length_jitter)
    dta = speed_down(data, 12+round(LFO2*10))
    grain2 = pygame.mixer.Sound(play_ready(dta,2))
    pygame.mixer.Sound.play(grain2, loops=soundloop_times)
    pygame.time.wait(10)

    dta = next_grain(data, playhead_position, playhead_jitter, length_jitter)
    dta = speed_up(data, 12+round(LFO1*10))
    grain3 = pygame.mixer.Sound(play_ready(dta,2))
    pygame.mixer.Sound.play(grain3, loops=soundloop_times)


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

    pygame.time.wait(10)
    #print(datetime.datetime.now() - begin_time)
    #print(pygame.mixer.get_busy())

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

