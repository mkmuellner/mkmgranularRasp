## this is the raspberry specific version
import os
import sys
import signal
import RPi.GPIO as GPIO #activate the GPIO pins for the rotary encoders
from os import walk
import pygame
import mido
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
import struct
from guizero import App, Picture, Drawing, Text
import random
import datetime
import math
#import tkinter as tk

GPIObuffer = []

def signal_handler(sig, frame): #needed for the interrupt. cleans GPIO when program is canceled
    GPIO.cleanup()
    sys.exit(0)

def button_pressed_callback(channel): #interrupt called
    global oneA_Last
    global oneB_Last
    global GPIObuffer
    
    oneA_Last = GPIO.input(Enc_oneA)
    oneB_Last = GPIO.input(Enc_oneB)
    
    if oneA_Last != oneB_Last:
        print(f"GPIOBUFFER!{GPIObuffer}")
        GPIObuffer = np.append(GPIObuffer, oneA_Last)
        GPIObuffer = np.append(GPIObuffer, oneB_Last)
        GPIObuffer = np.append(GPIObuffer, -1)

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
    global selector
    if not(filebrowsing):

        
        # use rotary one to select stuff
        selected = counter_one
        if counter_one > 14:
            selector = 14
        if counter_one < 0:
            selector = 0
        
        global Volume1
        global Pitch1
        global Tuning1
        global data_s16
        global LFO1 #only for debugging
        global im1
        global playhead_position

        newd.image(0,0,image = im1)

        #left column granular
        newd.text(145,127, Volume1)
        newd.text(145, 127+26*1, Pitch1)
        newd.text(145, 127+26*2-3, Tuning1)
        newd.text(145, 127+26*3+18, round(grain_length_ms))
        newd.text(145, 127+26*3+18+24, envelopetype)
        newd.text(145, 127 + 26 * 3 + 18 + 24*2, playhead_speed)
        newd.text(145, 127 + 26 * 3 + 18 + 24 * 3, soundloop_times )
        newd.text(145, 127 + 26 * 3 + 18 + 24 * 4, pausetime1)

        #left jitters
        newd.text(145+100, 127, volume1_jitter)
        newd.text(145+100, 127 + 26 * 1, pitch1_jitter)
        newd.text(145+100, 127 + 26 * 3 + 18, length_jitter)
        newd.text(145+100, 127 + 26 * 3 + 18 + 24 * 2, playhead_jitter)
        newd.text(145+100, 127 + 26 * 3 + 18 + 24 * 3, soundloop_times)

        #LFO
        newd.text(675, 75, LFO1_parameter1)
        newd.text(675, 73+75, LFO2_parameter1)
        newd.text(675, 70+75*2, LFO3_parameter1)
        newd.text(675, 70-3+75*3, LFO4_parameter1)

        #flip on
        #newd.image(225, 179, image="FLIP_on.png") # then flip it to off if needed.
        newd.image(225, 179, image="FLIP_off.png")  # then flip it to off if needed.

        #picture is 302 x 74 at position 318, 30
        #draws a picture of the waveform
        if newsample:
            my_monitors_dpi = 96
            plt.figure(figsize=(370/my_monitors_dpi, 74/my_monitors_dpi),dpi = my_monitors_dpi) #need to change DPI value here for the small monitor
            plt.axis('off')
            plt.xlim([0, len(data)])
            plt.plot(data, color = "black")
            plt.savefig(fname = "AudioA.png", bbox_inches='tight', transparent=True) #
            plt.close()

        newd.image(4, 30, image="AudioA.png")  # then flip it to off if needed.

        #playhead position
        xposA = 302/len(data) * playhead_position+10
        newd.line(xposA,40,xposA,90, color = "red")

def filebrowser():
    if filebrowsing:

        global filebrowser_selected
        newd.hide()
        fbrowser = Drawing(app, width="800", height="480")
        fbrowser.rectangle(10,10,370,400, color = "white")
        fbrowser.rectangle(380, 10, 790, 400, color="white")
        filelist = os.listdir()
        indices = [i for i, x in enumerate(filelist) if ".wav" in x] #return only wav files
        element = []
        for index in indices:
            element.append(filelist[index])

        filelist = element
        q = 0
        #might want to shorten file names in the middle if longer than x

        for i in filelist[0:21]:
            fbrowser.text(30+5, 30 + q*17, i)
            q += 1

        for i in filelist[22:43]: ##44 elements can be displayed on one screen.
            fbrowser.text(20+5+380, 30 + (q-21)*17, i)
            q += 1
        filebrowser_selected = 5
        fbrowser.rectangle(15, 19+15+17*filebrowser_selected, 15+17, 19+15+17+17*filebrowser_selected, color="red")


def speed_up(dta, shift): #make the sound play faster (and higher in pitch)
    return(np.delete(dta, np.arange(0, dta.size, shift)))
    #delete every "shift"th element in the array.

def speed_down(dta, shift): #slows down the audio by just duplicating data. could anti-alias by shifting a copy and then averaging?
    dta = (np.repeat(dta, 3)) #triple the datapoints
    return(np.delete(dta, np.arange(0, dta.size, shift))) #and now remove some again to get the pitch.

def decfun(x):
    y = 2**(-0.01*x)
    return y

def env_hann(dta):
    return(dta*np.hanning(len(dta))) #this creates a hanning envelope. an array between 0 and 1.0

def env_decay(dta):
    q = np.arange(1,len(dta)+1)
    q = decfun(q)
    return(dta*q) #this creates a hanning envelope. an array between 0 and 1.0

def env_exp(dta):
    q = np.arange(1, len(dta) + 1)
    q = decfun(q)
    q = np.flip(q)
    return(dta*q) #this creates a hanning envelope. an array between 0 and 1.0

def reverse(dta): #reverses the sample data
    return(np.flip(dta))

def limiter(dta):
    return(0)


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

def speedx(sound_array, factor): #PITCH SHIFT FUNCTION
    indices = np.round( np.arange(0, len(sound_array), factor) )
    indices = indices[indices < len(sound_array)].astype(int)
    return sound_array[ indices.astype(int) ]

def stretch(sound_array, f, window_size, h): #Stretches the sound by a factor `f`
    phase  = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros(int(len(sound_array) /f + window_size))

    for i in np.arange(0, len(sound_array)-(window_size+h), h*f):

        # two potentially overlapping subarrays
        a1 = sound_array[i: i + window_size]
        a2 = sound_array[i + h: i + window_size + h]

        # resynchronize the second array on the first
        s1 =  np.fft.fft(hanning_window * a1)
        s2 =  np.fft.fft(hanning_window * a2)
        phase = (phase + np.angle(s2/s1)) % 2*np.pi
        a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase)) #this gives a complex number
        # add to result
        i2 = int(i/f)
        result[i2 : i2 + window_size] += hanning_window*a2_rephased.real

    result = ((2**(16-4)) * result/result.max()) # normalize (16bit)

    return result.astype('int16')

def pitchshift(snd_array, n, window_size=2**13, h=2**11): #""" Changes the pitch of a sound by ``n`` semitones. """
    factor = 2**(1.0 * n / 12.0)
    stretched = stretch(snd_array, 1.0/factor, window_size, h)
    return speedx(stretched[window_size:], factor)

def mainfunc():
    if not(filebrowsing):
        global dta
        global grain1
        global grain2
        global grain3
        global playhead_position
        global changed
        global playhead_reversed
        global names
        global sample1
        global sample2
        global data
        global data_second
        global channels
        global im1
        global selector
        global LFO1_parameter1



        if selector==0: im1 = "GUI_perform_480.png"
        elif selector==1: im1 = "GUI_perform_480_A_Soundfile.png"
        elif selector==2: im1 = "GUI_perform_480_A_volume.png"
        elif selector==3: im1 = "GUI_perform_480_A_pitch.png"
        elif selector==4: im1 = "GUI_perform_480_A_Tuning.png"
        elif selector==5: im1 = "GUI_perform_480_A_grainsize.png"
        elif selector == 6:
            im1 = "GUI_perform_480_A_envtype.png"
        elif selector == 7:
            im1 = "GUI_perform_480_A_playspeed.png"
        elif selector == 8:
            im1 = "GUI_perform_480_A_grainloops.png"
        elif selector == 9:
            im1 = "GUI_perform_480_A_pausetime.png"
        elif selector == 10:
            im1 = "GUI_perform_480_B_Soundfile.png"
        elif selector == 11:
            im1 = "GUI_perform_480_LFO1.png"
        elif selector == 12:
            im1 = "GUI_perform_480_LFO2.png"
        elif selector == 13:
            im1 = "GUI_perform_480_LFO3.png"
        elif selector == 14:
            im1 = "GUI_perform_480_LFO4.png"


        for msg in port.iter_pending():
            if (msg.type == 'note_on') and not (changed):  # msg.note
                print(data)
                # data = speedx(data, msg.note)  # larger number more downtuning

                changed = True
            if msg.type == 'note_off':  # change pitch of constant sound only at note off
                changed = False
                # data_second = pitchshift(constant_sample, n = 2)
                constant_sample = pygame.mixer.Sound(play_ready(constant_sample, 0))  # no envelope
                constant_sample.set_volume(0.05)
                pygame.mixer.Channel(0).play(constant_sample, loops=-1, fade_ms=300)

            if msg.type == 'control_change':
                print(msg.control)
                print(msg.value)
        # data = speedx(data2, 3)  # larger number more downtuning

        # while True: #Grain generation
        updateLFO()
        # begin_time = datetime.datetime.now()
        dta = next_grain(data, playhead_position, playhead_jitter, length_jitter)

        # dta = speed_down(data, 12+round(LFO1*10)) #get some pitch variation with the LFO (just a test)
        # dta = cube_softclip(dta, 1)
        grain1 = pygame.mixer.Sound(play_ready(dta, envelopetype))
        pygame.mixer.Sound.play(grain1, loops=soundloop_times)
        pygame.time.wait(pausetime1)

        dta = next_grain(data, playhead_position, playhead_jitter, length_jitter)
        # dta = speed_down(data, 12+round(LFO2*10))
        grain2 = pygame.mixer.Sound(play_ready(dta, envelopetype))
        pygame.mixer.Sound.play(grain2, loops=soundloop_times)
        pygame.time.wait(pausetime1)

        dta = next_grain(data, playhead_position, playhead_jitter, length_jitter)
        # dta = speed_up(data, 12+round(LFO1*10))
        grain3 = pygame.mixer.Sound(play_ready(dta, envelopetype))
        pygame.mixer.Sound.play(grain3, loops=soundloop_times)

        if not (playhead_reversed):
            playhead_position = playhead_position + playhead_speed
        else:
            playhead_position = playhead_position - playhead_speed

        if playhead_position > len(data):
            playhead_position = len(data) - grain_length_samples
            playhead_reversed = True
            print("playhead reverse")
        if playhead_position < 1:
            playhead_position = grain_length_samples
            playhead_reversed = False
            print("playhead forward")

        #pygame.time.wait(pausetime1)
def update_rotaries(): #updates the rotary encoder info from GPIO
        global oneA_Last
        global oneB_Last
        global twoA_Last
        global threeA_Last
        global fourA_Last
        
        global counter_one
        global counter_two
        global counter_three
        global counter_four
        
#         if oneA_Last != -1 :
#             print("---")
#             print(f"EncA:{oneA_Last}")
#             print(f"EncB:{oneB_Last}")
#             oneA_Last = -1
#             oneB_Last = -1
        #oneA_State = GPIO.input(Enc_oneA)
        #oneB_State = GPIO.input(Enc_oneB)
        
#         twoA_State = GPIO.input(Enc_twoA)
#         twoB_State = GPIO.input(Enc_twoB)
#         
#         threeA_State = GPIO.input(Enc_threeA)
#         threeB_State = GPIO.input(Enc_threeB)
#         
#         fourA_State = GPIO.input(Enc_fourA)
#         fourB_State = GPIO.input(Enc_fourB)
# 
#         #print(f"----{datetime.datetime.now()}")
#         #print(oneA_State)
#         #print(oneB_State)
#         #print(counter_one)
# 
#         if oneA_State != oneA_Last:
#             if oneB_State != oneA_State:
#                 if int(counter_one/2) < 24:
#                     counter_one += 1
#             else:
#                 if int(counter_one/2) > 0:
#                     counter_one -= 1
#             
#             
# 
#                     
#         oneA_Last = oneA_State
#         
#         if twoA_State != twoA_Last:
#             if twoB_State != twoA_State:
#                 counter_two += 1
#             else:
#                 counter_two -= 1
#             
#         twoA_Last = twoA_State
# 
#         if threeA_State != threeA_Last:
#             if threeB_State != threeA_State:
#                 counter_three += 1
#             else:
#                 counter_three -= 1
#             
#         threeA_Last = threeA_State
#         
#         if fourA_State != fourA_Last:
#             if fourB_State != fourA_State:
#                 counter_four += 1
#             else:
#                 counter_four -= 1
#             
#         fourA_Last = fourA_State        

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


## GPIO configuration for rotary encoders.
GPIO.setmode(GPIO.BCM)

Enc_oneA = 16
Enc_oneB = 20
Enc_oneSW = 21

Enc_twoA = 9
Enc_twoB = 10
Enc_twoSW = 11

Enc_threeA = 27
Enc_threeB = 17
Enc_threeSW = 22

Enc_fourA = 2
Enc_fourB = 3
Enc_fourSW = 4

oneA_Last = -1
oneB_Last = -1

for pin in [16,20,21,10,9,11,17,27,22,2,3,4]: #initialize the pins we use
    GPIO.setup(pin ,GPIO.IN)

#initialize the counters

counter_one = 0
counter_two = 0
counter_three = 0
counter_four = 0

#oneA_Last = GPIO.input(Enc_oneA) # previous state of the encoder
twoA_Last = GPIO.input(Enc_twoA)
threeA_Last = GPIO.input(Enc_threeA)
fourA_Last = GPIO.input(Enc_fourA)

GPIO.add_event_detect(Enc_oneA, GPIO.RISING, callback=button_pressed_callback, bouncetime=100)
GPIO.add_event_detect(Enc_twoA, GPIO.RISING, callback=button_pressed_callback, bouncetime=100)
GPIO.add_event_detect(Enc_threeA, GPIO.RISING, callback=button_pressed_callback, bouncetime=100)
GPIO.add_event_detect(Enc_fourA, GPIO.RISING, callback=button_pressed_callback, bouncetime=100)

#GPIO.add_event_detect(Enc_oneB, GPIO.RISING, callback=button_pressed_callback, bouncetime=100)


## end of GPIO configurations



names = mido.get_input_names()
im1 = "GUI_perform_480.png"
print(names) #print the names of the input devices. the first one will be used.
# with mido.open_input(names[0]) as inport:
#     for msg in inport:
#         print(msg)

# status	128 is a note on
# 	144 is a note off
# 	176 mod y (data2 gives the mod position, data1 = gives the x pos I think)
# 	224 mod x (data2 gives the mod position, data1 = gives the y pos I think)
# 	176 K1 (data2 gives value, data1 = 1)
# 	176 K2 (data2 gives value, data1 = 2)
# 	and so on. K7 data 1 = 7.
#
# data 1 = note
# data 2 = velocity
# timestamp = how long the note is held

sourceFileDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(sourceFileDir)

sample1 = 'NI_Grains-31.wav'
sample2 = 'NI_Fairlight_Samples-34.wav'

#read the wave file and give some stats about it
Fs, data = read(sample1) #read the wave file
Fs_second, data_second = read(sample2)

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
data_second = (data_second[:,0]) #only process the left channel

# #plot the waveform
# plt.figure()
# plt.plot(data_s16)
# plt.plot(data_s16, color = "red")
# plt.show()

###global effects like pitch
#data = speed_up(data, 10) #larger number less uptuning
#data = speed_down(data, 4) #larger number more downtuning

data_backup = data #back up the original data to be able to reset
data_backup_second = data_second

#data = reverse(data)

grain_length_ms = 250.0  #in milliseconds (global)
grains_per_second = 4.0 # how many grains are triggered per second
number_of_grains = 4 # how many grain channels are there (for pygame)
playhead_speed = 500 # playhead movement in samples per second
playhead_jitter = 0.2 # jitter around the playhead as a factor. 1,0 = 10% of full sample size 0 = no jitter.
length_jitter = 0.1 #fold of original grain length
playhead_reversed = False # initial direction the playhead takes to trigger the samples.
soundloop_times = 0 #this repeats a given grain exactly after it is played for n times. 1 means repeated once.
## initialize the three LFOs
LFO1_type = 1 #sine
LFO2_type = 2 #sine
LFO1_parameter1 = 0.1 #for sine this will be frequency in Hz
LFO2_parameter1 = 0.2
LFO3_parameter1 = 0.2
LFO4_parameter1 = 0.2
LFO1_parameter2 = 0.2 #for sine this will be amplitude factor (multiplier)
LFO2_parameter2 = 0.3
LFO3_parameter2 = 0.3
LFO4_parameter2 = 0.4
envelopetype = 1
Volume1 = 1.0
Volume2 = 1.0
Tuning1 = "C"
Tuning2 = "C"
Pitch1 = 1.0
Pitch2 = 1.0
pausetime1 = 10
volume1_jitter = 0
volume2_jitter = 0
pitch1_jitter = 0
##
newsample = True #just for testing the drawing of the wavefile
selector = 0 # this sets which part of the GUI is highlighted.
filebrowser_selected = 4 # which file is selected

LFO1 = 0 #this stores the LFO value (ie the multiplier)
LFO2 = 0
## calculate additional information needed
grain_length_samples = round(grain_length_ms * (int(Fs)/1000)) #grain length now in samples
grain_samples = np.zeros(4) #contains the audio data
grain_waittime_ms = 1000.0 / grains_per_second  # how long to wait after one grain is triggered
currentgrain = 1 # which grain is currently supposed to be triggered
playhead_position = 0 #position of the playhead in samples

## the constant sample (played as two channels to overlap a bit)

### SAMPLE PLAYER
if False: # currently deactivated

    pygame.time.wait(300)
    constant_sample2 = pygame.mixer.Sound(sample2) #this needs to be a sample that endlessly loopable
    constant_sample2.set_volume(0.05)
    pygame.mixer.Channel(1).play(constant_sample2, loops=-1, fade_ms=300)

LastLFOcall1 = datetime.datetime.now()
LastLFOcall2 = datetime.datetime.now()

port = mido.open_input(names[0])
#port = mido.open_input('MPK Mini Mk II 0')

changed = False #only process the audio once

### start the sample playback
data_second = stretch(data_second, 10, 2**13, 2**11)
constant_sample = pygame.mixer.Sound(play_ready(data_second,0)) #no envelope
constant_sample.set_volume(0.05)
pygame.mixer.Channel(0).play(constant_sample, loops=-1, fade_ms=300)

app = App(width=800, height=480, bg="gray50")
# app.set_full_screen()
newd = Drawing(app, width="fill", height="fill")
dummy = Text(app, "")  # not sure this dummy procedure is really needed

filebrowsing = False

signal.signal(signal.SIGINT, signal_handler)


dummy.repeat(500, GUI) #update the GUI every 300ms
dummy.repeat(30, mainfunc)  # this will be the "work loop", update every 30ms
dummy.repeat(500, filebrowser) #update the GUI every 300ms
dummy.repeat(30, update_rotaries)
app.display()
