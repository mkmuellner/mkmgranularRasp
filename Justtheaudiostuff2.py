## this is the raspberry specific version
import os
import sys
import signal
import RPi.GPIO as GPIO  # activate the GPIO pins for the rotary encoders
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
from scipy.ndimage.filters import uniform_filter1d


# import tkinter as tk

GPIObuffer = []


def signal_handler(
    sig, frame
):  # needed for the interrupt. cleans GPIO when program is canceled
    GPIO.cleanup()
    sys.exit(0)


def button_pressed_callback(channel):  # interrupt called
    global onepressed
    global twopressed
    global threepressed
    global fourpressed

    #reset everything. assume only one buttonpress is supposed to be recorded at a time
    onepressed = False
    twopressed = False
    threepressed = False
    fourpressed = False
    
    if channel == 21:
        onepressed = True
        print("SW one pressed")
    if channel == 11:
        twopressed = True
        print("SW two pressed")
    if channel == 22:
        threepressed = True
        print("SW three pressed")
    if channel == 4:
        fourpressed = True
        print("SW four pressed")
    else:
        print("Not sure what was pressed. Got channel:")
        print(channel)

def turned_rotary_callback(channel):  # interrupt called
    global selector #could also be using counter_one here instead
    global counter_two
    global counter_three
    global counter_four
    
    oneB = GPIO.input(Enc_oneB)
    twoB = GPIO.input(Enc_twoB)
    threeB = GPIO.input(Enc_threeB)
    fourB = GPIO.input(Enc_fourB)
 
    
    if channel == Enc_oneA:
        if oneB == 1:
            selector += 1
        else:
            selector -= 1
    if channel == Enc_twoA: #on that one the orientation (hardware) seems to be flipped
        if twoB == 1:
            counter_two += 1
        else:
            counter_two -= 1
    if channel == Enc_threeA:
        if threeB == 1:
            counter_three += 1
        else:
            counter_three -= 1
    if channel == Enc_fourA:
        if fourB == 1:
            counter_four += 1
        else:
            counter_four -= 1
    print(f"one: {selector}")
    print(f"two: {counter_two}")
    print(f"three: {counter_three}")
    print(f"four: {counter_four}")



def filebrowser():
    global filebrowser_selected
    global selector
    global filebrowsing
    global oldselector
    global onepressed
    global sample1
    if filebrowsing:
        newd.clear()
        newd.rectangle(10, 10, 370, 400, color="white")
        newd.rectangle(380, 10, 790, 400, color="white")
        filelist = os.listdir()
        indices = [
            i for i, x in enumerate(filelist) if ".wav" in x
        ]  # return only wav files
        element = []
        for index in indices:
            element.append(filelist[index])

        filelist = element
        q = 0
        # might want to shorten file names in the middle if longer than x

        #make sure we don't allow the selector to become bigger than the max number of files shown
        if selector > len(filelist)-1:
            selector = 0  # 14 items total
        if selector < 0:
            selector = len(filelist)-1 #this wraps around when needed
            
        #now print the filenames

        for i in filelist[0:21]:
            if q == selector:
                col = "red"
            else:
                col = "black"
            newd.text(30 + 5, 30 + q * 17, i, color=col)
            q += 1

        for i in filelist[22:43]:  ##44 elements can be displayed on one screen.
            if q == selector:
                col = "red"  # highlight color
            else:
                col = "black"
            newd.text(20 + 5 + 380, 30 + (q - 21) * 17, i, color=col)
            q += 1

        if selector != oldselector:  # if there was a change in the rotary
            print(selector)
            filebrowser_selected = selector
            newd.rectangle(
                15,
                19 + 15 + 17 * oldselector,
                15 + 17,
                19 + 15 + 17 + 17 * oldselector,
                color="white",
            )
            newd.rectangle(
                15,
                19 + 15 + 17 * selector,
                15 + 17,
                19 + 15 + 17 + 17 * selector,
                color="red",
            )
            oldselector = selector
            
        if onepressed:
            onepressed = False
            sample1 = filelist[selector]
            loadsamples()
            filebrowsing = False #exit browser
            selector = 1 #reset the selector
            newd.clear()
            global newsample
            newsample = True
            print("switching")

def loadsamples(): #load a new sample 1 or 2
    global sample1
    global sample2
    global Fs
    global Fs_second
    global data
    global data_second
    global data_backup
    global data_backup_second
    global Right_Limit
    global Left_Limit
    
    # read the wave file and give some stats about it
    Fs, data = read(sample1)  # read the wave file
    Fs_second, data_second = read(sample2)
        
    data = data[:, 0]  # only process the left channel
    data_second = data_second[:, 0]  # only process the left channel


    data_backup = data  # back up the original data to be able to reset
    data_backup_second = data_second
    Right_Limit = len(data)
    Left_Limit = 1

def smoothen(dta): # reduce the harshness by running a simple running average as a lowpass filter
    return(uniform_filter1d(dta, size = 4)) #size gives the amount of smoothing
    

def speed_up(dta, shift):  # make the sound play faster (and higher in pitch)
    return np.delete(dta, np.arange(0, dta.size, shift))
    # delete every "shift"th element in the array.


def speed_down(
    dta, shift
):  # slows down the audio by just duplicating data. could anti-alias by shifting a copy and then averaging?
    dta = np.repeat(dta, 3)  # triple the datapoints
    return np.delete(
        dta, np.arange(0, dta.size, shift)
    )  # and now remove some again to get the pitch.


def decfun(x):
    y = 2 ** ((-0.01*1/(grain_length_ms/100)) * x) #the grain lethgth divided by 100 kinda works out
    return y


def env_hann(dta):
    return dta * np.hanning(
        len(dta)
    )  # this creates a hanning envelope. an array between 0 and 1.0


def env_decay(dta):
    q = np.arange(1, len(dta) + 1)
    q = decfun(q)
    return dta * q  # this creates a hanning envelope. an array between 0 and 1.0


def env_exp(dta):
    q = np.arange(1, len(dta) + 1)
    q = decfun(q)
    q = np.flip(q)
    return dta * q  # this creates a hanning envelope. an array between 0 and 1.0


def reverse(dta):  # reverses the sample data
    return np.flip(dta)


def limiter(dta):
    return 0

# consider putting multiple grains into one array to have something longer to play every time.
def play_ready(dta, envtype):  # all actions needed to play dta (as int16)
    # copy left channel onto right channel
    global currentgrain
    global grain_length_ms
    global soundloop_times
    global Pitch1
    
    # print(".")

    if envtype == 1:  # hann envelope
        dta = env_hann(dta)
    elif envtype == 2:  # decay 1/x envelope
        dta = env_decay(dta)
    elif envtype == 3:  # exp envelope
        dta = env_exp(dta)
    
    if Pitch1 != 1:
        dta = pitchshift(dta, Pitch1)
    
    ##reverse
    if reversegrain:
        dta = reverse(dta)
    
    if True: #speed change
        dta = speed_down(dta,random.randrange(1,6))
    
    if True: #might want to be able to turn smoothing off
        dta = smoothen(dta)
    
    
    
    dta = np.vstack(
        (dta, dta)
    ).T  # duplicate processed channels to create a "pseudo stereo" one to play"
    dta = dta.astype("i2")  # convert data to 16 bit int format
    sounddata = dta.tobytes()  # convert to buffer (sound data)
    # doesn't actually seem to make much of a difference
    # compared to just running a single call to pygame.Mixer.Sound.play with a single sound.
    return sounddata


def next_grain(
    data, playhead_position, playhead_jitter, length_jitter
):  # extract the next grain from full sample "data".
    global grain_length_samples
    global Fs
    
    
    sample_length = grain_length_samples
    jitter = int(sample_length * playhead_jitter * ((0.5 - random.random())))
    ex_position = playhead_position + jitter

    if ex_position > (len(data) - grain_length_samples - 1):
        ex_position = len(data) + (sample_length - ex_position)
    if ex_position < 0:
        ex_position = abs(ex_position)
    endposition = (
        ex_position
        + grain_length_samples
        + int(grain_length_samples * length_jitter * (0.5 - float(random.random())))
    )
    extracted = data[ex_position:endposition]
    grain_length_samples = len(extracted)

    #make a little dot at grainpos
    xposA1 = (300-5) / len(data) * ex_position + 5
    newd.text(xposA1, 65, "|", color="white")
#    newd.line(xposA, 60, xposA, 70, color="lightred")
    return extracted


def updateLFO():
    global LastLFOcall1
    global LastLFOcall2
    global LFO1
    global LFO2
    global LFO1_parameter1
    global LFO2_parameter1
    global LFO1_parameter2
    global LFO2_parameter2
    delta1 = (
        datetime.datetime.now() - LastLFOcall1
    )  # the difference from last period to now
    delta2 = datetime.datetime.now() - LastLFOcall2

    delta1 = delta1.total_seconds() + 0000.1
    delta2 = delta2.total_seconds() + 0000.1

    # this doesn't work yet

    LFO1 = LFO1_parameter2 * math.sin(
        delta1 * (2 * math.pi / (1 / LFO1_parameter1))
    )  # para1 is frequency para2 amplitude

    # print(f'LFO1 value: {LFO1}')

    LFO2 = LFO2_parameter2 * math.sin(delta2 * (2 * math.pi / (1 / LFO2_parameter1)))

    if delta1 > (1 / LFO1_parameter1):
        LastLFOcall1 = (
            datetime.datetime.now()
        )  # set a new timepoint when one full period is over.
    if delta2 > (1 / LFO2_parameter1):
        LastLFOcall2 = datetime.datetime.now()


def speedx(sound_array, factor):  # PITCH SHIFT FUNCTION
    indices = np.round(np.arange(0, len(sound_array), factor))
    indices = indices[indices < len(sound_array)].astype(int)
    return sound_array[indices.astype(int)]


def stretch(sound_array, f, window_size, h):  # Stretches the sound by a factor `f`
    phase = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros(int(len(sound_array) / f + window_size))

    for i in np.arange(0, len(sound_array) - (window_size + h), h * f):

        # two potentially overlapping subarrays
        a1 = sound_array[i : i + window_size]
        a2 = sound_array[i + h : i + window_size + h]

        # resynchronize the second array on the first
        s1 = np.fft.fft(hanning_window * a1)
        s2 = np.fft.fft(hanning_window * a2)
        phase = (phase + np.angle(s2 / s1)) % 2 * np.pi
        a2_rephased = np.fft.ifft(
            np.abs(s2) * np.exp(1j * phase)
        )  # this gives a complex number
        # add to result
        i2 = int(i / f)
        result[i2 : i2 + window_size] += hanning_window * a2_rephased.real

    result = (2 ** (16 - 4)) * result / result.max()  # normalize (16bit)

    return result.astype("int16")


def pitchshift(
    snd_array, n, window_size=2 ** 13, h=2 ** 11
):  # """ Changes the pitch of a sound by ``n`` semitones. """
    factor = 2 ** (1.0 * n / 12.0)
    stretched = stretch(snd_array, 1.0 / factor, window_size, h)
    return speedx(stretched[window_size:], factor)


def graintrigger(pdata): #this now allows to change up the datafile to pick the grain from
    global grain1
    update_playhead()
    dta = next_grain(pdata, playhead_position, playhead_jitter, length_jitter)
    # dta = speed_down(data, 12+round(LFO1*10)) #get some pitch variation with the LFO (just a test)
    # dta = cube_softclip(dta, 1)
    grain1 = pygame.mixer.Sound(play_ready(dta, envtype))
    
    #volume control grain 1
    volume_correction_factor = 0.06 #volume of 1 is ususally too loud for headphones
    
    calcvolume = volume_correction_factor*Volume1 + volume1_jitter*(0.5-random.random()) #add the jitter. 
    if calcvolume > 1:
        calcvolume =1
    if calcvolume <0.01:
        calcvolume = 0.01
    grain1.set_volume(calcvolume) #set the volume
    ## end of volume control
    
    if loop_jitter > 0:
        rnd = random.randrange(loop_jitter)
    else:
        rnd = 0
    pygame.mixer.Sound.play(grain1, loops=soundloop_times+rnd)
    #pygame.time.wait(pausetime1)

def GUI():
    global selector
    global filebrowsing
    if not (filebrowsing):
        newd.clear()
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
        global LFO1  # only for debugging
        global im1
        global playhead_position
        global newsample
        global grain_length_samples
        global grain_waittime_ms
        global reversegrain
        global Right_Limit
        global Left_Limit
        

        newd.image(0, 0, image=im1)

        #limiters
        #leftmost is 5 rightmost is 300
        Left_Limiter_Pos = 5+round(Left_Limit/len(data)*(300-5))
        Right_Limiter_Pos = 5+round(Right_Limit/len(data)*(300-5))
        newd.text(Left_Limiter_Pos,20,"|", color = "white") #y = 58 is center
        newd.text(Right_Limiter_Pos,20,"|", color = "white")
        
        
        newd.text(145, 127, '{:.2f}'.format(Volume1))
        newd.text(145, 127 + 26 * 1, '{:.2f}'.format(Pitch1))
        newd.text(145, 127 + 26 * 2 - 3, Tuning1)
        newd.text(145, 127 + 26 * 3 + 18, round(grain_length_ms))
        newd.text(145, 127 + 26 * 3 + 18 + 24, envtype)
        newd.text(145, 127 + 26 * 3 + 18 + 24 * 2, playhead_speed)
        newd.text(145, 127 + 26 * 3 + 18 + 24 * 3, soundloop_times)
        newd.text(145, 127 + 26 * 3 + 18 + 24 * 4, pausetime1)

        # left jitters
        newd.text(145 + 100, 127, '{:.2f}'.format(volume1_jitter))
        newd.text(145 + 100, 127 + 26 * 1, '{:.2f}'.format(pitch1_jitter))
        newd.text(145 + 100, 127 + 26 * 3 + 18, '{:.2f}'.format(length_jitter))
        newd.text(145 + 100, 127 + 26 * 3 + 18 + 24 * 2, '{:.2f}'.format(playhead_jitter))
        newd.text(145 + 100, 127 + 26 * 3 + 18 + 24 * 3, loop_jitter)

        # LFO
        newd.text(675, 75, '{:.2f}'.format(LFO1_parameter1))
        newd.text(675, 73 + 75, '{:.2f}'.format(LFO2_parameter1))
        newd.text(675, 70 + 75 * 2, '{:.2f}'.format(LFO3_parameter1))
        newd.text(675, 70 - 3 + 75 * 3, '{:.2f}'.format(LFO4_parameter1))

        # flip on
        # newd.image(225, 179, image="FLIP_on.png") # then flip it to off if needed.
        if reversegrain:
            newd.image(225, 179, image="FLIP_on.png")  # then flip it to off if needed.
        else:
            newd.image(225, 179, image="FLIP_off.png")  # then flip it to off if needed.
        

        # picture is 302 x 74 at position 318, 30
        # draws a picture of the waveform
        if newsample:
            my_monitors_dpi = 96
            plt.figure(
                figsize=(370 / my_monitors_dpi, 74 / my_monitors_dpi), #370
                dpi=my_monitors_dpi,
            )  # need to change DPI value here for the small monitor
            plt.axis("off")
            plt.xlim([0, len(data)])
            plt.plot(data, color="black")
            plt.savefig(fname="AudioA.png", bbox_inches="tight", transparent=True)  #
            plt.close()
            newsample = False

        newd.image(5, 30, image="AudioA.png")  # then flip it to off if needed.

        # playhead position
        xposA = (300-5) / len(data) * playhead_position + 5
        
        newd.line(xposA, 40, xposA, 90, color="red")
        
        #these two below here only need to be updated when the GUI value changes anyway
        grain_length_samples = round(grain_length_ms * (int(Fs) / 1000))  # grain length now in samples
        grain_waittime_ms = (1000.0 / grains_per_second)  # how long to wait after one grain is triggered




def mainfunc():
    global filebrowsing
    global selector
    global oldselector
    global reversegrain
    if filebrowsing:
        reversegrain = False
        filebrowser()
    else:
        global dta
        
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
        global counter_two
        global counter_three
        global Volume1
        global volume1_jitter
        global Pitch1
        global pitch1_jitter
        global LFO1_parameter1
        global onepressed
        global grain_length_ms
        global length_jitter
        global playhead_speed
        global playhead_jitter
        global soundloop_times
        global pausetime1
        global speed_jitter
        global envtype
        global Right_Limit
        global Left_Limit
        global loop_jitter

        if oldselector != selector: #if the selector changed reset the counters
            counter_two = 0
            counter_three = 0
            

        if selector == 1:
            im1 = "GUI_perform_480_SoundA.png"
                        
            
            Left_Limit = int(Left_Limit+counter_two*4*len(data)/100)
            Right_Limit = int(Right_Limit+counter_three*4*len(data)/100)
            
            if Left_Limit < 1:
                Left_Limit = 1
            if Right_Limit > len(data):
                Right_Limit = len(data)
            if Left_Limit > Right_Limit: #if left and right cross, just swap the values
                holder = Right_Limit
                Right_Limit = Left_Limit
                Left_Limit = holder
                
        elif selector == 0:
            im1 = "GUI_perform_480_A_Soundfile.png"

            
            
        elif selector == 2:
            im1 = "GUI_perform_480_A_volume.png"
            
            #changing volume
            Volume1 = Volume1+(counter_two*5/100)
            if Volume1 > 1:
                Volume1 = 1
                counter_two = 0
            if Volume1 < 0:
                Volume1 = 0.01
                counter_two = 19
            volume1_jitter = volume1_jitter+counter_three*2/100
            if volume1_jitter < 0:
                volume1_jitter = 0
                
            if volume1_jitter > 1:
                volume1_jitter = 1
                
            
        elif selector == 3:
            im1 = "GUI_perform_480_A_pitch.png"
            
            #changing pitch
            #this does not work yet, anything other than 1 crashes
            Pitch1 = Pitch1+(counter_two)
            if Pitch1 > 10:
                Pitch1 = 10
                counter_two = 0
            if Pitch1 < 1:
                Pitch1 = 1
                counter_two = 19
            pitch1_jitter = pitch1_jitter+counter_three
            if pitch1_jitter < 0:
                pitch1_jitter = 0
                
            if pitch1_jitter > 1:
                pitch1_jitter = 1
                
            
        elif selector == 4:
            im1 = "GUI_perform_480_A_Tuning.png"
            
            
            if (counter_three % 2 == 0): #even number
                reversegrain = False
            else:
                reversegrain = True
            
        elif selector == 5:
            im1 = "GUI_perform_480_A_grainsize.png"
            
              #changing grainsize
            grain_length_ms = grain_length_ms+(counter_two*5)
            if grain_length_ms > 1000:
                grain_length_ms = 1000
                
            if grain_length_ms < 30:
                grain_length_ms = 30
                
            length_jitter = length_jitter+counter_three*2/100
            if length_jitter < 0:
                length_jitter = 0
                
            if length_jitter > 1:
                length_jitter = 1
                   
            
            
        elif selector == 6:
            im1 = "GUI_perform_480_A_envtype.png"
            
            envtype = envtype+counter_two
            if envtype > 3:
                envtype = 1
            if envtype < 1:
                envtype = 3
            
            
            
        elif selector == 7:
            im1 = "GUI_perform_480_A_playspeed.png"

#changing playhead
            playhead_speed = playhead_speed+(counter_two*20)
            if playhead_speed > 2000:
                playhead_speed = 2000
                
            if playhead_speed < 20:
                playhead_speed = 20
                
            playhead_jitter = playhead_jitter+counter_three*1/4
            if playhead_jitter < 0:
                playhead_jitter = 0
                
            if playhead_jitter > 10:
                playhead_jitter = 10
                
            #did not include speed_jitter just now
            
        elif selector == 8:
            im1 = "GUI_perform_480_A_grainloops.png"
            
            #changing loops
            
            if counter_two > 0:
                soundloop_times += 1
            if counter_two < 0:
                soundloop_times -= 1
            
            if soundloop_times > 5:
                soundloop_times = 5
                counter_two = 10
            if soundloop_times < 0:
                soundloop_times = 0
                counter_two = 0
            
                        
            if counter_three > 0:
                loop_jitter += 1
            if counter_three < 0:
                loop_jitter -= 1
 
            if loop_jitter > 5:
                loop_jitter = 5
                counter_three = 10
            if loop_jitter < 0:
                loop_jitter = 0
                counter_three = 0
            
        elif selector == 9:
            im1 = "GUI_perform_480_A_pausetime.png"

#changing pausetime
            pausetime1 = pausetime1+(counter_two*10)
            if pausetime1 > 400:
                pausetime1 = 400
                counter_two = 40
            if pausetime1 < 0:
                pausetime1 = 0
                counter_two = 0
 
            
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

        if onepressed:
            onepressed = False
            if selector == 0:
                filebrowsing = True
        
        #reset things again
        counter_two = 0
        counter_three = 0
        
        oldselector = selector

        for msg in port.iter_pending():
            if (msg.type == "note_on") and not (changed):  # msg.note
                print(data)
                # data = speedx(data, msg.note)  # larger number more downtuning

                changed = True
            if (
                msg.type == "note_off"
            ):  # change pitch of constant sound only at note off
                changed = False
                # data_second = pitchshift(constant_sample, n = 2)
                constant_sample = pygame.mixer.Sound(
                    play_ready(constant_sample, 0)
                )  # no envelope
                constant_sample.set_volume(0.05)
                pygame.mixer.Channel(0).play(constant_sample, loops=-1, fade_ms=300)

            if msg.type == "control_change":
                print(msg.control)
                print(msg.value)
        # data = speedx(data2, 3)  # larger number more downtuning

        # while True: #Grain generation
        updateLFO()
        # begin_time = datetime.datetime.now()
        graintrigger(data)

        




def update_playhead():
    global playhead_position
    global speed_jitter
    global playhead_speed
    global playhead_reversed

       #pygame.time.wait(pausetime1)
    if not (playhead_reversed):
        
        playhead_position = int(playhead_position+playhead_speed+speed_jitter*(random.random()))
    else:
        playhead_position = int(playhead_position-playhead_speed-speed_jitter*(random.random()))


    if playhead_position > Right_Limit:
        playhead_position = Right_Limit - grain_length_samples
        playhead_reversed = True
        print("playhead reverse")
    if playhead_position < Left_Limit:
        playhead_position = grain_length_samples+Left_Limit
        playhead_reversed = False
        print("playhead forward")

## GPIO configuration for rotary encoders.
GPIO.setmode(GPIO.BCM)

Enc_oneA = 16
Enc_oneB = 20
Enc_oneSW = 21
onepressed = False

Enc_twoA = 9
Enc_twoB = 10
Enc_twoSW = 11
twopressed = False

Enc_threeA = 17
Enc_threeB = 27
Enc_threeSW = 22
Threepressed = False

Enc_fourA = 2
Enc_fourB = 3
Enc_fourSW = 4
fourpressed = False

oneA_Last = -1
oneB_Last = -1

for pin in [16, 20, 21, 10, 9, 11, 17, 27, 22, 2, 3, 4]:  # initialize the pins we use
    GPIO.setup(pin, GPIO.IN)

# initialize the counters
oldselector = -1  # this stores the value of the previous selector

counter_one = 0
counter_two = 0
counter_three = 0
counter_four = 0
onearmed = True  # rotary encoder 1 is ready for action.

# oneA_Last = GPIO.input(Enc_oneA) # previous state of the encoder
twoA_Last = GPIO.input(Enc_twoA)
threeA_Last = GPIO.input(Enc_threeA)
fourA_Last = GPIO.input(Enc_fourA)

Last_one = datetime.datetime.now()

GPIO.add_event_detect(
    Enc_oneSW, GPIO.RISING, callback=button_pressed_callback, bouncetime=200
)
GPIO.add_event_detect(
    Enc_twoSW, GPIO.RISING, callback=button_pressed_callback, bouncetime=100
)
GPIO.add_event_detect(
    Enc_threeSW, GPIO.RISING, callback=button_pressed_callback, bouncetime=100
)
GPIO.add_event_detect(
    Enc_fourSW, GPIO.RISING, callback=button_pressed_callback, bouncetime=100
)


GPIO.add_event_detect(
    Enc_oneA, GPIO.FALLING, callback=turned_rotary_callback, bouncetime=300
)
GPIO.add_event_detect(
    Enc_twoA, GPIO.FALLING, callback=turned_rotary_callback, bouncetime=300
)
GPIO.add_event_detect(
    Enc_threeA, GPIO.FALLING, callback=turned_rotary_callback, bouncetime=300
)
GPIO.add_event_detect(
    Enc_fourA, GPIO.FALLING, callback=turned_rotary_callback, bouncetime=300
)

# GPIO.add_event_detect(Enc_oneB, GPIO.RISING, callback=button_pressed_callback, bouncetime=100)


## end of GPIO configurations


names = mido.get_input_names()
im1 = "GUI_perform_480.png"
print(names)  # print the names of the input devices. the first one will be used.
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

sample1 = "NI_Grains-31.wav"
sample2 = "NI_Fairlight_Samples-34.wav"

# read the wave file and give some stats about it
Fs, data = read(sample1)  # read the wave file
Fs_second, data_second = read(sample2)

##initialize sound output via pygame
channels = 12
# pygame.mixer.pre_init(buffer = 2048*16, frequency = Fs, channels = channels) #lower buffer gives more clicks
pygame.mixer.pre_init(
    buffer=1024, frequency=Fs, channels=channels
)  # lower buffer gives more clicks but more lag
pygame.init()
pygame.mixer.init()

## apparently the below can also work
# pygame.mixer.quit()
# pygame.init(buffer = 2048*2, frequency = Fs, channels = channels)
## commented out because it was not necessary at least on PC

###describe the wave file
## make sure we transform stereo to mono here.
print(f"Samplerate = {Fs}")
print(f"number of channels = {data.shape[1]}")
length = data.shape[0] / Fs
print(f"length = {length}s")
# data = data[:,0] #keep only the left channel
# print(Fs) #Fs contains the sample rate now

data = data[:, 0]  # only process the left channel
data_second = data_second[:, 0]  # only process the left channel

# #plot the waveform
# plt.figure()
# plt.plot(data_s16)
# plt.plot(data_s16, color = "red")
# plt.show()

###global effects like pitch
# data = speed_up(data, 10) #larger number less uptuning
# data = speed_down(data, 4) #larger number more downtuning

data_backup = data  # back up the original data to be able to reset
data_backup_second = data_second

# data = reverse(data)
reversegrain = False #should the grain be reversed
grain_length_ms = 250.0  # in milliseconds (global)
grains_per_second = 4.0  # how many grains are triggered per second
number_of_grains = 4  # how many grain channels are there (for pygame)
playhead_speed = 500  # playhead movement in samples per second
speed_jitter = 0
playhead_jitter = 0.2  # jitter around the playhead as a factor. 1,0 = 10% of full sample size 0 = no jitter.
length_jitter = 0.1  # fold of original grain length
playhead_reversed = (
    False  # initial direction the playhead takes to trigger the samples.
)
soundloop_times = 0  # this repeats a given grain exactly after it is played for n times. 1 means repeated once.
## initialize the three LFOs
LFO1_type = 1  # sine
LFO2_type = 2  # sine
LFO1_parameter1 = 0.1  # for sine this will be frequency in Hz
LFO2_parameter1 = 0.2
LFO3_parameter1 = 0.2
LFO4_parameter1 = 0.2
LFO1_parameter2 = 0.2  # for sine this will be amplitude factor (multiplier)
LFO2_parameter2 = 0.3
LFO3_parameter2 = 0.3
LFO4_parameter2 = 0.4
Right_Limit = len(data)#pretty random number to initialize
Left_Limit = 1
envtype = 1
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
loop_jitter = 0
##
newsample = True  # just for testing the drawing of the wavefile
selector = 0  # this sets which part of the GUI is highlighted.
filebrowser_selected = 0  # which file is selected

LFO1 = 0  # this stores the LFO value (ie the multiplier)
LFO2 = 0
## calculate additional information needed
grain_length_samples = round(
    grain_length_ms * (int(Fs) / 1000)
)  # grain length now in samples
grain_samples = np.zeros(4)  # contains the audio data
grain_waittime_ms = (
    1000.0 / grains_per_second
)  # how long to wait after one grain is triggered
currentgrain = 1  # which grain is currently supposed to be triggered
playhead_position = 0  # position of the playhead in samples

## the constant sample (played as two channels to overlap a bit)

### SAMPLE PLAYER
if False:  # currently deactivated

    pygame.time.wait(300)
    constant_sample2 = pygame.mixer.Sound(
        sample2
    )  # this needs to be a sample that endlessly loopable
    constant_sample2.set_volume(0.05)
    pygame.mixer.Channel(1).play(constant_sample2, loops=-1, fade_ms=300)

LastLFOcall1 = datetime.datetime.now()
LastLFOcall2 = datetime.datetime.now()

port = mido.open_input(names[0])
# port = mido.open_input('MPK Mini Mk II 0')

changed = False  # only process the audio once

### start the sample playback
data_second = stretch(data_second, 10, 2 ** 13, 2 ** 11)
constant_sample = pygame.mixer.Sound(play_ready(data_second, 0))  # no envelope
constant_sample.set_volume(0.05)
#pygame.mixer.Channel(0).play(constant_sample, loops=-1, fade_ms=300) deactivaterd for now

app = App(width=800, height=480, bg="gray50")
# app.set_full_screen()
newd = Drawing(app, width="fill", height="fill")
dummy = Text(app, "")  # not sure this dummy procedure is really needed

filebrowsing = False

signal.signal(signal.SIGINT, signal_handler)


dummy.repeat(150, GUI)  # update the GUI every 300ms
# dummy.repeat(500, filebrowser) #update the GUI every 300ms
dummy.repeat(30, mainfunc)  # this will be the "work loop", update every 30ms
# dummy.repeat(1, update_rotaries)
app.display()
