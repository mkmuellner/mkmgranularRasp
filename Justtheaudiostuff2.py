#I should make sure all samples are 22000 Hz sample rate

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
    global GUIneedsUpdate

    # reset everything. assume only one buttonpress is supposed to be recorded at a time
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
    GUIneedsUpdate = True

def turned_rotary_callback(channel):  # interrupt called
    global selector  # could also be using counter_one here instead
    global counter_two
    global counter_three
    global counter_four
    global GUIneedsUpdate

    oneB = GPIO.input(Enc_oneB)
    twoB = GPIO.input(Enc_twoB)
    threeB = GPIO.input(Enc_threeB)
    fourB = GPIO.input(Enc_fourB)

    if channel == Enc_oneA:
        if oneB == 1:
            selector += 1
        else:
            selector -= 1
    if (
        channel == Enc_twoA
    ):  # on that one the orientation (hardware) seems to be flipped
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
            
    GUIneedsUpdate = True
    # print(f"one: {selector}")
    # print(f"two: {counter_two}")
    # print(f"three: {counter_three}")
    # print(f"four: {counter_four}")


def filebrowser():
    global filebrowser_selected
    global selector
    global filebrowsing
    global oldselector
    global onepressed
    global sample1
    global sample2
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

        # make sure we don't allow the selector to become bigger than the max number of files shown
        if selector > len(filelist) - 1:
            selector = 0  # 14 items total
        if selector < 0:
            selector = len(filelist) - 1  # this wraps around when needed

        # now print the filenames

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
            #now decide which of the samples will be handed back
            if chosensample == 1:
                sample1 = filelist[selector]
            if chosensample == 2:
                sample2 = filelist[selector]
            
            loadsamples()
            filebrowsing = False  # exit browser
            selector = 1  # reset the selector
            newd.clear()
            global newsample
            newsample = True
            print("switching")
        GUIneedsUpdate = False #


def loadsamples():  # load a new sample 1 or 2
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
    global len_data
    global len_data_second
    global Right_Limit_second
    global Left_Limit_second

    # read the wave file and give some stats about it
    Fs, data = read(sample1)  # read the wave file
    Fs_second, data_second = read(sample2)

    data = data[:, 0]  # only process the left channel
    data_second = data_second[:, 0]  # only process the left channel
    len_data = len(data) #update length variables so I don't have to constantly calc them
    len_data_second = len(data_second)
    data_backup = data  # back up the original data to be able to reset
    data_backup_second = data_second
    Right_Limit = len_data
    Left_Limit = 1
    Right_Limit_second = len_data_second
    Left_Limit_second = 1
    #create the files
    my_monitors_dpi = 96
    plt.figure(
        figsize=(370 / my_monitors_dpi, 74 / my_monitors_dpi),  # 370
        dpi=my_monitors_dpi,
    )  # need to change DPI value here for the small monitor
    plt.axis("off")
    plt.xlim([0, len_data])
    plt.plot(data, color="black")
    plt.savefig(fname="AudioA.png", bbox_inches="tight", transparent=True)  #
    plt.close()
    plt.figure(
        figsize=(370 / my_monitors_dpi, 74 / my_monitors_dpi),  # 370
        dpi=my_monitors_dpi,
    )  # need to change DPI value here for the small monitor
    plt.axis("off")
    plt.xlim([0, len_data_second])
    plt.plot(data_second, color="black")
    plt.savefig(fname="AudioB.png", bbox_inches="tight", transparent=True)  #
    plt.close()   

def smoothen(dta,size = 20,times = 1):  # reduce the harshness by running a simple running average as a lowpass filter
    # I can use the same function as a filter (low pass)
    ret = dta
    
    for i in range(times):
        ret = uniform_filter1d(ret, size)
    
    return ret  # size gives the amount of smoothing


def speed_up(dta, shift):  # make the sound play faster (and higher in pitch)
    return np.delete(dta, np.arange(0, dta.size, shift))
    # delete every "shift"th element in the array.


def decfun(x):
    y = 2 ** (
        (-0.01 / (grain_length_ms / 30)) * x
    )  # the grain lethgth divided by 100 kinda works out
    return y

def fillfadefunc(leng):
    global fadefunc
    for i in range(leng-1):
        fadefunc = np.append(fadefunc, 1/(i+1))

    


def env_fadein_out(dta):
    global fadefunc
    dtalength = len(dta)
    fadelength = len(fadefunc)
    dta = dta * np.pad(fadefunc,(0,dtalength-fadelength), 'constant', constant_values = (0,1)) #fadein and 1's
    dta = dta * np.pad(np.flipud(fadefunc),(dtalength-fadelength,0), 'constant', constant_values = (1,0)) #fadein and 1's
    return(dta)

def env_fadein_out2(dta, fadelength): # an alternative where the fadelength is given as an argument. Might generally be a cleaner solution.
    dtalength = len(dta)
    dta = np.pad(dta[fadelength:dtalength-fadelength], (fadelength,fadelength), mode = "linear_ramp", end_values = (0,0))
    return(dta)


def env_hann(dta):
    return dta * np.hanning(
        len(dta)
    )  # this creates a hanning envelope. an array between 0 and 1.0


def env_decay(dta):
    q = np.arange(1, len(dta) + 1)
    q = decfun(q)
    q = env_fadein_out2(q, 80) #blunt the extreme spikes a bit
    #now take down the extreme spike at the start a bit to avoid popping
    return dta * q  # this creates a hanning envelope. an array between 0 and 1.0


def env_exp(dta):
    q = np.arange(1, len(dta) + 1)
    q = decfun(q)
    q = env_fadein_out2(q, 80)
    q = np.flip(q)
    return dta * q  # this creates a hanning envelope. an array between 0 and 1.0


def reverse(dta):  # reverses the sample data
    return np.flipud(dta)


def limiter(dta):
    return 0


# consider putting multiple grains into one array to have something longer to play every time.
def play_ready(dta, envtype):  # all actions needed to play dta (as int16)
    # copy left channel onto right channel
    global currentgrain
    global grain_length_ms
    global soundloop_times
    global Pitch1

    if Pitch1 != 1:
        dta = pitchshift(dta, Pitch1)

    ##reverse
    if reversegrain:
        dta = reverse(dta)

    if True:  # speed change
        dta = speed_down(dta, random.randrange(1, 6))

    #just for dev
    useLFO = False
        
    if True:  # might want to be able to turn smoothing off
        if useLFO:
            dta = smoothen(dta, int(LFO1*100))
        else:
            dta = smoothen(dta,2)

    if envtype == 1:  # hann envelope
        dta = env_hann(dta)
    elif envtype == 2:  # decay 1/x envelope
        dta = env_decay(dta)
    elif envtype == 3:  # exp envelope
        dta = env_exp(dta)
    
    dta = np.vstack(
        (dta, dta)
    ).T  # duplicate processed channels to create a "pseudo stereo" one to play"
    dta = dta.astype("i2")  # convert data to 16 bit int format
    sounddata = dta.tobytes()  # convert to buffer (sound data)
    # doesn't actually seem to make much of a difference
    # compared to just running a single call to pygame.Mixer.Sound.play with a single sound.
    
    return sounddata

def FX(dta, envtype = 1, smooth = True): #this will replace the play_ready function
    global currentgrain
    global grain_length_ms
    global soundloop_times
    global Pitch1

    ##reverse

    if Pitch1 !=0:
        dta = pitchshift(dta,Pitch1*100+int((0.5-random.random())*pitch1_jitter*100))
    #if smooth:  # might want to be able to turn smoothing off
    #    dta = smoothen(dta)
    if envtype == 1:  # hann envelope
        dta = env_hann(dta)
    elif envtype == 2:  # decay 1/x envelope
        dta = env_decay(dta)
    elif envtype == 3:  # exp envelope
        dta = env_exp(dta)
    
    return(dta)

def play_ready_deprec(dta):
    if normalize_on:
        dta = normalize(dta) #normalize the grain
    dta = smoothen(dta) #this is full output level smoothing
    #print(f"max:{np.max(abs(dta))}")
    dta = np.vstack(
    (dta, dta)
    ).T  # duplicate processed channels to create a "pseudo stereo" one to play"
    dta = dta.astype("i2")  # convert data to 16 bit int format
    sounddata = dta.tobytes()  # convert to buffer (sound data)
    return sounddata

def next_grain(
    data_select, playhead_position, playhead_jitter, length_jitter
):  # extract the next grain from full sample "data".
    global grain_length_samples
    global Fs
    global data
    global data_second
        
    while True: #repeat until we have a non-zero sized grain
        if data_select == 1:
            dataex = data
            sample_length = grain_length_samples
            lgth = len_data
        if data_select == 2:
            dataex = data_second
            sample_length = grain_length_samples
            lgth = len_data_second
        
        jitter = int(sample_length * playhead_jitter * ((0.5 - random.random())))
        
        ex_position = playhead_position + jitter

        if ex_position > (lgth - grain_length_samples - 1):
            ex_position = lgth + (sample_length - ex_position)
        if ex_position < 0:
            ex_position = abs(ex_position)
        endposition = (
            ex_position
            + grain_length_samples
            + int(grain_length_samples * length_jitter * (0.5 - float(random.random())))
        )
        extracted = dataex[ex_position:endposition]
        
        grain_length_samples = len(extracted)
        if grain_length_samples == 0:
            print("!")
        
        if grain_length_samples > 0:
            break
    
    # make a little dot at grainpos
    #xposA1 = (300 - 5) / len_data * ex_position + 5
    #newd.text(xposA1, 65, "|", color="white")
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


def saveplot(dta, filename):
    my_monitors_dpi = 96
    plt.figure(
        figsize=(370 / my_monitors_dpi, 74 / my_monitors_dpi),  # 370
        dpi=my_monitors_dpi,
    )  # need to change DPI value here for the small monitor
    plt.axis("off")
    plt.xlim([0, len(dta)])
    plt.plot(data, color="black")
    plt.savefig(fname=filename, bbox_inches="tight", transparent=True)  #
    plt.close()

def pitchshift(dta, shift):  # """ Changes the pitch of a sound by ``n`` semitones. """
    fr = 100 #frames to process at a time
    sz = 22000//fr #22000 would be the rate. samplepoints to process per turn.
    #print(f"sz:{sz}") #220
    
    c = int(len(dta)/sz) #frames that need to be processed. 220 samplepoints at a time
    #print(f"c:{c}") #9
    
    shiftx = int(shift//fr) #shift in Hz. Shiftx is samples shift per window.
    #print(f"shiftx:{shiftx}") #3
    
    #saveplot(dta, "untransformed_grain.png")
    combined = np.empty(shape = (0,0))
    for num in range(c):
        dtalet = dta[sz*num:(sz*(num+1))-1]
        rf = np.fft.fft(dtalet)
        rf = np.roll(rf, shiftx) #shift the frequencies by "shiftx"
        
        if shiftx > 0:
            rf[0:shiftx] = 0 #overwrite the wrapped around frequencies with zeros.
        if shiftx < 0:
            rf[len(rf)-shiftx:] #for the other shift direction
        nr = np.fft.irfft(rf) #back from fourier transform
        ns = np.ravel(nr)
        combined = np.append(combined, ns)
    #saveplot(combined, "combined_grain.png")
    return (combined)


def crossfade(dta_out, dta, fademult):
    
    if len(dta) == 0:
        print("dta is zero")
    #print(f"fademult:{fademult}")
    if fademult < 1:
        fadelength = int((1-fademult) * len(dta)) # this way 0.1 becomes 0.9 or 90% of grain length

    if fademult > 1:
        fadelength = int((fademult-1) * len(dta)) #this way 1.1 becomes a pause of 0.1 sample length

    if (len(dta_out) > 0): #as long as this isn't the first grain
        #now take half of data and crossfade
        dta_length = len(dta)
        out_length = len (dta_out)
        dta = FX(dta,envtype, True) #apply the FX
        
        
        if fademult < 1: #here we are fading (in samples)
            cross_dta = dta[0:fadelength] #the left part of the new bit - the one that will get faded
            rest_dta = dta[fadelength+1:] # the right part of the new bit - the one that will remained unchanged.
            
            cross_dta_out = dta_out[out_length-fadelength:] # the right but that needs to get blended
            rest_dta_out = dta_out[0:out_length -fadelength-1] # get the bit that doesn't need to be blended
            
            blended = 0.5*cross_dta_out + 0.5*cross_dta #now blend the subfragments
            
            dta_out = np.append(rest_dta_out, blended) #and assemble ...
            dta_out = np.append(dta_out, rest_dta) #... all of it.
        
        else: #and here we are not fading any more but rather creating space between
            if fademult > 1: #don't do this if fadelength == len(dta)
                #print(f"fadelength:{fadelength}")
                dta = np.pad(dta,(fadelength,0)) #add zeroes to create some space
                dta_out = np.append(dta_out, dta) #join the two together
                #print(f"dta_out_len:{len(dta_out)}")
            if fademult == 1: #exactly zero space
                dta_out = np.append(dta_out, dta)
        
    else:
        dta_out = np.append(dta_out,FX(dta,envtype,True))
    
    return(dta_out)
      

def mixgrains(dta1, dta2): #mixes two grains equally (unequal sizes allowed)
    l1 = len(dta1)
    l2 = len(dta2)
    diffr = abs(l1-l2)
    if (l1 > l2):
        dta2 = np.pad(dta2, (0,diffr)) #make the two grains equal length
    if (l2 > l1):
        dta1 = np.pad(dta1, (0,diffr)) #make the two grains equal length                              
    return((0.5*dta1) + (0.5*dta2)) #mix the grains together
    #return(dta1) #debugging



def graintrigger(
    pdata
):  # this now allows to change up the datafile to pick the grain from
    global grain1
    global graincounter
    global dta_out
    global envtype
    
    mode = 4 #
    
    if mode == 1:
        update_playhead()
        dta = next_grain(pdata, playhead_position, playhead_jitter, length_jitter)
        dta = FX(dta, 1, True)
        grain1 = pygame.mixer.Sound(play_ready_deprec(dta))
        
        #experimental to test stuff:
        #dta = np.vstack((dta, dta)).T  # duplicate processed channels to create a "pseudo stereo" one to play"
        #dta = dta.astype("i2")  # convert data to 16 bit int format
        
        
        ##// experimental
        volume_correction_factor = 0.06  # volume of 1 is ususally too loud for headphones

        calcvolume = volume_correction_factor * Volume1 + volume1_jitter * (
            0.5 - random.random()
        )  # add the jitter.
        if calcvolume > 1:
            calcvolume = 1
        if calcvolume < 0.01:
            calcvolume = 0.01
        grain1.set_volume(calcvolume)  # set the volume
        ## end of volume control

        pygame.mixer.Sound.play(grain1, loops=0)
        
            
        #grain1 = pygame.mixer.Sound(play_ready(dta, envtype))
        #graincounter = grain_release

    if mode ==2: #link together 4 grains. should do in a loop next
        dta_out = np.empty(shape = (0,0))
        
        for i in range(2): #going higher here prevents drawing of the GUI
            update_playhead()
            dta = next_grain(pdata, playhead_position, playhead_jitter, length_jitter)
            dta_out = np.append(dta_out,FX(dta,1,True))
            #print(i)
        dta = dta_out
        grain1 = pygame.mixer.Sound(play_ready_deprec(dta))
        graincounter = grain_release
        
    if mode  == 3: #this is the mix mode. One grain from each sample. does not seem to work right now
        if graincounter == 0:
            dta_out = np.empty(shape = (0,0))
        
        for i in range(3): #going higher here prevents drawing of the GUI
            update_playhead()
            graincounter += 1
            dta1 = next_grain(data, playhead_position, playhead_jitter, length_jitter)
            dta2 = next_grain(data_second, playhead_position_second, playhead_jitter, length_jitter) #for now we use the same jitters
            l1 = len(dta1)
            l2 = len(dta2)
            diffr = abs(l1-l2)
            if (l1 > l2):
                dta2 = np.pad(dta2, (0,diffr)) #make the two grains equal length
            if (l2 > l1):
                dta1 = np.pad(dta1, (0,diffr)) #make the two grains equal length                              
                              
            
            dta = (0.5*dta1) + (0.5*dta2) #mix the grains together
            dta_out = np.append(dta_out,FX(dta,envtype,True))
            
    if mode  == 4: #this is the mix mode. One grain from each sample and crossfaded into the previous
        
        if graincounter == 0:
            dta_out = np.empty(shape = (0,0))
        
        #debug:
        grain_release = 10
        
        for i in range(3): #going higher here prevents drawing of the GUI
            update_playhead()
            graincounter += 1
            dta1 = next_grain(1, playhead_position, playhead_jitter, length_jitter)
            dta2 = next_grain(2, playhead_position_second, playhead_jitter, length_jitter) #for now we use the same jitters
            
            dta = mixgrains(dta1, dta2)
            dta_out = crossfade(dta_out, dta, fademult)
            
      
            #print(i)
        
        if graincounter >= grain_release: #play when the file is long enough
            #print(len(dta))
            #dta = normalize(dta)
            #dta_out = env_fadein_out(dta_out) #experimental
            #dta_out = env_hann(dta_out)
            grain1 = pygame.mixer.Sound(play_ready_deprec(dta_out))
            graincounter = 0  # reset the counter
                
                
            # dta = speed_down(data, 12+round(LFO1*10)) #get some pitch variation with the LFO (just a test)
            # dta = cube_softclip(dta, 1)
            
            # volume control grain 1
            volume_correction_factor = 0.06  # volume of 1 is ususally too loud for headphones

            calcvolume = volume_correction_factor * Volume1 + volume1_jitter * (
                0.5 - random.random()
            )  # add the jitter.
            if calcvolume > 1:
                calcvolume = 1
            if calcvolume < 0.01:
                calcvolume = 0.01
            grain1.set_volume(calcvolume)  # set the volume
            ## end of volume control

            if loop_jitter > 0:
                rnd = random.randrange(loop_jitter)
            else:
                rnd = 0
                
            if(pygame.mixer.find_channel() is not None): #if there is a channel available, don't stop one for playing.
                
                pygame.mixer.Sound.play(grain1, loops=soundloop_times + rnd)
                #np.savetxt(f'sound{pygame.time.get_ticks()}.csv', dta_out, delimiter=';') #debug: save as csv files
                

def normalize(dta):

    dta = (2 ** (16 - 2)) * dta / dta.max()  # normalize (16bit)
  
    return(dta) #do nothing for now

def GUI(): #this slows sound playback significantly. I wonder if I should use music instead of play
     
    global selector
    global filebrowsing
    global GUIneedsUpdate
    global fademult
    #if True:
    if GUIneedsUpdate:
        if not (filebrowsing):
            #newd.clear() # clear does not seem to be necessary
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
            global playhead_position_second
            global newsample
            global grain_length_samples
            global grain_waittime_ms
            global reversegrain
            global Right_Limit
            global Left_Limit
            

            newd.image(0, 0, image=im1) #this seems to be somewhat slow

            # limiters
            # leftmost is 5 rightmost is 300
            Left_Limiter_Pos = 5 + round(Left_Limit / len_data * (300 - 5))
            Right_Limiter_Pos = 5 + round(Right_Limit / len_data * (300 - 5))
            #Left_Limiter_Pos2 = 315 + round(Left_Limit_second / len_data_second * (300 - 5))
            #Right_Limiter_Pos2 = 315 + round(Right_Limit_second / len_data_second * (300 - 5))
            
            newd.text(Left_Limiter_Pos, 20, "|", color="white")  # y = 58 is center
            newd.text(Right_Limiter_Pos, 20, "|", color="white")
            
            #newd.text(Left_Limiter_Pos2, 20, "|", color="white")  # y = 58 is center
            #newd.text(Right_Limiter_Pos2, 20, "|", color="white")

            newd.text(145, 127, "{:.2f}".format(Volume1))
            newd.text(145, 127 + 26 * 1, "{:.2f}".format(Pitch1))
            newd.text(145, 127 + 26 * 2 - 3, Tuning1)
            newd.text(145, 127 + 26 * 3 + 18, round(grain_length_ms))
            newd.text(145, 127 + 26 * 3 + 18 + 24, envtype)
            newd.text(145, 127 + 26 * 3 + 18 + 24 * 2, playhead_speed)
            newd.text(145, 127 + 26 * 3 + 18 + 24 * 3, soundloop_times)
            newd.text(145, 127 + 26 * 3 + 18 + 24 * 4, "{:.2f}".format(fademult))

            # left jitters
            newd.text(145 + 100, 127, "{:.2f}".format(volume1_jitter))
            newd.text(145 + 100, 127 + 26 * 1, "{:.2f}".format(pitch1_jitter))
            newd.text(145 + 100, 127 + 26 * 3 + 18, "{:.2f}".format(length_jitter))
            newd.text(
                145 + 100, 127 + 26 * 3 + 18 + 24 * 2, "{:.2f}".format(playhead_jitter)
            )
            newd.text(145 + 100, 127 + 26 * 3 + 18 + 24 * 3, loop_jitter)

            # LFO
            newd.text(675, 75, "{:.2f}".format(LFO1_parameter1))
            newd.text(675, 73 + 75, "{:.2f}".format(LFO2_parameter1))
            newd.text(675, 70 + 75 * 2, "{:.2f}".format(LFO3_parameter1))
            newd.text(675, 70 - 3 + 75 * 3, "{:.2f}".format(LFO4_parameter1))

            # flip on
            # newd.image(225, 179, image="FLIP_on.png") # then flip it to off if needed.
            if reversegrain:
                newd.image(225, 179, image="FLIP_on.png")  # then flip it to off if needed.
            else:
                newd.image(225, 179, image="FLIP_off.png")  # then flip it to off if needed.

            # picture is 302 x 74 at position 318, 30
            # draws a picture of the waveform

            newd.image(5, 30, image="AudioA.png") #left and right waveform
            newd.image(315, 30, image="AudioB.png")
            
            #print(playhead_position)
            #print(playhead_position_second)
            # mark playhead position
            #if mark_playhead == True:
            xposA = (300 - 5) / len_data * playhead_position + 5
            newd.text(xposA, 65, "|", color = "red")
           #newd.line(xposA, 40, xposA, 90, color="red")

            xposA2 = (300 - 5) / len_data_second * playhead_position_second + 315
            newd.text(xposA2, 65, "|", color = "red")
                #newd.line(xposA2, 40, xposA2, 90, color="red")

            # these two below here only need to be updated when the GUI value changes anyway
            #grain_length_samples = round(
            #    grain_length_ms * (int(Fs) / 1000)
            #)  # grain length now in samples
            #grain_waittime_ms = (
            #    1000.0 / grains_per_second
            #)  # how long to wait after one grain is triggered
            
            GUIneedsUpdate = False #reset the update marker


def mainfunc():
    global filebrowsing
    global selector
    global oldselector
    global reversegrain
    global chosensample
    if filebrowsing:
        reversegrain = False
        filebrowser() #chosensample contains 1 or 2 for A or B
    else:
        global dta
        global playhead_position
        global playhead_position_second
        global changed
        global playhead_reversed
        global playhead_reversed_second
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
        global twopressed
        global threepressed
        global fourpressed
        global grain_length_ms
        global length_jitter
        global playhead_speed
        global playhead_jitter
        global soundloop_times
        global fademult
        global speed_jitter
        global envtype
        global Right_Limit
        global Left_Limit
        global loop_jitter
        global Right_Limit_second
        global Left_Limit_second
        global A_manual
        global playhead_speed_mult
        
        
        if oldselector != selector:  # if the selector changed reset the counters
            counter_two = 0
            counter_three = 0

        if selector == 1:
            im1 = "GUI_perform_480_SoundA.png"

            Left_Limit = int(Left_Limit + counter_two * 4 * len_data / 100)
            Right_Limit = int(Right_Limit + counter_three * 4 * len_data / 100)

            if Left_Limit < 1:
                Left_Limit = 1
            if Right_Limit > len_data:
                Right_Limit = len_data
            if (
                Left_Limit > Right_Limit
            ):  # if left and right cross, just swap the values
                holder = Right_Limit
                Right_Limit = Left_Limit
                Left_Limit = holder

        elif selector == 0:
            im1 = "GUI_perform_480_A_Soundfile.png"

        elif selector == 2:
            im1 = "GUI_perform_480_A_volume.png"

            # changing volume
            Volume1 = Volume1 + (counter_two * 5 / 100)
            if Volume1 > 1:
                Volume1 = 1
                counter_two = 0
            if Volume1 < 0:
                Volume1 = 0.01
                counter_two = 19
            volume1_jitter = volume1_jitter + counter_three * 2 / 100
            if volume1_jitter < 0:
                volume1_jitter = 0

            if volume1_jitter > 1:
                volume1_jitter = 1

        elif selector == 3:
            im1 = "GUI_perform_480_A_pitch.png"

            # changing pitch
            # this does not work yet, anything other than 1 crashes
            Pitch1 = Pitch1 + (counter_two)
            if Pitch1 > 10:
                Pitch1 = 10
            if Pitch1 < -10:
                Pitch1 = -10
            pitch1_jitter = pitch1_jitter + counter_three/2
            if pitch1_jitter < 0:
                pitch1_jitter = 0

            if pitch1_jitter > 10:
                pitch1_jitter = 10

        elif selector == 4:
            im1 = "GUI_perform_480_A_Tuning.png"

        elif selector == 5:
            im1 = "GUI_perform_480_A_grainsize.png"

            # changing grainsize
            grain_length_ms = grain_length_ms + (counter_two * 20)
            if grain_length_ms > 1000:
                grain_length_ms = 1000

            if grain_length_ms < 30:
                grain_length_ms = 30

            length_jitter = length_jitter + counter_three * 2 / 100
            if length_jitter < 0:
                length_jitter = 0

            if length_jitter > 1:
                length_jitter = 1

        elif selector == 6:
            im1 = "GUI_perform_480_A_envtype.png"

            envtype = envtype + counter_two
            if envtype > 3:
                envtype = 1
            if envtype < 1:
                envtype = 3

        elif selector == 7:
            im1 = "GUI_perform_480_A_playspeed.png"

            # changing playhead
            playhead_speed = playhead_speed + (counter_two * 20)
            if playhead_speed > 2000:
                playhead_speed = 2000

            if playhead_speed < 20:
                playhead_speed = 20

            playhead_jitter = playhead_jitter + counter_three * 1 / 4
            if playhead_jitter < 0:
                playhead_jitter = 0

            if playhead_jitter > 10:
                playhead_jitter = 10

            # did not include speed_jitter just now

        elif selector == 8:
            im1 = "GUI_perform_480_A_grainloops.png"

            # changing loops

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

            # changing pausetime
            fademult = fademult + (counter_two / 4)
            if counter_two != 0:
                GUIneedsUpdate = True
            if fademult > 5: #this would be 5 samples distance.
                fademult = 5
            if fademult < 0.1:
                fademult = 0.1 #this would be 90% overlap.

        elif selector == 10:
            im1 = "GUI_perform_480_B_Soundfile.png"
            
            Left_Limit_second = int(Left_Limit_second + counter_two * 4 * len_data_second / 100)
            Right_Limit_second = int(Right_Limit_second + counter_three * 4 * len_data_second / 100)

            if Left_Limit_second < 1:
                Left_Limit_second = 1
            if Right_Limit_second > len_data_second:
                Right_Limit_second = len_data_second
            if (
                Left_Limit_second > Right_Limit_second
            ):  # if left and right cross, just swap the values
                holder = Right_Limit_second
                Right_Limit_second = Left_Limit_second
                Left_Limit_second = holder


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
                chosensample = 1
            if selector == 10:
                filebrowsing = True
                chosensample = 2
            if selector == 1:
                A_manual = not(A_manual)
        
        if threepressed:
            threepressed = False
            if selector == 4:
                reversegrain = not(reversegrain)
                if reversegrain:
                    data = reverse(data)
                    saveplot(data,"AudioA.png")
                    GUIneedsUpdate = True

                
        if A_manual == True:
            playhead_speed_mult = 0
        if A_manual == False:
            playhead_speed_mult = 1
            
        # reset things again
        counter_two = 0
        counter_three = 0

        oldselector = selector
        
        midi = False
        
        if midi:
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
        
        #t_on()
        graintrigger(data)
        #t_off()
        #4 runs 7,000 - 18,000 ns
        #3 runs 3 - 12 ms
        
        

def t_on(): #start the timer
    global before
    before = time.time()*1000
    
def t_off(): #end the timer
    global before
    global t_measured
    
    if len(t_measured) < 10: #until 10 measurements are done add the time differences to a vector
        t_measured = np.append(t_measured, time.time()*1000-before)
    else:
        outp_num = np.average(t_measured)
        outp = "{:.2f}".format(outp_num)
        print(f"exec time (ms):{outp}")
        t_measured = np.zeros(0)

def update_playhead():
    global playhead_position
    global playhead_position_second
    global speed_jitter
    global playhead_speed
    global playhead_reversed
    global playhead_reversed_second
    global playh_xposA
    global playh_xposA2

    # pygame.time.wait(pausetime1)
    if not (playhead_reversed):

        playhead_position = int(
            playhead_position + playhead_speed*playhead_speed_mult + speed_jitter * (random.random())
        )
    else:
        playhead_position = int(
            playhead_position - playhead_speed*playhead_speed_mult - speed_jitter * (random.random())
        )

    if playhead_position > Right_Limit:
        playhead_position = Right_Limit - grain_length_samples
        playhead_reversed = True
        #print("playhead reverse two")
    if playhead_position < Left_Limit:
        playhead_position = grain_length_samples + Left_Limit
        playhead_reversed = False
        #print("playhead forward two")
        
# second sample
    if not (playhead_reversed_second):
        playhead_position_second = int(
            playhead_position_second + playhead_speed + speed_jitter * (random.random())
        )
    else:
        playhead_position_second = int(
            playhead_position_second - playhead_speed - speed_jitter * (random.random())
        )

    if playhead_position_second > Right_Limit_second:
        playhead_position_second = Right_Limit_second - grain_length_samples
        playhead_reversed_second = True
        #print("playhead reverse")
    if playhead_position_second < Left_Limit_second:
        playhead_position_second = grain_length_samples + Left_Limit_second
        playhead_reversed_second = False
        #print("playhead forward")
        


## config variables
t_measured= np.zeros(0)

normalize_on = False
buffsz = 1024 #buffersize
before = 0 #initialize the global timer variable
graincounter = 0 #this counts how many grains have been added to the new array yet.
grain_release = 10 #how many grains are in each cloud before it plays
chosensample = 0 #initialize
A_manual = False
mark_playhead = True # should the playhead be drawn as a red line on top of the waveform
GUIneedsUpdate = True #draw the GUI, then check if we need to update it.

fadefunc = 0
fademult = 0.5 #this controls the blending of the grains as fold of grainlength. 0,5 would be blended at the midpoint
fillfadefunc(200) #fill fadefunc with a linear fade 200 samples long

playh_xposA = 0
playh_xposA2 = 0

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

sample1 = "tori_22000.wav" #this is detected as being sample rate 48khz!!
sample2 = "Ashlight_Smp_12_22000.wav"

# read the wave file and give some stats about it
Fs, data = read(sample1)  # read the wave file
Fs_second, data_second = read(sample2)

##initialize sound output via pygame
channels = 8
# pygame.mixer.pre_init(buffer = 2048*16, frequency = Fs, channels = channels) #lower buffer gives more clicks
pygame.mixer.pre_init(
    buffer=buffsz, frequency=22000, channels=channels #consider playing around with the sample rate here
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

len_data = len(data)
len_data_second = len(data_second)

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
reversegrain = False  # should the grain be reversed
grain_length_ms = 90.0  # in milliseconds (global)
grains_per_second = 4.0  # how many grains are triggered per second
number_of_grains = 4  # how many grain channels are there (for pygame)
playhead_speed = 250  # playhead movement in samples per second
speed_jitter = 0
playhead_jitter = 0.0  # jitter around the playhead as a factor. 1,0 = 10% of full sample size 0 = no jitter.
length_jitter = 0.0  # fold of original grain length
playhead_reversed = (
    False  # initial direction the playhead takes to trigger the samples.
)
playhead_reversed_second = False
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
Right_Limit = len(data)  # pretty random number to initialize
Left_Limit = 1
Right_Limit_second = len(data_second)  # pretty random number to initialize
Left_Limit_second = 1

envtype = 1
Volume1 = 1.0
Volume2 = 1.0
Tuning1 = "C"
Tuning2 = "C"
#Pitch1 = 1.0
Pitch1 = 3.0
Pitch2 = 1.0
pausetime1 = 10
volume1_jitter = 0
volume2_jitter = 0
pitch1_jitter = 0
loop_jitter = 0
##
onepressed = False
twopressed = False
threepressed = False
fourpressed = False


dta_out = np.empty(shape = (0,0)) #the grain target file
#

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
playhead_position_second = 0
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
#data_second = stretch(data_second, 10, 2 ** 13, 2 ** 11)
#constant_sample = pygame.mixer.Sound(play_ready(data_second, 0))  # no envelope
#constant_sample.set_volume(0.05)
# pygame.mixer.Channel(0).play(constant_sample, loops=-1, fade_ms=300) deactivaterd for now

app = App(width=800, height=480, bg="gray50")
# app.set_full_screen()
newd = Drawing(app, width="fill", height="fill")
dummy = Text(app, "")  # not sure this dummy procedure is really needed

filebrowsing = False

signal.signal(signal.SIGINT, signal_handler)
loadsamples()

dummy.repeat(500, GUI)  # update the GUI every 300ms setting this value makes a big difference
# dummy.repeat(500, filebrowser) #update the GUI every 300ms
dummy.repeat(30, mainfunc)  # this will be the "work loop", update every 30ms
dummy.repeat(45, mainfunc)  #doing this twice seems to work as a workaround to prevent segmentation errors. I don't know why
# dummy.repeat(1, update_rotaries)
app.display()
