import os
import pygame
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
import struct
from guizero import App, Picture

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



app = App()

picture = Picture(app, image = "")
app.display()

sourceFileDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(sourceFileDir)

Fs, data = read('Ashlight_Sample-29.wav')
print(data[20000:20020])

#print(Fs) #Fs contains the sample rate now
channels = 2
pygame.mixer.pre_init(buffer = 2048*16, frequency = Fs, channels = channels) #lower buffer gives more clicks
# Remove pygame.init()
# pygame.init()

## note: I should convert all samples to mono for now.. makes it a LOT easier to handle them

pygame.mixer.init()

sound = pygame.mixer.Sound('Ashlight_Sample-29.wav')
sbuffer = sound.get_raw() #get the raw data of the larger wave file


data_s16 = np.frombuffer(sbuffer, dtype=np.int16, count=len(sbuffer)//2, offset=0)

print(data_s16[20000:20020])


hann = np.hanning(len(data_s16)) #this creates a hanning envelope. an array between 0 and 1.0

data_s16 = data_s16*hann
data_s16 = data_s16.astype('i2') #need to move to int again i2 is integer made up of 2 bytes (aka int16)

#plot the waveform
#plt.figure()
#plt.plot(data_s16)
#plt.plot(data_s16, color = "red")
#plt.show()

sbuffer = data_s16.tobytes()


grain1 = pygame.mixer.Sound(sbuffer)

s = pygame.mixer.Sound.play(grain1)

#t1 = round(time.time()*100)

while pygame.mixer.get_busy():
    time.sleep(0.02)

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

