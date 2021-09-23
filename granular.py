import time #needed for delays/waits
from pygame import mixer #needed for sound
from random import seed #needed for random numbers
from random import randint #random integers

from RPi import GPIO #activate the GPIO pins for the rotary encoders

from guizero import App, Text, Drawing #import the GUI library

from os import walk #needed for file browsing



def GUI():
    global app
    global drawing
    app = App(title="MKMGranular", width = 800, height = 480, bg ="floralwhite", layout = "auto")
    drawing = Drawing(app, width = 800, height = 480)
    selector = Drawing(app, width = 800, height = 480)
    drawing.line(400,0, 400, 480, color = "red", width = 2) #center line
    

def listfiles():
    global linespacing
    linespacing = 19 #line spacing in pixels for things displayed - basically the raster size
    global f
    f = []
    for (dirpath, dirnames, filenames) in walk("/mnt/usb/NI_Ashlight_Grains"): #the usb stick is mounted on /mnt/usb
        f.extend(filenames) #load all file names into the f variable
        break

    for i in range(len(f)):
        drawing.text(12,i*linespacing,f[i], size = linespacing-1,color="gray")
        if i == 24: break # fit max 25 filenames per row
    drawing.repeat(2,update_rotaries) #2ms update timer for checking the rotary switches

def update_rotaries(): #updates the rotary encoder info from GPIO
        global oneA_Last
        global twoA_Last
        global threeA_Last
        global fourA_Last
        
        global counter_one
        global counter_two
        global counter_three
        global counter_four
                
        oneA_State = GPIO.input(Enc_oneA)
        oneB_State = GPIO.input(Enc_oneB)
        
        twoA_State = GPIO.input(Enc_twoA)
        twoB_State = GPIO.input(Enc_twoB)
        
        threeA_State = GPIO.input(Enc_threeA)
        threeB_State = GPIO.input(Enc_threeB)
        
        fourA_State = GPIO.input(Enc_fourA)
        fourB_State = GPIO.input(Enc_fourB)
                
        if oneA_State != oneA_Last:
            if oneB_State != oneA_State:
                if int(counter_one/2) < 24:
                    counter_one += 1
                    one_util = int(counter_one/2)
                    # do a little bit of GUI updating
                    drawing.rectangle(1,1+linespacing*(one_util),10,linespacing+linespacing*(one_util), color = "red")
                    drawing.rectangle(1,1+linespacing*(one_util-1),10,linespacing+linespacing*(one_util-1), color = "floralwhite") #paint over previous square
                    
                    drawing.text(12,linespacing*(one_util),f[one_util], size = linespacing-1,color="black")
                    drawing.text(12,linespacing*(one_util-1),f[one_util-1], size = linespacing-1,color="gray")
            else:
                if int(counter_one/2) > 0:
                    counter_one -= 1
                    one_util = int(counter_one/2)
                    drawing.rectangle(1,1+linespacing*(one_util),10,linespacing+linespacing*(one_util), color = "red")
                    drawing.rectangle(1,1+linespacing*(one_util+1),10,linespacing+linespacing*(one_util+1), color = "floralwhite")
                    
                    drawing.text(12,linespacing*(one_util),f[one_util], size = linespacing-1,color="black")
                    drawing.text(12,linespacing*(one_util+1),f[one_util+1], size = linespacing-1,color="gray")
                    
            print(f[one_util])
            mixer.stop()
            sound = mixer.Sound("/mnt/usb/NI_Ashlight_Grains/"+f[one_util]) #preview the sound
            sound.set_volume(0.1) #sets the volume for that sound
            sound.play(fade_ms = 1000)
                    
        oneA_Last = oneA_State
        
        if twoA_State != twoA_Last:
            if twoB_State != twoA_State:
                counter_two += 1
            else:
                counter_two -= 1
            print(int(counter_two/2))
        twoA_Last = twoA_State

        if threeA_State != threeA_Last:
            if threeB_State != threeA_State:
                counter_three += 1
            else:
                counter_three -= 1
            print(int(counter_three/2))
        threeA_Last = threeA_State
        
        if fourA_State != fourA_Last:
            if fourB_State != fourA_State:
                counter_four += 1
            else:
                counter_four -= 1
            
        fourA_Last = fourA_State        
  

#define the pins for the encoders. 1 is leftmost 4 is rightmost encoder

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

for pin in [16,20,21,10,9,11,17,27,22,2,3,4]: #initialize the pins we use
    GPIO.setup(pin ,GPIO.IN)

#initialize the counters

counter_one = 0
counter_two = 0
counter_three = 0
counter_four = 0

oneA_Last = GPIO.input(Enc_oneA) # previous state of the encoder
twoA_Last = GPIO.input(Enc_twoA)
threeA_Last = GPIO.input(Enc_threeA)
fourA_Last = GPIO.input(Enc_fourA)

#initialize the sound mixer
mixer.pre_init()
mixer.init(buffer = 2048*16, frequency = 44100) #lower buffer gives more clicks

GUI()
listfiles()
app.display()


sound = mixer.Sound('/home/pi/mkmgranular/samples/ambi_swoosh.wav')
sound.set_volume(0.2) #sets the volume for that sound

sbuffer = sound.get_raw() #get the raw data of the larger wave file
print(len(sbuffer))

grainsize = 300 #in ms
fadetime = int(grainsize / 2)

playing1 = False #variable to see if the sound has been triggered
fading1 = False
p1time = 0

# while True:
#     # start2 = randint(1,12)*10000+100 #random startpoint
#     # start3 = randint(1,12)*10000+100 #random startpoint
#     # start4 = randint(1,12)*10000+100 #random startpoint
#     
#     #end1 = start1 + grainsize
#     #end2 = start2 + grainsize
#     #end3 = start3 + grainsize
#     #end4 = start4 + grainsize
#     
#     #endpoint can not exceed sample size
#     """
#     if end1 > len(sbuffer):
#         end1 = len(sbuffer)-1
#         
#     if end2 > len(sbuffer):
#         end2 = len(sbuffer)-1
#     
#     if end3 > len(sbuffer):
#         end3 = len(sbuffer)-1
#     
#     if end4 > len(sbuffer):
#         end4 = len(sbuffer)-1
#     """    
#     
# 
#     
#     #create a new sound (subset)
#     #grain2 = mixer.Sound(sbuffer[start2:128610]) #create a new sound (subset)
#     #grain3 = mixer.Sound(sbuffer[start1:138610]) #create a new sound (subset)
#     #grain4 = mixer.Sound(sbuffer[start2:128610]) #create a new sound (subset)
# 
#     
#     #grain2.set_volume(0.01)
#     #grain3.set_volume(0.01)
#     #grain4.set_volume(0.01)
# 
#     #print(grain1.get_raw())
#     
#     if playing1 == False:
#         start1 = randint(1,9)*10000+100 #random startpoint
#         grain1 = mixer.Sound(sbuffer[start1:len(sbuffer)])
#         grain1.set_volume(0.05)
#         playing1 = True
#         grain1.play(loops = 0, maxtime = 0, fade_ms = fadetime) #fade is a fade in, maxtime is how long the sample plays both in ms
#         p1time = time.time_ns() #get the time in nanoseconds
#         print(fadetime)
#     #time.sleep(0.1)
#     #grain2.play(loops = 0, maxtime = 0, fade_ms = 20) #fade is a fade in, maxtime is how long the sample plays both in ms
#     #time.sleep(0.1)
#     #grain3.play(loops = 0, maxtime = 0, fade_ms = 20) #fade is a fade in, maxtime is how long the sample plays both in ms
#     #time.sleep(0.1)
#     #grain4.play(loops = 0, maxtime = 0, fade_ms = 20) #fade is a fade in, maxtime is how long the sample plays both in ms
#     #time.sleep(0.1)
#     if (fading1 == False) and ((time.time_ns()-p1time) / 1000000) > (grainsize-fadetime): #wait before you start the sounds again
#         grain1.fadeout(fadetime)
#         fading1 = True #activate fadeout, run only once
#         print("_")
#         
#     if ((time.time_ns()-p1time) / 1000000) > (grainsize):
#         playing1 = False
#         fading1 = False
#         print(".")
#         
      
