import numpy as np
import sounddevice as sd
import soundfile as sf
from threading import Thread, Event
import queue
import random
import time
import keyboard

# Global variables for easy adjustment (e.g., via GPIO input)
grain_size = 2048         # Size of each grain in samples
grain_density = 20        # Grains per second
pitch_shift = 1.0         # Pitch shifting factor (1.0 = normal, -1.0 = reverse)
random_offset = 500       # Base random offset for grain start position
random_extent = 1.0       # Extent of randomness around the playhead (multiplier for random_offset)
move_playhead = False     # Flag to determine if the playhead moves (default: False)
playhead_speed = 1.0      # Speed at which the playhead moves when advancing
playhead_direction = 1    # 1 for forward, -1 for backward

# Audio settings - using the correct filename
filename = 'tori_amos_god_3.wav'
data, fs = sf.read(filename, dtype='float32')  # Load audio file

# Convert to mono if the audio data is stereo
if len(data.shape) > 1:
    data = np.mean(data, axis=1)  # Average the two channels to convert to mono

usb_device_index = 2  # Replace with your actual USB Soundblaster device index

# Envelope Types
def apply_envelope(grain, envelope_type='linear'):
    length = len(grain)
    if envelope_type == 'linear':
        envelope = np.linspace(0, 1, length // 2)
        envelope = np.concatenate((envelope, envelope[::-1]))  # Symmetric fade in/out
    elif envelope_type == 'exponential':
        envelope = np.linspace(1, 0.1, length)
    elif envelope_type == 'gaussian':
        mean = length // 2
        std_dev = length // 6
        envelope = np.exp(-0.5 * ((np.arange(length) - mean) ** 2) / (std_dev ** 2))
    else:
        envelope = np.ones(length)  # No envelope
    
    # Ensure the lengths match by trimming or padding if necessary
    if len(envelope) != len(grain):
        min_length = min(len(envelope), len(grain))
        envelope = envelope[:min_length]
        grain = grain[:min_length]

    return grain * envelope

# Granulation Function with Envelope
def generate_grain(data, start_sample, grain_size, pitch_shift=1.0, envelope_type='linear'):
    end_sample = min(len(data), start_sample + grain_size)
    grain = data[start_sample:end_sample]

    # Pitch shifting or reversing
    if pitch_shift != 1.0:
        grain = np.interp(
            np.arange(0, len(grain), pitch_shift),
            np.arange(0, len(grain)),
            grain
        )
    
    # Apply envelope
    grain = apply_envelope(grain, envelope_type)
    
    return grain

# Function to handle producing grains in a separate thread
def grain_producer(grain_queue, stop_event):
    global grain_size, grain_density, pitch_shift, random_offset, random_extent, move_playhead, playhead_speed, playhead_direction
    
    current_position = 0
    grain_interval_samples = int((fs // grain_density) * playhead_speed)  # Adjust speed of playhead
    
    while not stop_event.is_set():
        # Apply the extent of randomness to the random offset
        effective_random_offset = int(random_offset * random_extent)
        
        # Determine valid random offset range
        if current_position < effective_random_offset:
            # If near the beginning, only allow positive offset
            start_position = current_position + random.randint(0, effective_random_offset)
        elif current_position > (len(data) - grain_size - effective_random_offset):
            # If near the end, only allow negative offset
            start_position = current_position - random.randint(0, effective_random_offset)
        else:
            # In the middle, allow both positive and negative offsets
            start_position = current_position + random.randint(-effective_random_offset, effective_random_offset)
        
        # Ensure start_position stays within bounds
        start_position = max(0, min(len(data) - grain_size, start_position))
        
        # Generate grain at randomized start position
        grain = generate_grain(data, start_position, grain_size, pitch_shift)
        
        try:
            grain_queue.put_nowait(grain)  # Use non-blocking put
        except queue.Full:
            pass  # If the queue is full, just skip adding this grain

        # Move playhead forward or backward if allowed
        if move_playhead:
            current_position += grain_interval_samples * playhead_direction
            if current_position >= len(data):
                current_position = 0
            elif current_position < 0:
                current_position = len(data) - grain_size

        # Control the rate of grain production based on grain density
        time.sleep(1 / grain_density)

# Audio Callback Function for Real-Time Playback
def audio_callback(outdata, frames, time, status):
    if status:
        print(status)  # Print any errors or warnings
    try:
        grain = grain_queue.get_nowait()
        if len(grain) < len(outdata):
            outdata[:len(grain)] = grain.reshape(-1, 1)
            outdata[len(grain):] = 0  # Fill the rest with silence if grain is smaller
        else:
            outdata[:] = grain[:frames].reshape(-1, 1)
    except queue.Empty:
        outdata.fill(0)  # Output silence if no grains are available

# Initialize the grain queue
grain_queue = queue.Queue(maxsize=100)  # Max size to prevent overproduction

# Event to control the stopping of the grain producer thread
stop_event = Event()

# Start the grain production thread
producer_thread = Thread(target=grain_producer, args=(grain_queue, stop_event))
producer_thread.daemon = True
producer_thread.start()

# Start the sounddevice output stream with the callback
stream = sd.OutputStream(callback=audio_callback, samplerate=fs, blocksize=grain_size, device=usb_device_index)

# Pre-fill the grain queue to ensure smooth playback
print("Pre-filling grain queue...")
while not grain_queue.full():
    start_position = random.randint(0, len(data) - grain_size)
    grain = generate_grain(data, start_position, grain_size, pitch_shift)
    grain_queue.put_nowait(grain)

# Keyboard handling for real-time control
def handle_keyboard_input():
    global move_playhead, playhead_direction, pitch_shift, random_extent
    
    while True:
        if keyboard.is_pressed('f'):  # Move playhead forward once
            move_playhead = False
            playhead_direction = 1
            break
        elif keyboard.is_pressed('b'):  # Move playhead backward once
            move_playhead = False
            playhead_direction = -1
            break
        elif keyboard.is_pressed('k'):  # Keep moving playhead in current direction
            move_playhead = not move_playhead
            break
        elif keyboard.is_pressed('r'):  # Reverse grain playback
            pitch_shift *= -1  # Reverse the pitch shift direction
            break
        elif keyboard.is_pressed('u'):  # Increase randomness around playhead
            random_extent += 0.1  # Increase randomness
            break
        elif keyboard.is_pressed('i'):  # Decrease randomness around playhead
            random_extent = max(0, random_extent - 0.1)  # Decrease randomness
            break

# Start a thread for keyboard handling
keyboard_thread = Thread(target=handle_keyboard_input)
keyboard_thread.daemon = True
keyboard_thread.start()

# Start the audio stream and let it run indefinitely
with stream:
    print("Granular synthesis running. Press Ctrl+C to stop.")
    try:
        while True:
            sd.sleep(1000)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Stopping the granular synthesis.")
        stop_event.set()  # Stop the grain producer thread
        producer_thread.join()  # Wait for the thread to finish
