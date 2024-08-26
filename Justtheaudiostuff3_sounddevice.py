import numpy as np
import sounddevice as sd
import soundfile as sf
from threading import Thread, Event
import queue
import time
import sys
import select
import os

# Import helper functions from audio_helpers.py
from audio_helpers import ms_to_samples, apply_envelope, generate_grain
from input_helpers import print_key_mappings, handle_keyboard_input

# Global variables
grain_size_ms = 200       # Grain size in milliseconds (will be converted to samples)
min_grain_size_ms = 50    # Minimum grain size in milliseconds
grain_density = 20        # Grains per second (default)
min_grain_density = 0.5   # Minimum grain density (1 grain every 2 seconds)
max_grain_density = 40    # Maximum grain density
random_offset = 500       # Base random offset for grain start position
random_extent = 1.0       # Extent of randomness around the playhead (multiplier for random_offset)
move_playhead = False     # Flag to determine if the playhead moves (default: False)
playhead_speed = 1.0      # Speed at which the playhead moves when advancing
playhead_direction = 1    # 1 for forward, -1 for backward
mix = 0.5                 # Determines the probability of using regular or reverse audio
random_grain_variation = 0  # Percent variation in grain size
envelope_type = 'soft'    # Default envelope type
random_grain_density_factor = 0  # Percent variation in grain density (default 0%)
grain_pitch = 1.0         # Grain pitch (1.0 is normal)
random_pitch_variation = 0  # Percent variation in grain pitch (default 0%)
# Binary variables to control function execution
apply_pitch = True
apply_envelope = True
apply_random_pitch = True
apply_random_grain_size = True
apply_random_grain_density = True
apply_random_position = True

# Global variable to control keyboard input
keyboard_input_enabled = True  # Set to False by default

# Audio settings - using the correct filename
filename = 'tori_amos_god_3.wav'
data, fs = sf.read(filename, dtype='float32')  # Load audio file in float32

# Convert to mono if the audio data is stereo
if len(data.shape) > 1:
    data = np.mean(data, axis=1).astype('float32')  # Convert to mono in float32

# Convert the data to float16 for further processing (after loading)
data = data.astype('float16')
data_reverse = data[::-1].astype('float16')  # Create reversed buffer in float16

usb_device_index = 2  # Replace with your actual USB Soundblaster device index

# Function to handle producing grains in a separate thread
def grain_producer(grain_queue, stop_event):
    global grain_size_ms, grain_density, random_offset, random_extent, move_playhead, playhead_speed, playhead_direction, mix

    current_position = 0

    while not stop_event.is_set():
        # Calculate grain size samples
        grain_size_samples = ms_to_samples(grain_size_ms, fs)

        # Apply random variation to grain density if enabled
        if apply_random_grain_density:
            random_density_variation = 1 + (random_grain_density_factor / 100.0) * (np.random.random() - 0.5) * 2
            effective_grain_density = max(min_grain_density, grain_density * random_density_variation)
        else:
            effective_grain_density = grain_density

        # Apply random position if enabled
        if apply_random_position:
            effective_random_offset = int(random_offset * random_extent)
            start_position = current_position + int(np.random.uniform(-effective_random_offset, effective_random_offset))
        else:
            start_position = current_position

        start_position = np.clip(start_position, 0, len(data) - grain_size_samples)

        # Generate the grain
        grain = generate_grain(
            data, data_reverse, start_position, grain_size_samples, envelope_type=envelope_type, mix=mix, 
            pitch=grain_pitch, pitch_variation=random_pitch_variation, apply_pitch=apply_pitch, 
            apply_envelope=apply_envelope, apply_random_pitch=apply_random_pitch, apply_random_grain_size=apply_random_grain_size
        )

        try:
            grain_queue.put_nowait(grain)  # Non-blocking put
        except queue.Full:
            pass  # Skip adding this grain if queue is full

        # Move playhead forward or backward if allowed
        if move_playhead:
            current_position += int(fs // effective_grain_density) * playhead_speed * playhead_direction
            current_position %= len(data)  # Loop around if necessary

        # Sleep for the calculated grain interval to match the grain density
        time.sleep(1 / effective_grain_density)



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

# Initialize the grain queue with a larger size for better buffering
grain_queue = queue.Queue(maxsize=100)  # Increased max size for more buffering

# Event to control the stopping of the grain producer thread
stop_event = Event()

# Start the grain production thread
producer_thread = Thread(target=grain_producer, args=(grain_queue, stop_event))
producer_thread.daemon = True

# Reduce the nice value of the current process to give higher priority (requires root)
try:
    os.nice(-10)  # Set a lower nice value for higher priority (range -20 to 19, lower is higher priority)
except PermissionError:
    print("Permission denied: cannot change process priority without root permissions.")

producer_thread.start()

# Start the sounddevice output stream with the callback
stream = sd.OutputStream(callback=audio_callback, samplerate=fs, blocksize=ms_to_samples(grain_size_ms, fs), device=usb_device_index)

# Pre-fill the grain queue to ensure smooth playback
print("Pre-filling grain queue...")
while not grain_queue.full():  # Only fill if there's space in the queue
    try:
        start_position = np.random.randint(0, len(data) - ms_to_samples(grain_size_ms, fs))
        grain = generate_grain(data, data_reverse, start_position, ms_to_samples(grain_size_ms,fs), envelope_type=envelope_type, mix=mix, pitch=grain_pitch, pitch_variation=random_pitch_variation)
        grain_queue.put_nowait(grain)
    except queue.Full:
        # If the queue is full, stop pre-filling
        print("Grain queue is full, stopping pre-fill.")
        break

# Start a thread for keyboard handling, only if keyboard input is enabled
if keyboard_input_enabled:
    print_key_mappings()
    keyboard_thread = Thread(target=handle_keyboard_input, args=(
        keyboard_input_enabled, grain_queue, move_playhead, playhead_direction, mix, 
        grain_size_ms, envelope_type, random_extent, random_grain_variation, playhead_speed, 
        grain_density, random_grain_density_factor, grain_pitch, random_pitch_variation,
        min_grain_size_ms, max_grain_density, min_grain_density
    ))
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
