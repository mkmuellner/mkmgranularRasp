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
            # If near the beginning, only allow po
