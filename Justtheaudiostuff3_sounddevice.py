import numpy as np
import sounddevice as sd
import soundfile as sf
from threading import Thread
import queue

# Audio settings - using the correct filename
filename = 'tori_amos_god_3.wav'
data, fs = sf.read(filename, dtype='float32')  # Load audio file

# Convert to mono if the audio data is stereo
if len(data.shape) > 1:
    data = np.mean(data, axis=1)  # Average the two channels to convert to mono

usb_device_index = 2  # Replace with your actual USB Soundblaster device index

# Granular synthesis parameters
grain_size = 2048     # Size of each grain in samples
grain_interval = 1024  # Hop size for each grain (overlap or gap between grains)
pitch_shift = 1.0      # Modify pitch by stretching or compressing the grains (1.0 means no pitch shift)

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

    # Pitch shifting
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
def grain_producer(data, grain_size, grain_interval, pitch_shift):
    current_position = 0
    while True:
        grain = generate_grain(data, current_position, grain_size, pitch_shift)
        grain_queue.put(grain)
        current_position += grain_interval
        if current_position >= len(data):
            current_position = 0

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
grain_queue = queue.Queue()

# Start the grain production thread
producer_thread = Thread(target=grain_producer, args=(data, grain_size, grain_interval, pitch_shift))
producer_thread.daemon = True
producer_thread.start()

# Start the sounddevice output stream with the callback
stream = sd.OutputStream(callback=audio_callback, samplerate=fs, blocksize=grain_size, device=usb_device_index)

# Start the audio stream and let it run indefinitely
with stream:
    print("Granular synthesis running with effects. Press Ctrl+C to stop.")
    try:
        while True:
            sd.sleep(1000)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Stopping the granular synthesis.")
