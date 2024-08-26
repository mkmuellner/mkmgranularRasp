import numpy as np
import sounddevice as sd
import soundfile as sf
from threading import Thread
import queue

# Import any additional DSP effects or custom functions you had in the original code
# from your_original_effects_module import your_effect_function_1, your_effect_function_2

# Audio settings
filename = 'tori_amos_god_3.wav'
data, fs = sf.read(filename, dtype='float32')  # Load audio file
usb_device_index = 2  # Replace with your actual USB Soundblaster device index

# Define your original effects function placeholders (as you may have already)
def effect_1(grain):
    # Placeholder for an effect function
    # Replace with actual DSP effect
    return grain

def effect_2(grain):
    # Placeholder for another effect function
    # Replace with actual DSP effect
    return grain

# Original Granulation Function, now updated to work with the new audio backend
def generate_grain(data, start_sample, grain_size, pitch_shift=1.0):
    # Extract a grain from the audio sample
    end_sample = min(len(data), start_sample + grain_size)
    grain = data[start_sample:end_sample]

    # Interpolating the grain for pitch shifting
    interpolated_grain = np.interp(
        np.arange(0, len(grain), pitch_shift),
        np.arange(0, len(grain)),
        grain
    )

    # Apply your effects (replace with your actual effect functions)
    grain = effect_1(grain)
    grain = effect_2(grain)

    return grain

# Function to handle producing grains in a separate thread
def grain_producer(data, grain_size, grain_interval, pitch_shift):
    current_position = 0
    while True:
        # Generate the next grain
        grain = generate_grain(data, current_position, grain_size, pitch_shift)

        # Place the grain in the queue for playback
        grain_queue.put(grain)

        # Update the current position
        current_position += grain_interval

        # If we've reached the end of the data, loop back to the beginning
        if current_position >= len(data):
            current_position = 0

# Callback function for real-time audio playback
def audio_callback(outdata, frames, time, status):
    if status:
        print(status)  # Print any errors or warnings
    try:
        # Retrieve the next grain from the queue
        grain = grain_queue.get_nowait()

        # Fill the output buffer with the grain, ensuring it matches the buffer size
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
