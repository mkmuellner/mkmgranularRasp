import numpy as np
import sounddevice as sd
import soundfile as sf
from threading import Thread, Event, Lock
import queue
import time
import os
import sys
from audio_helpers import ms_to_samples, apply_envelope, generate_grain
from input_helpers import print_key_mappings, handle_keyboard_input

# Global variables
grain_size_ms = 200
min_grain_size_ms = 50
grain_density = 20
min_grain_density = 0.5
max_grain_density = 40
random_offset = 500
random_extent = 1.0
move_playhead = False
playhead_speed = 1.0
playhead_direction = 1
mix = 0.5
random_grain_variation = 0
envelope_type = 'soft'
random_grain_density_factor = 0
grain_pitch = 1.0
random_pitch_variation = 0

# Binary variables
apply_pitch = True
envelope_enabled = True
apply_random_pitch = True
apply_random_grain_size = True
apply_random_grain_density = True
apply_random_position = True

# Audio settings
filename = 'tori_amos_god_3.wav'
data, fs = sf.read(filename, dtype='float32')
if len(data.shape) > 1:
    data = np.mean(data, axis=1).astype('float32')
data = data.astype('float16')
data_reverse = data[::-1].astype('float16')

usb_device_index = 2
grain_queue = queue.Queue(maxsize=50)  # Queue for storing pre-buffered grains
stop_event = Event()
lock = Lock()

# Function to generate and pre-buffer grains
def grain_producer(grain_queue, stop_event):
    current_position = 0

    while not stop_event.is_set():
        grain_size_samples = ms_to_samples(grain_size_ms, fs)

        # Generate grain outside of callback
        grain = generate_grain(
            data, data_reverse, current_position, grain_size_samples, envelope_type=envelope_type, mix=mix,
            pitch=grain_pitch, pitch_variation=random_pitch_variation, apply_pitch=apply_pitch,
            envelope_enabled=envelope_enabled, apply_random_pitch=apply_random_pitch,
            apply_random_grain_size=apply_random_grain_size, random_grain_variation=random_grain_variation
        )

        try:
            grain_queue.put(grain, timeout=0.1)  # Timeout to prevent blocking
        except queue.Full:
            pass  # Skip adding this grain if queue is full

        # Move playhead forward or backward if allowed
        if move_playhead:
            current_position += int(fs // grain_density) * playhead_speed * playhead_direction
            current_position %= len(data)  # Loop around if necessary

        # Sleep for the grain interval to match the grain density
        time.sleep(1 / grain_density)

# Audio Callback Function for mixing pre-buffered grains
def audio_callback(outdata, frames, time, status):
    if status:
        print(status)

    # Clear the output buffer (silence)
    outdata.fill(0)

    # Fetch a grain from the queue and mix it into the output buffer
    try:
        grain = grain_queue.get_nowait()

        # Ensure grain is the correct length for the current buffer
        if len(grain) < frames:
            # Pad grain with zeros if it's shorter than the expected frame size
            padded_grain = np.zeros((frames,))
            padded_grain[:len(grain)] = grain
            grain = padded_grain
        elif len(grain) > frames:
            # Trim the grain if it's longer than the expected frame size
            grain = grain[:frames]

        # Mix the grain into the output buffer
        if outdata.shape[1] == 2:  # Stereo
            outdata[:, 0] += grain
            outdata[:, 1] += grain
        else:  # Mono
            outdata[:, 0] += grain

    except queue.Empty:
        pass  # If the queue is empty, play silence

# Pre-fill the queue for smoother playback
def prefill_queue():
    print("Pre-filling grain queue...")
    for _ in range(grain_queue.maxsize):
        grain_size_samples = ms_to_samples(grain_size_ms, fs)
        start_position = np.random.randint(0, len(data) - grain_size_samples)
        grain = generate_grain(
            data, data_reverse, start_position, grain_size_samples, envelope_type=envelope_type, mix=mix,
            pitch=grain_pitch, pitch_variation=random_pitch_variation, apply_pitch=apply_pitch,
            envelope_enabled=envelope_enabled, apply_random_pitch=apply_random_pitch,
            apply_random_grain_size=apply_random_grain_size, random_grain_variation=random_grain_variation
        )
        grain_queue.put_nowait(grain)

# Main program
if __name__ == "__main__":
    prefill_queue()

    # Start grain producer thread
    producer_thread = Thread(target=grain_producer, args=(grain_queue, stop_event))
    producer_thread.daemon = True
    producer_thread.start()

    # Print key mappings at the start of the program
    print_key_mappings()

    # Start audio stream
    stream = sd.OutputStream(callback=audio_callback, samplerate=fs, blocksize=ms_to_samples(grain_size_ms, fs), device=usb_device_index)

    with stream:
        print("Granular synthesis running. Press Ctrl+C to stop.")

        try:
            while True:
                sd.sleep(1000)  # Keep the main thread alive
        except KeyboardInterrupt:
            print("Stopping the granular synthesis.")
            stop_event.set()
            producer_thread.join()
