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
grain_queue = queue.Queue(maxsize=100)  # Larger queue for better buffering
stop_event = Event()
lock = Lock()

# Function for precise scheduling
def precise_sleep(target_time):
    current_time = time.perf_counter()
    while current_time < target_time:
        current_time = time.perf_counter()

# Grain producer with improved scheduling
def grain_producer(grain_queue, stop_event):
    global grain_size_ms, grain_density, random_offset, random_extent, move_playhead
    global playhead_speed, playhead_direction, mix, grain_pitch, random_pitch_variation
    global envelope_type, apply_pitch, envelope_enabled, apply_random_pitch, apply_random_grain_size
    global apply_random_grain_density, apply_random_position, random_grain_variation, random_grain_density_factor

    current_position = 0
    last_time = time.perf_counter()

    while not stop_event.is_set():
        start_time = time.perf_counter()

        grain_size_samples = ms_to_samples(grain_size_ms, fs)

        with lock:
            if apply_random_grain_density:
                random_density_variation = 1 + (random_grain_density_factor / 100.0) * (np.random.random() - 0.5) * 2
                effective_grain_density = max(min_grain_density, grain_density * random_density_variation)
            else:
                effective_grain_density = grain_density

        interval = 1 / effective_grain_density

        if apply_random_position:
            effective_random_offset = int(random_offset * random_extent)
            start_position = current_position + int(np.random.uniform(-effective_random_offset, effective_random_offset))
        else:
            start_position = current_position

        start_position = np.clip(start_position, 0, len(data) - grain_size_samples)

        grain = generate_grain(
            data, data_reverse, start_position, grain_size_samples, envelope_type=envelope_type, mix=mix,
            pitch=grain_pitch, pitch_variation=random_pitch_variation, apply_pitch=apply_pitch,
            envelope_enabled=envelope_enabled, apply_random_pitch=apply_random_pitch,
            apply_random_grain_size=apply_random_grain_size, random_grain_variation=random_grain_variation
        )

        try:
            grain_queue.put_nowait(grain)
        except queue.Full:
            pass

        if move_playhead:
            current_position += int(fs // effective_grain_density) * playhead_speed * playhead_direction
            current_position %= len(data)

        target_time = last_time + interval
        precise_sleep(target_time)
        last_time = target_time

# Audio Callback Function
def audio_callback(outdata, frames, time, status):
    if status:
        print(status)
    try:
        grain = grain_queue.get_nowait()
        outdata[:frames] = grain[:frames].reshape(-1, 1)
    except queue.Empty:
        outdata.fill(0)

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

        # Continuously update parameters based on keyboard input
        try:
            while True:
                updated_params = handle_keyboard_input(
                    True, grain_queue, move_playhead, playhead_direction, mix, 
                    grain_size_ms, envelope_type, random_extent, random_grain_variation, playhead_speed, 
                    grain_density, random_grain_density_factor, grain_pitch, random_pitch_variation,
                    min_grain_size_ms, max_grain_density, min_grain_density,
                    apply_pitch, envelope_enabled, apply_random_pitch, apply_random_grain_size, 
                    apply_random_grain_density, apply_random_position
                )

                # Unpack and update global variables with the returned values
                if updated_params:
                    (
                        apply_pitch, envelope_enabled, apply_random_pitch, apply_random_grain_size,
                        apply_random_grain_density, apply_random_position
                    ) = updated_params
                
                sd.sleep(100)  # Sleep briefly to allow thread processing
        except KeyboardInterrupt:
            print("Stopping the granular synthesis.")
            stop_event.set()
            producer_thread.join()
