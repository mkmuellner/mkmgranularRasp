import numpy as np
import sounddevice as sd
import soundfile as sf
from threading import Thread, Event, Lock
import queue
import time
import random
from audio_helpers import ms_to_samples, apply_envelope, generate_grain
from input_helpers import print_key_mappings, handle_keyboard_input

# Global variables
empty_queue_count = 0  # Counts how many times the grain_queue was empty
grain_size_ms = 200
grain_density = 20  # Number of grains per second
random_offset = 500
random_extent = 1.0
move_playhead = False
playhead_speed = 1.0
playhead_direction = 1
mix = 0.5
random_grain_variation = 0
envelope_type = 'soft'
grain_pitch = 1.0
random_pitch_variation = 0
recycle_fraction = 0.5  # Fraction of grains that can be recycled
grains_queue = 30

# Binary variables
apply_pitch = True
envelope_enabled = True
apply_random_pitch = True
apply_random_grain_size = True
apply_random_position = True
keyboard_input = True

# Audio settings
filename = 'tori_amos_god_3.wav'
data, fs = sf.read(filename, dtype='float32')

# If the audio has multiple channels, convert it to mono
if len(data.shape) > 1:
    data = np.mean(data, axis=1).astype('float32')

# Audio settings
filename = 'tori_amos_god_3.wav'
data, fs = sf.read(filename, dtype='float32')
if len(data.shape) > 1:
    data = np.mean(data, axis=1).astype('float32')
data = data.astype('float16')
data_reverse = data[::-1].astype('float16')


usb_device_index = 2
grain_queue = queue.Queue(maxsize=grains_queue)  # Queue for storing batches of mixed grains
stop_event = Event()
lock = Lock()

def grain_producer(grain_queue, stop_event):
    current_position = 0
    previous_grains = []  # List to keep track of previous grains for recycling
    grain_interval = 1.0 / (grain_density * 2)  # Shorter interval for more frequent grain production

    while not stop_event.is_set():
        # Generate a new grain
        grain_size_samples = ms_to_samples(grain_size_ms, fs)
        if random.random() < recycle_fraction and previous_grains:
            grain = random.choice(previous_grains)
        else:
            start_position = np.random.randint(0, len(data) - grain_size_samples)
            grain = generate_grain(
                data, data_reverse, start_position, grain_size_samples, envelope_type=envelope_type, mix=mix,
                pitch=grain_pitch, pitch_variation=random_pitch_variation, apply_pitch=apply_pitch,
                envelope_enabled=envelope_enabled, apply_random_pitch=apply_random_pitch,
                apply_random_grain_size=apply_random_grain_size, random_grain_variation=random_grain_variation
            )

            previous_grains.append(grain)
            if len(previous_grains) > grains_queue:
                previous_grains.pop(0)

        try:
            grain_queue.put(grain, timeout=0.1)
        except queue.Full:
            pass

        time.sleep(grain_interval)  # Reduced sleep interval


def audio_callback(outdata, frames, time, status):
    global empty_queue_count

    if status:
        print(status)

    # Clear the output buffer (silence)
    outdata.fill(0)

    # Try to mix multiple grains into the output buffer
    try:
        # Fetch and mix multiple grains for overlap
        for _ in range(3):  # Adjust this number based on desired overlap
            grain = grain_queue.get_nowait()

            # Ensure the grain is the correct length for the current buffer
            if len(grain) < frames:
                padded_grain = np.zeros((frames,))
                padded_grain[:len(grain)] = grain
                grain = padded_grain
            elif len(grain) > frames:
                grain = grain[:frames]

            # Apply the mixed grain batch to the output buffer
            if outdata.shape[1] == 2:  # Stereo
                outdata[:, 0] += grain
                outdata[:, 1] += grain
            else:  # Mono
                outdata[:, 0] += grain

    except queue.Empty:
        empty_queue_count += 1


# Keyboard input thread
def keyboard_input_thread():
    global grain_size_ms, grain_density, playhead_speed, apply_pitch, envelope_enabled, mix
    global move_playhead, playhead_direction, random_extent, random_grain_variation
    global grain_pitch, random_pitch_variation, apply_random_pitch, apply_random_grain_size
    global apply_random_grain_density, apply_random_position

    while keyboard_input:
        # Handle keyboard input and update the parameters
        updated_params = handle_keyboard_input(
            True, grain_queue, move_playhead, playhead_direction, mix,
            grain_size_ms, envelope_type, random_extent, random_grain_variation, playhead_speed,
            grain_density, grain_pitch, random_pitch_variation,
            min_grain_size_ms, max_grain_density, min_grain_density,
            apply_pitch, envelope_enabled, apply_random_pitch, apply_random_grain_size,
            apply_random_grain_density, apply_random_position
        )

        # Unpack and update global variables with the returned values
        if updated_params:
            (
                move_playhead, playhead_direction, random_extent, grain_size_ms, playhead_speed,
                mix, grain_density, grain_pitch, apply_pitch,
                envelope_enabled, apply_random_pitch, apply_random_grain_size, apply_random_grain_density,
                apply_random_position
            ) = updated_params

        time.sleep(0.1)  # Short sleep to avoid high CPU usage

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

    # Start keyboard input thread
    keyboard_thread = Thread(target=keyboard_input_thread)
    keyboard_thread.daemon = True
    keyboard_thread.start()

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
            print(f"\nGrain queue was empty {empty_queue_count} times during execution.")
            print("Stopping the granular synthesis.")
            stop_event.set()
            producer_thread.join()
