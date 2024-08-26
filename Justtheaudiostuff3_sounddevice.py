import numpy as np
import sounddevice as sd
import soundfile as sf
from threading import Thread, Event
import queue
import time
import sys
import select
import os

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

# Global variable to control keyboard input
keyboard_input_enabled = False  # Set to False by default

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

# Function to convert grain size in ms to samples
def ms_to_samples(ms):
    return int(ms * fs / 1000)

# Envelope Types with reduced precision
def apply_envelope(grain, envelope_type):
    length = len(grain)

    # Use smoother Hanning window by default, reduced precision
    if envelope_type == 'soft':
        envelope = np.hanning(length).astype('float16')
    elif envelope_type == 'linear':
        envelope = np.linspace(0, 1, length // 2, dtype='float16')
        envelope = np.concatenate((envelope, envelope[::-1]), dtype='float16')  # Symmetric fade in/out
    elif envelope_type == 'exponential':
        envelope = np.linspace(1, 0.1, length, dtype='float16')
    elif envelope_type == 'gaussian':
        mean, std_dev = length // 2, length // 6
        envelope = np.exp(-0.5 * ((np.arange(length) - mean) ** 2) / (std_dev ** 2)).astype('float16')
    else:
        envelope = np.ones(length, dtype='float16')  # No envelope (not recommended)

    return grain[:len(envelope)] * envelope  # Ensure the lengths match

# Optimized grain generation
def generate_grain(normal_data, reverse_data, start_sample, grain_size_samples, envelope_type='soft', mix=0.5, pitch=1.0, pitch_variation=0):
    # Randomly select whether to use normal or reversed buffer based on mix parameter
    data_source = reverse_data if np.random.random() < mix else normal_data

    # Apply pitch variation
    variation_factor = 1 + (pitch_variation / 100.0) * (np.random.random() - 0.5) * 2
    effective_pitch = max(0.1, pitch * variation_factor)

    # Generate grain window with interpolation if pitch is varied
    grain = data_source[start_sample:start_sample + grain_size_samples]
    if effective_pitch != 1.0 and len(grain) > 1:
        interp_points = np.arange(0, len(grain), effective_pitch)
        grain = np.interp(interp_points, np.arange(0, len(grain)), grain)

    grain = apply_envelope(grain, envelope_type)
    return grain

# Function to handle producing grains in a separate thread
def grain_producer(grain_queue, stop_event):
    global grain_size_ms, grain_density, random_offset, random_extent, move_playhead, playhead_speed, playhead_direction, mix, random_grain_variation, envelope_type, random_grain_density_factor, grain_pitch, random_pitch_variation

    current_position = 0

    while not stop_event.is_set():
        # Apply random variation to grain size
        size_variation_factor = 1 + (random_grain_variation / 100.0) * (np.random.random() - 0.5) * 2
        grain_size_samples = ms_to_samples(grain_size_ms * size_variation_factor)

        # Apply random variation to grain density
        random_density_variation = 1 + (random_grain_density_factor / 100.0) * (np.random.random() - 0.5) * 2
        effective_grain_density = max(min_grain_density, grain_density * random_density_variation)

        # Apply the extent of randomness to the random offset
        effective_random_offset = int(random_offset * random_extent)

        # Adjust start position using numpy's random
        start_position = current_position + int(np.random.uniform(-effective_random_offset, effective_random_offset))
        start_position = np.clip(start_position, 0, len(data) - grain_size_samples)

        # Generate grain at randomized start position
        grain = generate_grain(data, data_reverse, start_position, grain_size_samples, envelope_type=envelope_type, mix=mix, pitch=grain_pitch, pitch_variation=random_pitch_variation)

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
stream = sd.OutputStream(callback=audio_callback, samplerate=fs, blocksize=ms_to_samples(grain_size_ms), device=usb_device_index)

# Pre-fill the grain queue to ensure smooth playback
print("Pre-filling grain queue...")
while not grain_queue.full():  # Only fill if there's space in the queue
    try:
        start_position = np.random.randint(0, len(data) - ms_to_samples(grain_size_ms))
        grain = generate_grain(data, data_reverse, start_position, ms_to_samples(grain_size_ms), envelope_type=envelope_type, mix=mix, pitch=grain_pitch, pitch_variation=random_pitch_variation)
        grain_queue.put_nowait(grain)
    except queue.Full:
        # If the queue is full, stop pre-filling
        print("Grain queue is full, stopping pre-fill.")
        break

# Non-blocking input method using select
def input_with_timeout(prompt, timeout=0.1):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.readline().strip()
    return None

# Print key mappings at startup
def print_key_mappings():
    print("Key mappings:")
    print("f: Move playhead forward")
    print("b: Move playhead backward")
    print("k: Toggle continuous playhead movement")
    print("u: Increase randomness around playhead")
    print("i: Decrease randomness around playhead")
    print("m: Increase mix towards reversed grains")
    print("n: Increase mix towards normal grains")
    print("+: Increase grain size by 50 ms")
    print("-: Decrease grain size by 50 ms")
    print("e: Change envelope type")
    print("v: Increase random variation around grain size by 50%")
    print("c: Decrease random variation around grain size by 50%")
    print("p: Increase playhead speed by 10%")
    print("l: Decrease playhead speed by 10%")
    print("g: Double grain density")
    print("h: Halve grain density")
    print("r: Increase random grain density by 50%")
    print("t: Decrease random grain density by 50%")
    print("z: Increase pitch by 10%")
    print("x: Decrease pitch by 10%")
    print("q: Increase random pitch variation by 50%")
    print("w: Decrease random pitch variation by 50%")

# Main loop for keyboard input handling
def handle_keyboard_input():
    global move_playhead, playhead_direction, mix, grain_size_ms, envelope_type, random_extent, random_grain_variation, playhead_speed, grain_density, random_grain_density_factor, grain_pitch, random_pitch_variation

    envelope_options = ['linear', 'exponential', 'soft', 'gaussian']
    current_envelope_index = envelope_options.index(envelope_type)

    if not keyboard_input_enabled:
        return  # Skip keyboard handling if disabled

    while True:
        key = input_with_timeout('', timeout=0.1)  # Wait for input
        if key == 'f':  # Move playhead forward once
            move_playhead = False
            playhead_direction = 1
        elif key == 'b':  # Move playhead backward once
            move_playhead = False
            playhead_direction = -1
        elif key == 'k':  # Keep moving playhead in current direction
            move_playhead = not move_playhead
        elif key == 'u':  # Increase randomness around playhead
            random_extent += 0.1  # Increase randomness
        elif key == 'i':  # Decrease randomness around playhead
            random_extent = max(0, random_extent - 0.1)  # Decrease randomness
        elif key == 'm':  # Increase mix towards reversed grains
            mix = min(1.0, mix + 0.1)
            print(f"Mix: {mix}")
        elif key == 'n':  # Increase mix towards normal grains
            mix = max(0.0, mix - 0.1)
            print(f"Mix: {mix}")
        elif key == '+':  # Increase grain size by 50 ms
            grain_size_ms = min(grain_size_ms + 50, 10000)  # Cap at 10 seconds
            print(f"Grain Size: {grain_size_ms} ms")
        elif key == '-':  # Decrease grain size by 50 ms
            grain_size_ms = max(grain_size_ms - 50, min_grain_size_ms)
            print(f"Grain Size: {grain_size_ms} ms")
        elif key == 'e':  # Change envelope type
            current_envelope_index = (current_envelope_index + 1) % len(envelope_options)
            envelope_type = envelope_options[current_envelope_index]
            print(f"Envelope: {envelope_type}")
        elif key == 'v':  # Increase random variation around grain size by 50%
            random_grain_variation += 50
            print(f"Random Grain Variation: {random_grain_variation}%")
        elif key == 'c':  # Decrease random variation around grain size by 50%
            random_grain_variation = max(0, random_grain_variation - 50)
            print(f"Random Grain Variation: {random_grain_variation}%")
        elif key == 'p':  # Increase playhead speed by 10%
            playhead_speed += 0.1
            print(f"Playhead Speed: {playhead_speed}")
        elif key == 'l':  # Decrease playhead speed by 10%
            playhead_speed = max(0.1, playhead_speed - 0.1)  # Minimum playhead speed is 0.1
            print(f"Playhead Speed: {playhead_speed}")
        elif key == 'g':  # Double grain density
            grain_density = min(max_grain_density, grain_density * 2)
            print(f"Grain Density: {grain_density} grains/second")
        elif key == 'h':  # Halve grain density
            grain_density = max(min_grain_density, grain_density / 2)
            print(f"Grain Density: {grain_density} grains/second")
        elif key == 'r':  # Increase random variation around grain density by 50%
            random_grain_density_factor += 50
            print(f"Random Grain Density Factor: {random_grain_density_factor}%")
        elif key == 't':  # Decrease random variation around grain density by 50%
            random_grain_density_factor = max(0, random_grain_density_factor - 50)
            print(f"Random Grain Density Factor: {random_grain_density_factor}%")
        elif key == 'z':  # Increase pitch by 10%
            grain_pitch += 0.1
            print(f"Grain Pitch: {grain_pitch}")
        elif key == 'x':  # Decrease pitch by 10%
            grain_pitch = max(0.1, grain_pitch - 0.1)  # Minimum pitch is 0.1
            print(f"Grain Pitch: {grain_pitch}")
        elif key == 'q':  # Increase random pitch variation by 50%
            random_pitch_variation += 50
            print(f"Random Pitch Variation: {random_pitch_variation}%")
        elif key == 'w':  # Decrease random pitch variation by 50%
            random_pitch_variation = max(0, random_pitch_variation - 50)
            print(f"Random Pitch Variation: {random_pitch_variation}%")

        # Clear the grain queue after significant changes
        if key in {'+', '-', 'g', 'h', 'r', 't', 'z', 'x', 'q', 'w'}:
            with grain_queue.mutex:
                grain_queue.queue.clear()

# Start a thread for keyboard handling, only if keyboard input is enabled
if keyboard_input_enabled:
    print_key_mappings()
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
