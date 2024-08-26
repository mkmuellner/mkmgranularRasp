# input_helpers.py

import sys
import select

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

# Non-blocking input method using select
def input_with_timeout(prompt, timeout=0.1):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.readline().strip()
    return None

def handle_keyboard_input(
    keyboard_input_enabled, grain_queue, move_playhead, playhead_direction, mix, 
    grain_size_ms, envelope_type, random_extent, random_grain_variation, playhead_speed, 
    grain_density, random_grain_density_factor, grain_pitch, random_pitch_variation,
    min_grain_size_ms, max_grain_density, min_grain_density
):
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
