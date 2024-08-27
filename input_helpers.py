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

    # New toggles for boolean controls
    print("y: Toggle pitch application on/off")
    print("t: Toggle envelope application on/off")
    print("r: Toggle random pitch variation on/off")
    print("g: Toggle random grain size variation on/off")
    print("d: Toggle random grain density variation on/off")
    print("z: Toggle random position variation on/off")

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
    min_grain_size_ms, max_grain_density, min_grain_density,
    apply_pitch, envelope_enabled, apply_random_pitch, apply_random_grain_size, 
    apply_random_grain_density, apply_random_position
):
    envelope_options = ['linear', 'exponential', 'soft', 'gaussian']
    current_envelope_index = envelope_options.index(envelope_type)

    if not keyboard_input_enabled:
        return  # Skip keyboard handling if disabled

    while True:
        key = input("Press a key: ")  # Simple blocking input for testing

        if key == 'f':  # Move playhead forward once
            move_playhead = False
            playhead_direction = 1
            print(f"Move playhead forward. Direction: {playhead_direction}")
        elif key == 'b':  # Move playhead backward once
            move_playhead = False
            playhead_direction = -1
            print(f"Move playhead backward. Direction: {playhead_direction}")
        elif key == 'k':  # Keep moving playhead in current direction
            move_playhead = not move_playhead
            print(f"Continuous playhead movement: {move_playhead}")
        elif key == 'u':  # Increase randomness around playhead
            random_extent += 0.1
            print(f"Increased randomness around playhead. New extent: {random_extent}")
        elif key == 'i':  # Decrease randomness around playhead
            random_extent = max(0, random_extent - 0.1)
            print(f"Decreased randomness around playhead. New extent: {random_extent}")
        elif key == 'y':  # Toggle pitch application
            apply_pitch = not apply_pitch
            print(f"Pitch Application: {apply_pitch}")
        elif key == 't':  # Toggle envelope application
            envelope_enabled = not envelope_enabled
            print(f"Envelope Application: {envelope_enabled}")
        elif key == 'r':  # Toggle random pitch variation
            apply_random_pitch = not apply_random_pitch
            print(f"Random Pitch Variation: {apply_random_pitch}")
        elif key == 'g':  # Toggle random grain size variation
            apply_random_grain_size = not apply_random_grain_size
            print(f"Random Grain Size Variation: {apply_random_grain_size}")
        elif key == 'd':  # Toggle random grain density variation
            apply_random_grain_density = not apply_random_grain_density
            print(f"Random Grain Density Variation: {apply_random_grain_density}")
        elif key == 'z':  # Toggle random position variation
            apply_random_position = not apply_random_position
            print(f"Random Position Variation: {apply_random_position}")
        elif key == '+':  # Increase grain size
            grain_size_ms = min(grain_size_ms + 50, 2000)  # Arbitrary max grain size
            print(f"Increased grain size to {grain_size_ms} ms")
        elif key == '-':  # Decrease grain size
            grain_size_ms = max(grain_size_ms - 50, min_grain_size_ms)
            print(f"Decreased grain size to {grain_size_ms} ms")
        elif key == 'p':  # Increase playhead speed
            playhead_speed *= 1.1
            print(f"Increased playhead speed to {playhead_speed}")
        elif key == 'l':  # Decrease playhead speed
            playhead_speed *= 0.9
            print(f"Decreased playhead speed to {playhead_speed}")
        elif key == 'm':  # Increase mix towards reversed grains
            mix = min(mix + 0.1, 1.0)
            print(f"Increased mix towards reversed grains: {mix}")
        elif key == 'n':  # Increase mix towards normal grains
            mix = max(mix - 0.1, 0.0)
            print(f"Increased mix towards normal grains: {mix}")
        elif key == 'g':  # Double grain density
            grain_density = min(grain_density * 2, max_grain_density)
            print(f"Doubled grain density: {grain_density}")
        elif key == 'h':  # Halve grain density
            grain_density = max(grain_density / 2, min_grain_density)
            print(f"Halved grain density: {grain_density}")
        elif key == 'r':  # Increase random grain density by 50%
            random_grain_density_factor += 0.5
            print(f"Increased random grain density factor to {random_grain_density_factor}")
        elif key == 't':  # Decrease random grain density by 50%
            random_grain_density_factor = max(0, random_grain_density_factor - 0.5)
            print(f"Decreased random grain density factor to {random_grain_density_factor}")
        elif key == 'z':  # Increase pitch by 10%
            grain_pitch *= 1.1
            print(f"Increased pitch to {grain_pitch}")
        elif key == 'x':  # Decrease pitch by 10%
            grain_pitch *= 0.9
            print(f"Decreased pitch to {grain_pitch}")

        # Return the updated values
        return (
            move_playhead, playhead_direction, random_extent, grain_size_ms, playhead_speed, 
            mix, grain_density, random_grain_density_factor, grain_pitch, apply_pitch,
            envelope_enabled, apply_random_pitch, apply_random_grain_size, apply_random_grain_density, 
            apply_random_position
        )
