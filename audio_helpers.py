# audio_helpers.py

import numpy as np

def ms_to_samples(ms, fs):
    return int(ms * fs / 1000)

def apply_envelope(grain, envelope_type):
    length = len(grain)

    if envelope_type == 'soft':
        envelope = np.hanning(length).astype('float16')
    elif envelope_type == 'linear':
        envelope = np.linspace(0, 1, length // 2, dtype='float16')
        envelope = np.concatenate((envelope, envelope[::-1]), dtype='float16')
    elif envelope_type == 'exponential':
        envelope = np.linspace(1, 0.1, length, dtype='float16')
    elif envelope_type == 'gaussian':
        mean, std_dev = length // 2, length // 6
        envelope = np.exp(-0.5 * ((np.arange(length) - mean) ** 2) / (std_dev ** 2)).astype('float16')
    else:
        envelope = np.ones(length, dtype='float16')

    return grain[:len(envelope)] * envelope

def generate_grain(normal_data, reverse_data, start_sample, grain_size_samples, 
                   envelope_type='soft', mix=0.5, pitch=1.0, pitch_variation=0,
                   apply_pitch=True, apply_envelope=True, apply_random_pitch=True, 
                   apply_random_grain_size=True, random_grain_variation=0):
    # Randomly select whether to use normal or reversed buffer based on mix parameter
    data_source = reverse_data if np.random.random() < mix else normal_data

    # Apply random grain size variation if enabled
    if apply_random_grain_size:
        variation_factor = 1 + (random_grain_variation / 100.0) * (np.random.random() - 0.5) * 2
        grain_size_samples = int(grain_size_samples * variation_factor)

    # Generate grain window
    grain = data_source[int(start_sample):int(start_sample) + int(grain_size_samples)]

    # Apply pitch variation if enabled
    if apply_pitch:
        if apply_random_pitch:
            variation_factor = 1 + (pitch_variation / 100.0) * (np.random.random() - 0.5) * 2
            effective_pitch = max(0.1, pitch * variation_factor)
        else:
            effective_pitch = pitch

        if effective_pitch != 1.0 and len(grain) > 1:
            interp_points = np.arange(0, len(grain), effective_pitch)
            grain = np.interp(interp_points, np.arange(0, len(grain)), grain)

    # Apply envelope if enabled
    if apply_envelope:
        grain = apply_envelope(grain, envelope_type)

    return grain


