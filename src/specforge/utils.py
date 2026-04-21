import numpy as np
import torch
import torchaudio


def load_waveform(path, sample_rate):
    waveform, sr = torchaudio.load(path)

    # Convert to mono
    if waveform.shape[0] != 1:
        idx = torch.argmax(torch.mean(waveform, dim=1))
        waveform = waveform[idx:idx+1]

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    return waveform.squeeze().numpy()


def window_generator(data, overlap, window_length, sample_rate, frame_length=0.025):
    samples_per_window = int(sample_rate * window_length)
    data_length = len(data)

    windows = []

    if data_length <= samples_per_window:
        minimum_samples = int(np.ceil(frame_length * sample_rate))
        missing = max(0, minimum_samples - data_length)

        padded = np.pad(data, (0, missing))
        windows.append(padded)

    else:
        offset = int(samples_per_window * (1 - overlap))

        for start in range(0, data_length, offset):
            stop = start + samples_per_window
            window = data[start:stop]

            if len(window) < samples_per_window:
                window = np.pad(window, (0, samples_per_window - len(window)))

            windows.append(window)

    return np.stack(windows)