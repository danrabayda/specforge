from dataclasses import dataclass
from typing import Iterable, Type

import numpy as np
import torch
from torchaudio.transforms import Spectrogram, MelSpectrogram


@dataclass
class CustomFeatureExtractor:
    sampling_rate: int
    window_samples: int
    transform: torch.nn.Module
    time_dim: int

    @property
    def window_length(self) -> float:
        return self.window_samples / self.sampling_rate

    def __call__(self, x: Iterable[np.ndarray]) -> dict[str, torch.Tensor]:
        padded = []

        for window in x:
            if len(window) < self.window_samples:
                pad_width = self.window_samples - len(window)
                window = np.pad(window, (0, pad_width))
            padded.append(window.astype(np.float32))

        padded = np.stack(padded)

        spec = self.transform(torch.tensor(padded))
        spec = spec.transpose(-2, -1)

        return {"input_values": spec}

    @classmethod
    def from_sample_rate(
        cls,
        sample_rate: int,
        window_length: float,
        spec_freq_dim: int,
        spec_time_dim: int,
        transform_cls: Type = Spectrogram,
    ):
        # Adjust window length to align with time bins
        window_length = (
            int(np.ceil(sample_rate * window_length / spec_time_dim))
            * spec_time_dim
            / sample_rate
        )

        window_samples = int(np.ceil(sample_rate * window_length))

        transform = cls.generate_transform(
            transform_cls,
            sample_rate,
            window_length,
            freq_dim=spec_freq_dim,
            time_dim=spec_time_dim,
        )

        return cls(sample_rate, window_samples, transform, spec_time_dim)

    @staticmethod
    def generate_transform(
        transform_cls,
        sample_rate,
        window_length,
        freq_dim,
        time_dim,
    ):
        hop = sample_rate * window_length / time_dim
        hop_length = int(np.ceil(hop))

        n_fft = freq_dim * 2 - 1
        pad = int(((hop_length - hop) * time_dim - 1) / 2)

        if hop_length > n_fft:
            print("Warning: hop_length > n_fft (may skip data)")

        return transform_cls(
            n_fft=n_fft,
            hop_length=hop_length,
            pad=pad,
            power=2.0,
        )
