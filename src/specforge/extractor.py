from dataclasses import dataclass
from typing import Iterable, Type, Union

import torch
import torch.nn.functional as F
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

    def __call__(
        self,
        x: Union[Iterable[torch.Tensor], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Accepts:
        - list of 1D tensors [T]
        - OR tensor [batch, T]
        """

        if isinstance(x, torch.Tensor):
            if x.ndim == 1:
                x = x.unsqueeze(0)  # [T] → [1, T]
            windows = x
        else:
            windows = []
            for w in x:
                if not isinstance(w, torch.Tensor):
                    w = torch.tensor(w, dtype=torch.float32)

                if w.ndim > 1:
                    raise ValueError("Each window must be 1D")

                windows.append(w)

            windows = torch.nn.utils.rnn.pad_sequence(
                windows, batch_first=True
            )

        # Ensure correct length
        if windows.shape[1] < self.window_samples:
            pad_amount = self.window_samples - windows.shape[1]
            windows = F.pad(windows, (0, pad_amount))

        elif windows.shape[1] > self.window_samples:
            windows = windows[:, :self.window_samples]

        windows = windows.float()

        spec = self.transform(windows)  # [B, F, T]
        spec = spec.transpose(-2, -1)   # → [B, T, F]

        return {"input_values": spec}

    def to(self, device):
        self.transform = self.transform.to(device)
        return self

    @classmethod
    def from_sample_rate(
        cls,
        sample_rate: int,
        window_length: float,
        spec_freq_dim: int,
        spec_time_dim: int,
        transform_cls: Type = Spectrogram,
    ):
        window_length = (
            int(torch.ceil(torch.tensor(sample_rate * window_length / spec_time_dim)))
            * spec_time_dim
            / sample_rate
        )

        window_samples = int(torch.ceil(torch.tensor(sample_rate * window_length)))

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
        hop_length = int(torch.ceil(torch.tensor(hop)))

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