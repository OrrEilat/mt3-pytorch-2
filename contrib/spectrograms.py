# Copyright 2022 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import torch
import torchaudio
from contrib import spectral_ops

# Defaults for spectrogram config
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_HOP_WIDTH = 128
DEFAULT_NUM_MEL_BINS = 512

# Fixed constants; add these to SpectrogramConfig before changing
FFT_SIZE = 2048
MEL_LO_HZ = 20.0


@dataclasses.dataclass
class SpectrogramConfig:
    """Spectrogram configuration parameters."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    hop_width: int = DEFAULT_HOP_WIDTH
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS

    @property
    def abbrev_str(self):
        s = ""
        if self.sample_rate != DEFAULT_SAMPLE_RATE:
            s += f"sr{self.sample_rate}"
        if self.hop_width != DEFAULT_HOP_WIDTH:
            s += f"hw{self.hop_width}"
        if self.num_mel_bins != DEFAULT_NUM_MEL_BINS:
            s += f"mb{self.num_mel_bins}"
        return s

    @property
    def frames_per_second(self):
        return self.sample_rate / self.hop_width


def split_audio(samples, spectrogram_config):
    """Split audio into frames."""
    # Convert samples from NumPy array to PyTorch tensor if it's not already a tensor
    if not isinstance(samples, torch.Tensor):
        samples = torch.from_numpy(samples)

    frame_length = frame_step = spectrogram_config.hop_width
    # Calculate the amount of padding needed to make samples' length a multiple of frame_length
    pad_amount = (frame_length - samples.size(-1) % frame_length) % frame_length
    # Pad the samples tensor
    samples_padded = torch.nn.functional.pad(samples, (0, pad_amount), "constant", 0)

    # Use unfold to create frames from the padded tensor
    frames = samples_padded.unfold(0, frame_length, frame_step)
    return frames


def compute_spectrogram(samples, spectrogram_config):
    """Compute a mel spectrogram using PyTorch."""
    overlap = 1 - (spectrogram_config.hop_width / FFT_SIZE)
    log_mel_spec = spectral_ops.compute_logmel(
        samples,
        lo_hz=MEL_LO_HZ,
        hi_hz=spectrogram_config.sample_rate
        // 2,  # Assuming the high frequency limit is Nyquist
        bins=spectrogram_config.num_mel_bins,
        fft_size=FFT_SIZE,
        overlap=overlap,
        sample_rate=spectrogram_config.sample_rate,
    )
    return log_mel_spec


def flatten_frames(frames):
    print(f"Type of frames: {type(frames)}")
    if isinstance(frames, torch.Tensor):
        print(f"Shape of frames: {frames.shape}")
    return frames.reshape(-1)


def input_depth(spectrogram_config):
    """Input depth is the number of mel bins."""
    return spectrogram_config.num_mel_bins


# Example of usage with PyTorch and torchaudio adapted functions
# Assuming compute_logmel is already adapted for PyTorch as shown in the previous instructions
