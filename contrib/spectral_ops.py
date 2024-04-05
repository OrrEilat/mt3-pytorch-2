import torch
import torchaudio


def safe_log(x, eps=1e-5):
    """Avoid taking the log of a non-positive number."""
    safe_x = torch.where(x <= 0.0, eps, x)
    return torch.log(safe_x)


def compute_mel(
    audio,
    lo_hz=80.0,
    hi_hz=7600.0,
    bins=64,
    fft_size=2048,
    overlap=0.75,
    pad_end=True,
    sample_rate=16000,
):
    """Calculate Mel Spectrogram in PyTorch."""
    # Assuming `audio` is a 1D or 2D tensor of shape (audio_length,) or (batch_size, audio_length)

    # Compute the hop length as the frame step
    hop_length = int(fft_size * (1.0 - overlap))

    # Compute the STFT
    spectrogram = torch.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=fft_size,
        center=True,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )

    # Convert to magnitude spectrogram
    mag = torch.abs(spectrogram)

    # Create Mel filter
    mel_filter = torchaudio.transforms.MelScale(
        n_mels=bins,
        sample_rate=sample_rate,
        f_min=lo_hz,
        f_max=hi_hz,
        n_stft=fft_size // 2 + 1,
    )

    # Apply Mel filter
    mel_spec = mel_filter(mag.pow(2).sum(-1).sqrt())

    return mel_spec


def compute_logmel(
    audio,
    lo_hz=80.0,
    hi_hz=7600.0,
    bins=64,
    fft_size=2048,
    overlap=0.75,
    pad_end=True,
    sample_rate=16000,
):
    """Logarithmic amplitude of mel-scaled spectrogram in PyTorch."""
    mel = compute_mel(
        audio, lo_hz, hi_hz, bins, fft_size, overlap, pad_end, sample_rate
    )
    logmel = safe_log(mel)
    return logmel


# Example usage
# Assuming `audio` is a PyTorch tensor of audio samples. Shape: (batch_size, audio_length)
# audio = torch.randn(1, 16000)  # Example audio tensor
# log_mel_spec = compute_logmel(audio, sample_rate=16000)
