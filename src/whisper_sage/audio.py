"""Audio loading for the server, including multichannel microphone arrays.

Example: a ReSpeaker 4-Mic Array with the 6-channel firmware:
    channel 0    processed audio (beamforming + echo cancellation + noise suppression) -> best for ASR
    channels 1-4 raw microphones
    channel 5    playback reference
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


def load(data: bytes, channel: int = 0) -> np.ndarray:
    """Decode an audio file (wav, flac, ogg, …) into mono float32 at 16 kHz.

    Multichannel input keeps only ``channel`` (0 = the ReSpeaker's processed channel); an out-of-range
    channel falls back to 0.
    """
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if not 0 <= channel < audio.shape[1]:
        channel = 0
    audio = audio[:, channel]
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(audio)
