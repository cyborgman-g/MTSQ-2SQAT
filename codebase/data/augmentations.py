import os
import glob
import random
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
from typing import Tuple, List, Optional

class Augmentor:
    def __init__(
        self,
        musan_path: str = "./Datasets/DATASET_MUSAN/musan",
        rirs_path: str = "./Datasets/DATASET_RIRS_NOISES/RIRS_NOISES",
        sample_rate: int = 16000
    ):
        self.sample_rate = sample_rate

        self.noise_files = []
        self.music_files = []
        self.speech_files = []

        if os.path.exists(musan_path):
            noise_dir = os.path.join(musan_path, 'noise')
            music_dir = os.path.join(musan_path, 'music')
            speech_dir = os.path.join(musan_path, 'speech')

            if os.path.exists(noise_dir):
                self.noise_files = glob.glob(os.path.join(noise_dir, '**', '*.wav'), recursive=True)
            if os.path.exists(music_dir):
                self.music_files = glob.glob(os.path.join(music_dir, '**', '*.wav'), recursive=True)
            if os.path.exists(speech_dir):
                self.speech_files = glob.glob(os.path.join(speech_dir, '**', '*.wav'), recursive=True)

        self.rir_files = []
        if os.path.exists(rirs_path):
            self.rir_files = glob.glob(os.path.join(rirs_path, '**', '*.wav'), recursive=True)

    def augment(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.squeeze(0)

        aug_type = random.choice(['noise', 'music', 'speech', 'reverb', 'clean'])

        if aug_type == 'noise' and self.noise_files:
            audio = self._add_noise(audio, self.noise_files, snr_range=(0, 15))
        elif aug_type == 'music' and self.music_files:
            audio = self._add_noise(audio, self.music_files, snr_range=(5, 15))
        elif aug_type == 'speech' and self.speech_files:
            audio = self._add_noise(audio, self.speech_files, snr_range=(13, 20))
        elif aug_type == 'reverb' and self.rir_files:
            audio = self._add_reverb(audio)

        return audio

    def _add_noise(
        self,
        audio: torch.Tensor,
        noise_files: List[str],
        snr_range: Tuple[float, float] = (0, 15)
    ) -> torch.Tensor:
        noise_path = random.choice(noise_files)

        try:
            noise, sr = torchaudio.load(noise_path)
            noise = noise.squeeze(0)

            if sr != self.sample_rate:
                noise = torchaudio.functional.resample(noise, sr, self.sample_rate)

            audio_len = audio.shape[0]
            noise_len = noise.shape[0]

            if noise_len < audio_len:
                repeats = audio_len // noise_len + 1
                noise = noise.repeat(repeats)

            if noise.shape[0] > audio_len:
                start = random.randint(0, noise.shape[0] - audio_len)
                noise = noise[start:start + audio_len]

            snr = random.uniform(*snr_range)

            audio_power = (audio ** 2).mean() + 1e-8
            noise_power = (noise ** 2).mean() + 1e-8
            scale = torch.sqrt(audio_power / (noise_power * (10 ** (snr / 10))))

            return audio + scale * noise

        except Exception as e:
            return audio

    def _add_reverb(self, audio: torch.Tensor) -> torch.Tensor:
        rir_path = random.choice(self.rir_files)

        try:
            rir, sr = torchaudio.load(rir_path)
            rir = rir.squeeze(0)

            if sr != self.sample_rate:
                rir = torchaudio.functional.resample(rir, sr, self.sample_rate)

            rir = rir / (rir.abs().max() + 1e-8)

            peak_idx = rir.abs().argmax()
            rir = rir[peak_idx:]

            max_rir_len = self.sample_rate // 2
            if len(rir) > max_rir_len:
                rir = rir[:max_rir_len]

            audio_expanded = audio.unsqueeze(0).unsqueeze(0)
            rir_kernel = rir.flip(0).unsqueeze(0).unsqueeze(0)

            reverbed = torch.nn.functional.conv1d(
                audio_expanded, rir_kernel, padding=rir.shape[0] - 1
            )

            reverbed = reverbed.squeeze()[:audio.shape[0]]

            reverbed = reverbed * (audio.abs().max() / (reverbed.abs().max() + 1e-8))

            return reverbed

        except Exception as e:
            return audio

class SpecAugment(nn.Module):
    def __init__(self, freq_mask: int = 27, time_mask: int = 100, n_masks: int = 2):
        super().__init__()
        self.freq_masker = torchaudio.transforms.FrequencyMasking(freq_mask)
        self.time_masker = torchaudio.transforms.TimeMasking(time_mask)
        self.n_masks = n_masks

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spectrogram

        for _ in range(self.n_masks):
            spectrogram = self.freq_masker(spectrogram)
            spectrogram = self.time_masker(spectrogram)

        return spectrogram

def get_augmentation_pipeline(
    model_name: str,
    musan_path: str = "./Datasets/DATASET_MUSAN/musan",
    rirs_path: str = "./Datasets/DATASET_RIRS_NOISES/RIRS_NOISES",
    sample_rate: int = 16000
) -> Optional[Augmentor]:
    return Augmentor(musan_path, rirs_path, sample_rate)
