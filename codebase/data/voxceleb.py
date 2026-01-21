import os
import torch
import torchaudio
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset

from .augmentations import Augmentor

def load_audio(path: str, desired_duration: float = 3.0,
               desired_samplerate: int = 16000) -> Tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != desired_samplerate:
        waveform = torchaudio.functional.resample(waveform, sr, desired_samplerate)

    num_samples = int(desired_duration * desired_samplerate)
    current_samples = waveform.shape[1]

    if current_samples > num_samples:
        start = torch.randint(0, current_samples - num_samples, (1,)).item()
        waveform = waveform[:, start:start + num_samples]
    elif current_samples < num_samples:
        pad_length = num_samples - current_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))

    return waveform.squeeze(0), desired_samplerate

class VoxCelebDataset(Dataset):
    def __init__(
        self,
        audios_path: str = "./Datasets/DATASET_VOXCELEB/2/train",
        duration: float = 3.0,
        sample_rate: int = 16000,
        num_classes: Optional[int] = None,
        augment: bool = False,
        musan_path: str = "./Datasets/DATASET_MUSAN/musan",
        rirs_path: str = "./Datasets/DATASET_RIRS_NOISES/RIRS_NOISES",
        return_fbank: bool = False
    ):
        super().__init__()

        self.audios_path = audios_path
        self.duration = duration
        self.sample_rate = sample_rate
        self.return_fbank = return_fbank

        all_classes = sorted(os.listdir(audios_path))
        if num_classes is not None:
            classes = all_classes[:num_classes]
        else:
            classes = all_classes

        self.audios = []
        for c in classes:
            class_path = os.path.join(audios_path, c)
            if not os.path.isdir(class_path):
                continue
            for yid in os.listdir(class_path):
                yid_path = os.path.join(class_path, yid)
                if not os.path.isdir(yid_path):
                    continue
                for a in os.listdir(yid_path):
                    if a.endswith(('.wav', '.m4a', '.flac')):
                        self.audios.append(os.path.join(audios_path, c, yid, a))

        self.class_dict = {k: v for k, v in zip(classes, range(len(classes)))}
        self.num_speakers = len(classes)
        self._labels_cache = None

        self.augment = augment
        if self.augment:
            self.augmentor = Augmentor(musan_path, rirs_path, sample_rate)
        else:
            self.augmentor = None

    def __len__(self) -> int:
        return len(self.audios)

    @property
    def labels(self) -> List[int]:
        if self._labels_cache is None:
            self._labels_cache = [
                self.class_dict[path.split("/")[-3]]
                for path in self.audios
            ]
        return self._labels_cache

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = self.audios[index]

        audio, sr = load_audio(
            path,
            desired_duration=self.duration,
            desired_samplerate=self.sample_rate
        )

        if self.augment and self.augmentor is not None:
            audio = self.augmentor.augment(audio)

        speaker_id = self.class_dict[path.split("/")[-3]]

        if self.return_fbank:
            fbank = torchaudio.compliance.kaldi.fbank(
                audio.unsqueeze(0),
                num_mel_bins=80,
                sample_frequency=self.sample_rate,
                frame_length=25,
                frame_shift=10
            )
            return fbank, speaker_id
        else:
            return audio, speaker_id

class VoxCelebTrials(Dataset):
    def __init__(
        self,
        trial_file: str,
        audio_paths: List[str],
        sample_rate: int = 16000
    ):
        self.audio_paths = audio_paths
        self.sample_rate = sample_rate

        self.trials = self._load_trials(trial_file)
        self._audio_path_cache: Dict[str, str] = {}

    def _load_trials(self, trial_file: str) -> List[Tuple[int, str, str]]:
        trials = []

        with open(trial_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('match'):
                    continue

                if ',' in line:
                    parts = [p.strip() for p in line.split(',')]
                else:
                    parts = line.split()

                if len(parts) >= 3:
                    label = int(parts[0])
                    sample1 = parts[1]
                    sample2 = parts[2]
                    trials.append((label, sample1, sample2))

        return trials

    def _find_audio_file(self, relative_path: str) -> str:
        if relative_path in self._audio_path_cache:
            return self._audio_path_cache[relative_path]

        for base_path in self.audio_paths:
            full_path = os.path.join(base_path, relative_path)
            if os.path.exists(full_path):
                self._audio_path_cache[relative_path] = full_path
                return full_path

        raise FileNotFoundError(f"Audio file not found: {relative_path}")

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        label, sample1_path, sample2_path = self.trials[idx]

        sample1_full = self._find_audio_file(sample1_path)
        sample2_full = self._find_audio_file(sample2_path)

        waveform1 = self._load_audio(sample1_full)
        waveform2 = self._load_audio(sample2_full)

        return label, waveform1, waveform2

    def _load_audio(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        return waveform.squeeze(0)

    def get_labels_and_paths(self) -> List[Tuple[int, str, str]]:
        return self.trials

DEFAULT_TRIAL_PATHS = {
    "o": "./Datasets/DATASET_VOXCELEB/trial_pairs/voxceleb_O_cleaned.csv",
    "e": "./Datasets/DATASET_VOXCELEB/trial_pairs/voxceleb_E_cleaned.csv",
    "h": "./Datasets/DATASET_VOXCELEB/trial_pairs/voxceleb_H_cleaned.csv"
}

DEFAULT_AUDIO_PATHS = [
    "./Datasets/DATASET_VOXCELEB/1/test",
    "./Datasets/DATASET_VOXCELEB/2/test",
    "./Datasets/DATASET_VOXCELEB/1/train"
]

DEFAULT_TRAIN_PATH = "./Datasets/DATASET_VOXCELEB/2/train"
DEFAULT_MUSAN_PATH = "./Datasets/DATASET_MUSAN/musan"
DEFAULT_RIRS_PATH = "./Datasets/DATASET_RIRS_NOISES/RIRS_NOISES"
