import os
import torch
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
import torchaudio

from .onnx_runner import ONNXInferenceRunner
from ..utils.metrics import compute_eer, compute_min_dcf

class SpeakerVerificationEvaluator:
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        onnx_runner: Optional[ONNXInferenceRunner] = None,
        device: str = 'cuda'
    ):
        self.model = model
        self.onnx_runner = onnx_runner
        self.device = device

        if model is None and onnx_runner is None:
            raise ValueError("Must provide either model or onnx_runner")

        if model is not None:
            self.model = model.to(device)
            self.model.eval()

    def extract_embedding(self, audio: torch.Tensor) -> np.ndarray:
        if self.onnx_runner is not None:
            audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]
            return self.onnx_runner.infer(audio_np.astype(np.float32))
        else:
            with torch.no_grad():
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                audio = audio.to(self.device)
                embedding = self.model(audio)
                return embedding.cpu().numpy()

    def evaluate_trials(
        self,
        trial_file: str,
        audio_paths: List[str],
        sample_rate: int = 16000,
        max_trials: Optional[int] = None
    ) -> Dict[str, float]:
        trials = self._load_trials_csv(trial_file)

        if max_trials:
            trials = trials[:max_trials]

        labels = []
        scores = []

        embedding_cache = {}
        audio_path_cache = {}

        for label, sample1_path, sample2_path in trials:
            if sample1_path not in embedding_cache:
                full_path = self._find_audio(sample1_path, audio_paths, audio_path_cache)
                waveform = self._load_audio(full_path, sample_rate)
                embedding_cache[sample1_path] = self.extract_embedding(waveform)

            if sample2_path not in embedding_cache:
                full_path = self._find_audio(sample2_path, audio_paths, audio_path_cache)
                waveform = self._load_audio(full_path, sample_rate)
                embedding_cache[sample2_path] = self.extract_embedding(waveform)

            enroll_emb = embedding_cache[sample1_path]
            test_emb = embedding_cache[sample2_path]

            score = self._cosine_similarity(enroll_emb, test_emb)

            labels.append(label)
            scores.append(score)

        eer = compute_eer(labels, scores)
        min_dcf = compute_min_dcf(labels, scores)

        return {
            'eer': eer,
            'min_dcf': min_dcf,
            'num_trials': len(trials),
            'num_target': sum(labels),
            'num_nontarget': len(labels) - sum(labels)
        }

    def _load_trials_csv(self, trial_file: str) -> List[tuple]:
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

    def _find_audio(
        self,
        relative_path: str,
        audio_paths: List[str],
        cache: Dict[str, str]
    ) -> str:
        if relative_path in cache:
            return cache[relative_path]

        for base_path in audio_paths:
            full_path = os.path.join(base_path, relative_path)
            if os.path.exists(full_path):
                cache[relative_path] = full_path
                return full_path

        raise FileNotFoundError(f"Audio not found: {relative_path}")

    def _load_audio(self, path: str, sample_rate: int) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

        return waveform.squeeze(0)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a.flatten()
        b = b.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
