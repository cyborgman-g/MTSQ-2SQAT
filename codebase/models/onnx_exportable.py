import torch
import torch.nn as nn

class SpeechBrainFeatureExtractor(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.compute_features = classifier.mods.compute_features
        self.mean_var_norm = classifier.mods.mean_var_norm

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.compute_features(waveform)

            wav_lens = torch.ones(waveform.shape[0], device=waveform.device)
            features = self.mean_var_norm(features, wav_lens)

        return features

class SpeechBrainEmbeddingWrapper(nn.Module):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_model(fbank)

        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)

        return embeddings

class FbankExtractor:
    def __init__(self, classifier=None):
        if classifier is None:
            try:
                from speechbrain.inference import EncoderClassifier
            except ImportError:
                from speechbrain.pretrained import EncoderClassifier

            classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="./pretrained_models/ecapa_tdnn_512",
                run_opts={"device": "cpu"}
            )

        self.classifier = classifier
        self.compute_features = classifier.mods.compute_features
        self.mean_var_norm = classifier.mods.mean_var_norm

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.compute_features(waveform)
            wav_lens = torch.ones(waveform.shape[0], device=waveform.device)
            features = self.mean_var_norm(features, wav_lens)
        return features

    def to(self, device):
        self.compute_features = self.compute_features.to(device)
        self.mean_var_norm = self.mean_var_norm.to(device)
        return self

def create_onnx_exportable_ecapa(device: str = 'cpu'):
    try:
        from speechbrain.inference import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="./pretrained_models/ecapa_tdnn_512",
        run_opts={"device": device}
    )

    embedding_wrapper = SpeechBrainEmbeddingWrapper(classifier.mods.embedding_model)
    feature_extractor = FbankExtractor(classifier)

    return embedding_wrapper, feature_extractor
