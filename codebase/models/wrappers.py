import torch
import torch.nn as nn
import torchaudio

class SpeakerEmbeddingModel(nn.Module):
    def __init__(self, model: nn.Module, model_name: str):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.sample_rate = 16000

        self._speechbrain_classifier = None
        if model_name.startswith('ecapa'):
            self._init_speechbrain_ecapa()

    def _init_speechbrain_ecapa(self):
        try:
            from speechbrain.inference import EncoderClassifier

            self._speechbrain_classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="./pretrained_models/ecapa_tdnn_512",
                run_opts={"device": "cpu"}
            )
        except Exception as e:
            self._speechbrain_classifier = None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.model_name.startswith('ecapa'):
            return self._forward_ecapa(waveform)
        elif self.model_name == 'resnet34':
            return self._forward_resnet34(waveform)
        elif self.model_name == 'rawnet3':
            return self._forward_rawnet3(waveform)
        elif self.model_name == 'titanet_large':
            return self._forward_titanet(waveform)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _forward_ecapa(self, waveform: torch.Tensor) -> torch.Tensor:
        device = waveform.device

        if self._speechbrain_classifier is not None:
            self._speechbrain_classifier.device = device
            self._speechbrain_classifier.mods.to(device)

            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            with torch.no_grad():
                embeddings = self._speechbrain_classifier.encode_batch(waveform)

            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(1)

            return embeddings
        else:
            return self._forward_ecapa_training(waveform)

    def _forward_ecapa_training(self, waveform: torch.Tensor) -> torch.Tensor:
        device = waveform.device
        original_dim = waveform.dim()
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        fbanks = []
        for i in range(waveform.shape[0]):
            single_waveform = waveform[i:i+1]
            
            fbank = torchaudio.compliance.kaldi.fbank(
                single_waveform.to('cpu'),
                num_mel_bins=80,
                sample_frequency=16000,
                frame_length=25,
                frame_shift=10
            )
            
            fbanks.append(fbank)
        
        fbank = torch.stack(fbanks, dim=0).to(device)
        
        embeddings = self.model(fbank)


        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)

        return embeddings

    def _forward_resnet34(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        embeddings = self.model(waveform)
        return embeddings

    def _forward_rawnet3(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        embeddings = self.model(waveform)
        return embeddings

    def _forward_titanet(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        lengths = torch.tensor([waveform.shape[1]] * waveform.shape[0])
        lengths = lengths.to(waveform.device)

        _, embeddings = self.model(
            input_signal=waveform,
            input_signal_length=lengths
        )

        return embeddings

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(waveform)

    def to(self, device):
        super().to(device)
        if self._speechbrain_classifier is not None:
            self._speechbrain_classifier.device = device
            self._speechbrain_classifier.mods.to(device)
        return self
