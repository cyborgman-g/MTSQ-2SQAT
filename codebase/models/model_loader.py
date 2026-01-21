import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path
import sys

class ModelLoader:
    SUPPORTED_MODELS = [
        'ecapa_tdnn_512',
        'ecapa_tdnn_1024',
        'resnet34',
        'rawnet3',
        'titanet_large'
    ]

    EMBEDDING_DIMS = {
        'ecapa_tdnn_512': 192,
        'ecapa_tdnn_1024': 192,
        'resnet34': 256,
        'rawnet3': 256,
        'titanet_large': 192
    }

    def __init__(self, cache_dir: str = './pretrained_models'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._speechbrain_classifier = None

    def load(self, model_name: str, device: str = 'cuda') -> nn.Module:
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {model_name}. "
                           f"Supported: {self.SUPPORTED_MODELS}")

        if model_name == 'ecapa_tdnn_512':
            model = self._load_ecapa_tdnn(channels=512)
        elif model_name == 'ecapa_tdnn_1024':
            model = self._load_ecapa_tdnn(channels=1024)
        elif model_name == 'resnet34':
            model = self._load_resnet34()
        elif model_name == 'rawnet3':
            model = self._load_rawnet3()
        elif model_name == 'titanet_large':
            model = self._load_titanet_large()

        model = model.to(device)
        model.eval()

        return model

    def load_speechbrain_classifier(self, model_name: str = 'ecapa_tdnn_512'):
        try:
            from speechbrain.inference import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier

        self._speechbrain_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(self.cache_dir / f"ecapa_tdnn_512"),
            run_opts={"device": "cpu"}
        )
        return self._speechbrain_classifier

    def _load_ecapa_tdnn(self, channels: int = 512) -> nn.Module:
        try:
            from speechbrain.inference import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(self.cache_dir / f"ecapa_tdnn_{channels}"),
            run_opts={"device": "cpu"}
        )

        self._speechbrain_classifier = classifier

        return classifier.mods.embedding_model

    def _load_resnet34(self) -> nn.Module:
        try:
            import wespeaker
            model = wespeaker.load_model('english')
            return model.model
        except ImportError:
            try:
                from pyannote.audio import Model
                model = Model.from_pretrained(
                    "pyannote/wespeaker-voxceleb-resnet34-LM"
                )
                return model
            except ImportError:
                raise ImportError(
                    "Please install wespeaker or pyannote.audio"
                )

    def _load_rawnet3(self) -> nn.Module:
        rawnet_path = Path('./external/RawNet/models')
        """
        CLone the Rawnet Repository or Change the code to get from wespeaker
        """
        if rawnet_path.exists():
            sys.path.insert(0, str(rawnet_path))
            from RawNet3 import RawNet3

            model = RawNet3(
                block='Bottle2neck',
                model_scale=8,
                context=True,
                summed=True,
                encoder_type='ECA',
                nOut=256,
                out_bn=False,
                sinc_stride=10,
                log_sinc=True,
                norm_sinc='mean',
                grad_mult=1
            )

            checkpoint_path = self.cache_dir / 'rawnet3' / 'model.pt'
            if checkpoint_path.exists():
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(state_dict)

            return model
        else:
            raise FileNotFoundError(f"RawNet repository not found at {rawnet_path}")

    def _load_titanet_large(self) -> nn.Module:
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise ImportError("Please install NeMo: pip install nemo_toolkit[asr]")

        model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name="nvidia/speakerverification_en_titanet_large"
        )

        return model

    def get_embedding_dim(self, model_name: str) -> int:
        return self.EMBEDDING_DIMS[model_name]

    def get_speechbrain_classifier(self):
        return self._speechbrain_classifier
