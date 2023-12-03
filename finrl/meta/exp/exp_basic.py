from __future__ import annotations

import os

import torch

from ..models import Autoformer
from ..models import DLinear
from ..models import ETSformer
from ..models import FEDformer
from ..models import Informer
from ..models import LightTS
from ..models import Nonstationary_Transformer
from ..models import PatchTST
from ..models import Pyraformer
from ..models import Reformer
from ..models import TimesNet
from ..models import Transformer


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "TimesNet": TimesNet,
            "Autoformer": Autoformer,
            "Transformer": Transformer,
            "Nonstationary_Transformer": Nonstationary_Transformer,
            "DLinear": DLinear,
            "FEDformer": FEDformer,
            "Informer": Informer,
            "LightTS": LightTS,
            "Reformer": Reformer,
            "ETSformer": ETSformer,
            "PatchTST": PatchTST,
            "Pyraformer": Pyraformer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
