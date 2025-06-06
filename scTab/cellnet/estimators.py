from os.path import join
from typing import Dict, List

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.tuner.tuning import Tuner

from cellnet.datamodules import MerlinDataModule
from cellnet.models import LinearClassifier, MLPClassifier, TabnetClassifier


class EstimatorCellTypeClassifier:
    datamodule: MerlinDataModule
    model: pl.LightningModule
    trainer: pl.Trainer

    def __init__(self, data_path: str, class_weights_filename: str = "class_weights.npy"):
        self.data_path = data_path
        self.class_weights_filename = class_weights_filename

    def init_datamodule(
        self,
        batch_size: int = 2048,
        sub_sample_frac: float = 1.0,
        dataloader_kwargs_train: Dict = None,
        dataloader_kwargs_inference: Dict = None,
        merlin_dataset_kwargs_train: Dict = None,
        merlin_dataset_kwargs_inference: Dict = None,
    ):
        self.datamodule = MerlinDataModule(
            self.data_path,
            columns=["cell_type", "soma_joinid"],  # Always include cell_type
            batch_size=batch_size,
            sub_sample_frac=sub_sample_frac,
            dataloader_kwargs_train=dataloader_kwargs_train,
            dataloader_kwargs_inference=dataloader_kwargs_inference,
            dataset_kwargs_train=merlin_dataset_kwargs_train,
            dataset_kwargs_inference=merlin_dataset_kwargs_inference,
        )

    def init_model(self, model_type: str, model_kwargs):
        if model_type == "tabnet":
            self.model = TabnetClassifier(
                **{**self.get_fixed_model_params(model_type), **model_kwargs}
            )
        elif model_type == "linear":
            self.model = LinearClassifier(
                **{**self.get_fixed_model_params(model_type), **model_kwargs}
            )
        elif model_type == "mlp":
            self.model = MLPClassifier(
                **{**self.get_fixed_model_params(model_type), **model_kwargs}
            )
        else:
            raise ValueError(
                f'model_type has to be in ["linear", "mlp", "tabnet"]. '
                f'You supplied: {model_type}'
            )

    def init_trainer(self, trainer_kwargs):
        self.trainer = pl.Trainer(**trainer_kwargs)

    def _check_is_initialized(self):
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("You need to call self.init_model before calling self.train")
        if not hasattr(self, 'datamodule') or self.datamodule is None:
            raise RuntimeError("You need to call self.init_datamodule before calling self.train")
        if not hasattr(self, 'trainer') or self.trainer is None:
            raise RuntimeError("You need to call self.init_trainer before calling self.train")

    def get_fixed_model_params(self, model_type: str):
        model_params = {
            "gene_dim": len(pd.read_parquet(join(self.data_path, "var.parquet"))),
            "train_set_size": sum(self.datamodule.train_dataset.partition_lens),
            "val_set_size": sum(self.datamodule.val_dataset.partition_lens),
            "batch_size": self.datamodule.batch_size,
        }
        
        # Add augmentations for models that need them
        if model_type in ["tabnet", "mlp"]:
            model_params["augmentations"] = np.load(join(self.data_path, "augmentations.npy"))
        
        # Add classification-specific parameters
        if model_type in ["tabnet", "linear", "mlp"]:
            model_params.update({
                "type_dim": len(
                    pd.read_parquet(join(self.data_path, "categorical_lookup/cell_type.parquet"))
                ),
                "class_weights": np.load(join(self.data_path, self.class_weights_filename)),
                "child_matrix": np.load(join(self.data_path, "cell_type_hierarchy/child_matrix.npy")),
            })
            
        return model_params

    def find_lr(self, lr_find_kwargs, plot_results: bool = False):
        self._check_is_initialized()
        tuner = Tuner(self.trainer)
        lr_finder = tuner.lr_find(
            self.model,
            train_dataloaders=self.datamodule.train_dataloader(),
            val_dataloaders=self.datamodule.val_dataloader(),
            **lr_find_kwargs,
        )
        if plot_results:
            lr_finder.plot(suggest=True)

        return lr_finder.suggestion(), lr_finder.results

    def train(self, ckpt_path: str = None):
        self._check_is_initialized()
        self.trainer.fit(
            self.model,
            datamodule=self.datamodule,
            ckpt_path=ckpt_path,
        )

    def validate(self, ckpt_path: str = None):
        self._check_is_initialized()
        return self.trainer.validate(
            self.model, dataloaders=self.datamodule.val_dataloader(), ckpt_path=ckpt_path
        )

    def test(self, ckpt_path: str = None):
        self._check_is_initialized()
        return self.trainer.test(
            self.model, dataloaders=self.datamodule.test_dataloader(), ckpt_path=ckpt_path
        )

    def predict(self, dataloader=None, ckpt_path: str = None) -> np.ndarray:
        self._check_is_initialized()
        predictions_batched: List[torch.Tensor] = self.trainer.predict(
            self.model,
            dataloaders=dataloader if dataloader else self.datamodule.predict_dataloader(),
            ckpt_path=ckpt_path,
        )
        return torch.vstack(predictions_batched).numpy()
