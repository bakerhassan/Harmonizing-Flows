###The basline of the codes are adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html

from src.globals import globals

## Standard libraries
import os
import time
from torch.utils.data import DataLoader

## Progress bar

import torch
# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import Callback

# DataLoader
from src.step2_NF_model.flow_guided_dataloader import MedicalImage2DDataset

# NF model
from src.step2_NF_model.NF_model import flow_model


device = globals.device
print("Using device", device)

for site in ['CALTECH']:
    root_dir = '../../data/'
    CHECKPOINT_PATH = f'../checkpoints/ABIDE-FLOW-{site}'
    train_set = MedicalImage2DDataset('train', globals.affine_file, globals.training_data_location)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    val_set = MedicalImage2DDataset('val', globals.affine_file, globals.validation_data_location)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)


    class PrintCallback(Callback):
        def on_epoch_end(self, trainer, pl_module):
            print("")


    def train_flow(flow, model_name="ABIDE-Guided-Flow-variational"):
        # Create a PyTorch Lightning trainer
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
                             max_epochs=1600,
                             gradient_clip_val=1.0,
                             callbacks=[PrintCallback(),
                                        ModelCheckpoint(save_weights_only=True, save_top_k=-1, every_n_epochs=250,
                                                        save_last=True),
                                        LearningRateMonitor("epoch")],
                             check_val_every_n_epoch=200)
        trainer.logger._log_graph = True
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

        result = None

        # Check whether pretrained model exists. If yes, load it and skip training)

        print("Start training", model_name)
        trainer.fit(flow, train_loader, val_loader)

        # Test best model on validation and test set if no result has been found
        # Testing can be expensive due to the importance sampling.
        start_time = time.time()
        val_result = trainer.test(flow, val_loader, verbose=False)
        duration = time.time() - start_time
        result = {"val": val_result, "time": duration / len(val_loader) / flow.import_samples}

        print(result)


    train_flow(flow_model(dequant_mode='variational'))
