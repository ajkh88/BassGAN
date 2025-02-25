import argparse
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from audio_data_module import AudioDataModule
from bass_gan import BassGAN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=100)
    args = parser.parse_args()

    chunk_files_dir = os.environ.get(
        "CHUNK_FILES_DIR", "/teamspace/datasets/imagenet/test"
    )
    data_module = AudioDataModule(
        chunk_files_dir=chunk_files_dir,
        batch_size=args.batch_size,
        num_workers=16,
    )
    data_module.setup(stage=None)

    segment_length = 4 * 16000  # 4 seconds at 16KHz

    model = BassGAN(seg_length=segment_length, batch_size=args.batch_size)

    torch.set_float32_matmul_precision("medium")

    csv_logger = CSVLogger("logs", "bass-gan")
    tb_logger = TensorBoardLogger("tb_logs", name="bass-gan")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_g_loss",
        dirpath="checkpoints/",
        filename="best-checkpoint-1",
        save_top_k=1,
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_g_loss", min_delta=0.001, patience=100, verbose=True, mode="min"
    )

    trainer = Trainer(
        max_epochs=1500,
        logger=[tb_logger, csv_logger],
        strategy=DDPStrategy(find_unused_parameters=True),
        log_every_n_steps=40,
        default_root_dir="checkpoints",
        # callbacks=[checkpoint_callback, early_stop_callback]
    )
    checkpoint_path = os.environ.get(
        "BEST_CHECKPOINT",
        "tb_logs/bass-gan/version_96/checkpoints/epoch=682-step=117714.ckpt",
    )
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=checkpoint_path,
    )
