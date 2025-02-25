from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from moises_dataset import MoisesDataset


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, chunk_files_dir: str, batch_size: int = 2, num_workers: int = 4):
        super().__init__()
        self.chunk_files_dir = chunk_files_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str]):
        self.dataset = MoisesDataset(self.chunk_files_dir)
        dataset_size = len(self.dataset)
        train_size = int(0.8 * dataset_size)
        val_size = int(0.1 * dataset_size)
        test_size = dataset_size - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
