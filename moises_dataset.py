import os
import pickle
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ChunkKey:
    track: str
    start: int


class MoisesDataset(Dataset):
    def __init__(self, chunk_files_dir: str):
        self.chunk_files_dir = chunk_files_dir

        self.chunk_keys = []
        self._read_chunk_files()

    def _read_chunk_files(self) -> None:

        for obj in os.listdir(self.chunk_files_dir):
            track_id = "-".join(obj.split("-")[:-1])
            chunk_start = int((obj.split("-")[-1].split(".")[0]))
            chunk_key = ChunkKey(track=track_id, start=chunk_start)
            self.chunk_keys.append(chunk_key)

    def __len__(self) -> int:
        return len(self.chunk_keys)

    def __getitem__(self, idx: int):
        chunk_key = self.chunk_keys[idx]

        with open(
            f"{self.chunk_files_dir}/{chunk_key.track}-{chunk_key.start}.pkl", "rb"
        ) as f:
            chunk = pickle.load(f)

        bass = chunk["bass"]
        context = chunk["context"]

        def safe_normalize(x):
            x_min = x.min()
            x_max = x.max()
            denom = x_max - x_min
            if (
                denom < 1e-8
            ):
                return torch.zeros_like(x)
            else:
                return 2 * (x - x_min) / denom - 1

        bass = safe_normalize(bass)
        context = safe_normalize(context)

        del chunk

        if bass.dim() == 1:
            bass = bass.unsqueeze(0)

        bass = bass.clone()
        context = context.clone()

        return bass, context
