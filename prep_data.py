import os
import pickle

from dataclasses import dataclass

import numpy as np
import torch

from lightning.data import map, Machine
from moisesdb.dataset import MoisesDB
from moisesdb.track import MoisesDBTrack


@dataclass(frozen=True)
class ChunkKey:
    track: str
    start: int

    def to_filename(self) -> str:
        return f"{self.track}-{self.start}"


STEM_CATEGORIES = {
    "other_keys",
    "vocals",
    "other",
    "drums",
    "piano",
    "bowed_strings",
    "other_plucked",
    "percussion",
    "wind",
    "guitar",
    "bass",
}
INPUT_CATEGORIES = sorted(list(STEM_CATEGORIES - {"bass"}))
CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(INPUT_CATEGORIES)}

SAMPLE_RATE = 16000
SEGMENT_LENGTH_SECS = 4

CHUNK_SIZE = SAMPLE_RATE * SEGMENT_LENGTH_SECS

OUTPUT_DIR: str = "/teamspace/datasets/prep-data/chunks"
MOISES_DATA_PATH: str = "/teamspace/uploads/moisesdb/moisesdb"

db = MoisesDB(MOISES_DATA_PATH, sample_rate=SAMPLE_RATE)


def convert_stems_to_mono(stems: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mono_stems = {}
    for stem_name, stem in stems.items():
        mono_stems[stem_name] = stem.mean(axis=0)
    return mono_stems


def get_min_stem_length(stems: dict[str, np.ndarray]) -> int:
    lengths = set([stem.shape[0] for stem in stems.values()])
    return min(lengths)


def split_track_into_chunks(idx: int, output_dir: str) -> None:
    track = db[idx]
    print(f"Processing track {track.id}")
    try:
        stems = convert_stems_to_mono(track.stems)
        track_length = get_min_stem_length(stems)
        print(
            f"Processing track {track.id} splitting into {track_length // CHUNK_SIZE} chunks"
        )
        bass = stems.pop("bass")
        for chunk_start in range(0, track_length, CHUNK_SIZE):
            chunk_key = ChunkKey(track=track.id, start=chunk_start)
            chunk_file_path = f"{output_dir}/{chunk_key.to_filename()}.pkl"
            context_channels = []
            chunk_end = min(chunk_start + CHUNK_SIZE, track_length)
            size = chunk_end - chunk_start
            print(f"Processing chunk starting at {chunk_start} of size {size}")
            if size != CHUNK_SIZE:
                print(
                    f"Chunk size is not correct {size} - need {CHUNK_SIZE}. WIll discard..."
                )
                continue 
            mask = []
            for stem_category in INPUT_CATEGORIES:
                stem = stems.get(stem_category, None)
                if stem is not None:
                    context_channels.append(stem[chunk_start:chunk_end])
                    mask.append(1)
                else:
                    context_channels.append(np.zeros(size))
                    mask.append(0)

            context_tensor = torch.stack(
                [torch.from_numpy(arr) for arr in context_channels]
            )

            chunk = {
                "bass": torch.from_numpy(bass[chunk_start:chunk_end]),
                "context": context_tensor,
                "stem_mask": torch.tensor(mask),
                "track_id": track.id,
                "start": chunk_start,
            }

            with open(chunk_file_path, "wb") as f:
                print(f"Saving {chunk_key.to_filename()} to disk")
                pickle.dump(chunk, f)

    except Exception as e:
        print(e)
        raise e


if __name__ == "__main__":
    outputs = map(
        split_track_into_chunks,
        list(range(len(db))),
        output_dir="/teamspace/datasets/imagenet/test",
        num_workers=os.cpu_count(),
    )
