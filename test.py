import gc
import os
import sys

import librosa
import numpy as np
import pytorch_lightning as pl
import scipy.linalg
import torch
import torch.nn as nn
import torchaudio
import torchopenl3
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import Dataset

from audio_data_module import AudioDataModule
from bass_gan import BassGAN


def compute_fid(features_real, features_fake, eps=1e-6):
    mu_real = np.mean(features_real, axis=0)
    mu_fake = np.mean(features_fake, axis=0)

    sigma_real = np.cov(features_real, rowvar=False)
    sigma_fake = np.cov(features_fake, rowvar=False)

    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_fake += np.eye(sigma_fake.shape[0]) * eps

    diff = mu_real - mu_fake
    diff_squared = diff.dot(diff)

    covmean, _ = scipy.linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff_squared + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid


def extract_features_openl3(audio, sample_rate):
    audio_np = audio.squeeze(0).detach().cpu().numpy().astype(np.float32)

    embeddings, _ = torchopenl3.get_audio_embedding(
        audio_np,
        sr=sample_rate,
        input_repr="mel256",
        content_type="music",
        embedding_size=512,
        hop_size=1.0,
    )
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    emb_mean = np.mean(embeddings, axis=0)
    emb_mean = emb_mean.reshape(1, -1)
    return emb_mean


def extract_beats(audio_np, sample_rate):
    tempo, beat_frames = librosa.beat.beat_track(y=audio_np, sr=sample_rate)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
    return tempo, beat_times


def extract_key(audio_np, sample_rate):
    chroma = librosa.feature.chroma_stft(y=audio_np, sr=sample_rate)
    chroma_mean = np.mean(chroma, axis=1)
    key_index = np.argmax(chroma_mean)
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    detected_key = keys[key_index]
    return detected_key, None, None


class DummyBassDataset(Dataset):
    def __init__(self, num_samples=100, seg_length=64000, context_channels=10):
        self.num_samples = num_samples
        self.seg_length = seg_length
        self.context_channels = context_channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        real_bass = torch.randn(1, self.seg_length)
        stems = torch.randn(self.context_channels, self.seg_length)
        return real_bass, stems


class BassGANTest(pl.LightningModule):
    def __init__(self, model, sample_rate=16000):
        super().__init__()
        self.model = model
        self.sample_rate = sample_rate

        self.real_embeddings = []
        self.fake_embeddings = []

        self.tempo_diffs = []
        self.key_matches = []

    def test_step(self, batch, batch_idx):
        real_bass, stems = batch
        real_bass = real_bass.float()
        stems = stems.float()

        latent_dim = getattr(self.model, "latent_dim", 100)
        noise = torch.randn(real_bass.size(0), latent_dim, device=self.device)
        fake_bass = self.model.generator(noise, stems)

        for i in range(real_bass.size(0)):
            real_audio = real_bass[i]
            fake_audio = fake_bass[i]

            r_feat = extract_features_openl3(real_audio, self.sample_rate)
            f_feat = extract_features_openl3(fake_audio, self.sample_rate)

            self.real_embeddings.append(r_feat)
            self.fake_embeddings.append(f_feat)

        real_context_np = stems.mean(dim=1)[0].cpu().numpy()
        tempo_real, _ = extract_beats(real_context_np, self.sample_rate)

        fake_bass_np = fake_bass[0].squeeze(0).cpu().numpy()
        tempo_fake, _ = extract_beats(fake_bass_np, self.sample_rate)
        tempo_diff = abs(tempo_real - tempo_fake)
        self.tempo_diffs.append(float(tempo_diff))

        real_audio_np = real_bass[0].squeeze(0).cpu().numpy()
        fake_audio_np = fake_bass[0].squeeze(0).cpu().numpy()
        key_real, _, _ = extract_key(real_audio_np, self.sample_rate)
        key_fake, _, _ = extract_key(fake_audio_np, self.sample_rate)
        key_match = 1.0 if key_real == key_fake else 0.0
        self.key_matches.append(key_match)

        self.log("tempo_diff", float(tempo_diff), prog_bar=True)
        self.log("key_match", float(key_match), prog_bar=True)

        del real_bass, stems, fake_bass
        gc.collect()
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):

        real_arr = np.concatenate(self.real_embeddings, axis=0)
        fake_arr = np.concatenate(self.fake_embeddings, axis=0)

        fid_score = compute_fid(real_arr, fake_arr, eps=1e-6)

        self.log("avg_fid", float(fid_score))
        self.log("avg_tempo_diff", float(np.mean(self.tempo_diffs)))
        self.log("avg_key_match", float(np.mean(self.key_matches)))

        self.real_embeddings = []
        self.fake_embeddings = []
        self.tempo_diffs = []
        self.key_matches = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class DummyConditionalGAN(pl.LightningModule):
    def __init__(self, latent_dim=100, seg_length=64000):
        super().__init__()
        self.latent_dim = latent_dim
        self.seg_length = seg_length
        self.generator = nn.Sequential(nn.Linear(latent_dim, seg_length), nn.Tanh())

    def forward(self, noise, stems):
        x = self.generator(noise)
        x = x.view(noise.size(0), 1, self.seg_length)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":

    pl.seed_everything(42)
    checkpoint_path = os.environ.get(
        "BEST_CHECKPOINT",
        "tb_logs/bass-gan/version_96/checkpoints/epoch=682-step=117714.ckpt",
    )
    model = BassGAN.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()

    chunk_files_dir = os.environ.get(
        "CHUNK_FILES_DIR", "/teamspace/datasets/imagenet/test"
    )
    data_module = AudioDataModule(
        chunk_files_dir=chunk_files_dir,
        batch_size=8,
        num_workers=4,
    )
    data_module.setup(stage=None)

    test_module = BassGANTest(model=model, sample_rate=16000)

    csv_logger = CSVLogger("test-logs", "test-bass-gan")
    tb_logger = TensorBoardLogger("test-tb_logs", name="test-bass-gan")

    trainer = pl.Trainer(devices=1, max_epochs=1, logger=[csv_logger, tb_logger])
    trainer.test(test_module, datamodule=data_module)
