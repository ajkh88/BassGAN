import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio.transforms as T


class PhaseShuffle(nn.Module):
    def __init__(self, rad):
        super(PhaseShuffle, self).__init__()
        self.rad = rad

    def forward(self, x):
        if self.rad == 0:
            return x
        batch_size, channels, t = x.size()
        phase_shifts = torch.randint(
            -self.rad, self.rad + 1, (batch_size,), device=x.device
        )
        x_shifted = []
        for i, shift in enumerate(phase_shifts):
            shift = int(shift.item())
            if shift == 0:
                x_shifted.append(x[i : i + 1])
            elif shift > 0:
                left_pad = x[i, :, :shift].flip(dims=[-1])
                x_shifted.append(
                    torch.cat([left_pad.unsqueeze(0), x[i : i + 1, :, :-shift]], dim=2)
                )
            else:
                shift = abs(shift)
                right_pad = x[i, :, -shift:].flip(dims=[-1])
                x_shifted.append(
                    torch.cat([x[i : i + 1, :, shift:], right_pad.unsqueeze(0)], dim=2)
                )
        return torch.cat(x_shifted, dim=0)


class ConditionEncoder(nn.Module):
    def __init__(self, condition_dim=100):
        super(ConditionEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=10, out_channels=64, kernel_size=25, stride=4, padding=11
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 512, kernel_size=25, stride=4, padding=11),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(512, condition_dim)

    def forward(self, cond):
        x = self.encoder(cond)
        x = self.fc(x)
        return x


class BassGANGenerator(nn.Module):
    def __init__(
        self,
        latent_dim=100,
        condition_dim=100,
        out_length=64000,
        channels=64,
        upsample_layers=4,
    ):
        super(BassGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.out_length = out_length
        self.upsample_layers = upsample_layers

        self.initial_length = out_length // (4**upsample_layers)

        self.condition_encoder = ConditionEncoder(condition_dim=condition_dim)
        self.fc = nn.Linear(latent_dim + condition_dim, channels * self.initial_length)

        channel_list = [channels, channels // 2, channels // 4, channels // 8, 1]
        layers = []
        for i in range(upsample_layers):
            layers.append(nn.Upsample(scale_factor=4, mode="nearest"))
            layers.append(
                nn.Conv1d(
                    in_channels=channel_list[i],
                    out_channels=channel_list[i + 1],
                    kernel_size=9,
                    stride=1,
                    padding=4,
                )
            )
            if i < upsample_layers - 1:
                layers.append(nn.InstanceNorm1d(channel_list[i + 1], affine=True))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(0.2))
            else:
                layers.append(nn.Tanh())
        self.deconv = nn.Sequential(*layers)

    def forward(self, z, cond):
        cond_emb = self.condition_encoder(cond)
        combined = torch.cat([z, cond_emb], dim=1)
        x = self.fc(combined)
        x = x.view(z.size(0), -1, self.initial_length)
        x = self.deconv(x)
        return x


class BassGANDiscriminator(nn.Module):
    def __init__(
        self, in_length=64000, channels=[11, 64, 128, 256, 512], phase_shuffle_rad=2
    ):
        super(BassGANDiscriminator, self).__init__()
        layers = []
        for i in range(len(channels) - 1):
            conv = nn.Conv1d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=25,
                stride=4,
                padding=11,
            )
            conv = nn.utils.spectral_norm(conv)
            layers.append(conv)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv = nn.Sequential(*layers)
        self.phase_shuffle = PhaseShuffle(rad=phase_shuffle_rad)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], 1)

    def forward(self, bass, cond, return_features=False):
        x = torch.cat([bass, cond], dim=1)
        features = self.conv(x)
        features = self.phase_shuffle(features)
        pooled = self.global_pool(features)
        pooled = pooled.squeeze(-1)
        score = self.fc(pooled)
        if return_features:
            return pooled, score
        return score


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif "Linear" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class BassGAN(pl.LightningModule):
    def __init__(
        self,
        seg_length: int = 4 * 16000,
        latent_dim: int = 100,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.9,
        batch_size: int = 64,
        lambda_gp: float = 0.5,
        lambda_fm: float = 0.5,
        lambda_perceptual: float = 2.0,
        n_critic: int = 2,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.seg_length = seg_length
        self.n_critic = n_critic

        self.generator = BassGANGenerator(
            latent_dim=self.hparams.latent_dim,
            out_length=seg_length,
            channels=64,
            upsample_layers=4,
        )
        self.discriminator = BassGANDiscriminator(
            in_length=seg_length, channels=[11, 64, 128, 256, 512], phase_shuffle_rad=2
        )

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.sample_rate = kwargs.get("sample_rate", 16000)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=1024, hop_length=256, n_mels=64
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def forward(self, noise, stems):
        return self.generator(noise, stems)

    def compute_gradient_penalty(self, real_samples, fake_samples, stems):
        alpha = torch.rand(real_samples.size(0), 1, 1, device=real_samples.device)
        interpolates = (
            alpha * real_samples + (1 - alpha) * fake_samples
        ).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates, stems)
        grad_outputs = torch.ones_like(d_interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_norm = (
            gradients.view(gradients.size(0), -1).norm(2, dim=1).clamp(min=1e-8)
        )
        penalty = self.hparams.lambda_gp * ((gradient_norm - 1) ** 2).mean()
        return penalty

    def compute_perceptual_loss(self, real_audio, fake_audio):
        mel_real = self.mel_transform(real_audio)
        mel_fake = self.mel_transform(fake_audio)
        mel_real_db = self.amplitude_to_db(mel_real)
        mel_fake_db = self.amplitude_to_db(mel_fake)
        return F.l1_loss(mel_fake_db, mel_real_db.detach())

    def training_step(self, batch, batch_idx):
        real_bass, stems = batch
        self.log("input_min", real_bass.min(), on_step=True)
        self.log("input_max", real_bass.max(), on_step=True)
        real_bass = real_bass.float()
        stems = stems.float()
        batch_size = real_bass.size(0)
        opt_g, opt_d = self.optimizers()

        for _ in range(self.n_critic):
            noise = torch.randn(
                batch_size, self.hparams.latent_dim, device=self.device
            ).type_as(real_bass)
            fake_bass = self.generator(noise, stems).detach()
            d_real = self.discriminator(real_bass, stems)
            d_fake = self.discriminator(fake_bass, stems)
            gp = self.compute_gradient_penalty(real_bass, fake_bass, stems)
            d_loss = d_fake.mean() - d_real.mean() + gp

            self.toggle_optimizer(opt_d)
            self.manual_backward(d_loss)
            grad_norm_d = sum(
                p.grad.data.norm(2).item()
                for p in self.discriminator.parameters()
                if p.grad is not None
            )
            self.log(
                "d_grad_norm",
                grad_norm_d,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), max_norm=1.0
            )
            opt_d.step()
            opt_d.zero_grad()
            self.untoggle_optimizer(opt_d)

        d_real = self.discriminator(real_bass, stems)
        d_fake = self.discriminator(fake_bass, stems)
        self.log(
            "d_real_mean",
            d_real.mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "d_fake_mean",
            d_fake.mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        noise = torch.randn(
            batch_size, self.hparams.latent_dim, device=self.device
        ).type_as(real_bass)
        fake_bass = self.generator(noise, stems)
        fake_features, d_fake_score = self.discriminator(
            fake_bass, stems, return_features=True
        )
        real_features, _ = self.discriminator(real_bass, stems, return_features=True)
        fm_loss = F.l1_loss(fake_features, real_features.detach())
        adv_loss = -d_fake_score.mean()
        perc_loss = self.compute_perceptual_loss(real_bass, fake_bass)
        g_loss = (
            adv_loss
            + self.hparams.lambda_fm * fm_loss
            + self.hparams.lambda_perceptual * perc_loss
        )

        self.toggle_optimizer(opt_g)
        self.manual_backward(g_loss)
        grad_norm_g = sum(
            p.grad.data.norm(2).item()
            for p in self.generator.parameters()
            if p.grad is not None
        )
        self.log(
            "g_grad_norm",
            grad_norm_g,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        fake_mean = fake_bass.mean().item()
        fake_std = fake_bass.std(unbiased=False).item()
        self.log(
            "g_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            "d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            "fm_loss",
            fm_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "perc_loss",
            perc_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "fake_mean",
            fake_mean,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "fake_std",
            fake_std,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):
        real_bass, stems = batch
        real_bass = real_bass.float()
        stems = stems.float()
        batch_size = real_bass.size(0)
        with torch.no_grad():
            noise = torch.randn(
                batch_size, self.hparams.latent_dim, device=self.device
            ).type_as(real_bass)
            fake_bass = self.generator(noise, stems)
            d_real = self.discriminator(real_bass, stems)
            d_fake = self.discriminator(fake_bass, stems)
            g_loss = -d_fake.mean()
            d_loss = d_fake.mean() - d_real.mean()
            real_score = d_real.mean()
            fake_score = d_fake.mean()
        self.log("val_g_loss", g_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_d_loss", d_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(
            "val_real_score", real_score, prog_bar=True, on_epoch=True, sync_dist=True
        )
        self.log(
            "val_fake_score", fake_score, prog_bar=True, on_epoch=True, sync_dist=True
        )

        if batch_idx == 0:
            sample_fake = fake_bass[0].cpu()
            sample_real = real_bass[0].cpu()
            sample_stems = stems[0].cpu()
            stems_mix = sample_stems.mean(dim=0, keepdim=True)

            sample_fake = self.normalise_audio(sample_fake)
            sample_real = self.normalise_audio(sample_real)
            stems_mix = self.normalise_audio(stems_mix)
            generated_in_context = stems_mix + sample_fake
            generated_in_context = self.normalise_audio(generated_in_context)

            if hasattr(self.logger, "experiment"):
                tensorboard = self.logger.experiment
                tensorboard.add_audio(
                    "generated_bass",
                    sample_fake,
                    global_step=self.current_epoch,
                    sample_rate=self.sample_rate,
                )
                tensorboard.add_audio(
                    "real_bass",
                    sample_real,
                    global_step=self.current_epoch,
                    sample_rate=self.sample_rate,
                )
                tensorboard.add_audio(
                    "stems_mix",
                    stems_mix,
                    global_step=self.current_epoch,
                    sample_rate=self.sample_rate,
                )
                tensorboard.add_audio(
                    "generated_in_context",
                    generated_in_context,
                    global_step=self.current_epoch,
                    sample_rate=self.sample_rate,
                )

            real_sample = real_bass[0].unsqueeze(0)
            fake_sample = fake_bass[0].unsqueeze(0)
            mel_real = self.mel_transform(real_sample).squeeze(0)
            mel_fake = self.mel_transform(fake_sample).squeeze(0)
            mel_real_db = self.amplitude_to_db(mel_real).squeeze(0)
            mel_fake_db = self.amplitude_to_db(mel_fake).squeeze(0)

            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            im0 = axs[0].imshow(
                mel_real_db.cpu().numpy(),
                origin="lower",
                aspect="auto",
                interpolation="none",
            )
            axs[0].set_title("Real Mel Spectrogram")
            plt.colorbar(im0, ax=axs[0])
            im1 = axs[1].imshow(
                mel_fake_db.cpu().numpy(),
                origin="lower",
                aspect="auto",
                interpolation="none",
            )
            axs[1].set_title("Fake Mel Spectrogram")
            plt.colorbar(im1, ax=axs[1])

            self.logger.experiment.add_figure(
                "Spectrograms", fig, global_step=self.current_epoch
            )
            plt.close(fig)

    def normalise_audio(self, audio_tensor):
        max_val = audio_tensor.abs().max()
        if max_val > 1:
            return audio_tensor / max_val
        return audio_tensor

    def configure_optimizers(self):
        lr_gen = self.hparams.lr
        lr_disc = 5e-5
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_g = optim.Adam(self.generator.parameters(), lr=lr_gen, betas=(b1, b2))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr_disc, betas=(b1, b2))
        scheduler_g = optim.lr_scheduler.StepLR(opt_g, step_size=500, gamma=0.5)
        scheduler_d = optim.lr_scheduler.StepLR(opt_d, step_size=500, gamma=0.5)
        return [opt_g, opt_d], [scheduler_g, scheduler_d]
