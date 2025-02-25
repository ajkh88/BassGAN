# BassGAN

A Generative Adversarial Network (GAN) model for generating bass lines from musical stems.


## Setup

```bash
pip install -r requirements.txt
```

## Usage
 - See [MoisesDB](https://github.com/moises-ai/moises-db) for details on downloading the dataset.
 - Once the dataset is downloaded, set the `MOISES_DATA_PATH` in prep data to point at the directory you downloaded to.
 - Set the `CHUNK_FILES_DIR` to the output dir from the prep data job.
 - Use the following command to run the training:
```bash
python train.py
```
See `train.py` for more details of flags and hyperparameters.

For best results, use the [Lightning.ai](https://lightning.ai/) platform and a GPU.
