## Deep Speaker: An End-to-End Neural Speaker Embedding System.
Unofficial Keras implementation of Deep Speaker | [Paper](https://arxiv.org/pdf/1705.02304.pdf) | [Pretrained Models](https://drive.google.com/open?id=18h2bmsAWrqoUMsh_FQHDDxp7ioGpcNBa).

### Sample Results

Models were trained on clean speech data. Keep in mind that the performance will be lower on noisy data.

 *Model name* | *Testing dataset* | *Num speakers* | *F* | *TPR* | *ACC* | *EER* | Training Logs | Download model
 | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
ResCNN Softmax trained          | [LibriSpeech](http://www.openslr.org/12/) all(*) | 2484 | 0.789 | 0.733 | 0.996 | 0.043 | [Click](https://docs.google.com/document/d/1ZZjBk5TgFgaY9GgOcHaieOpiyB9lB6oTRSdk6g8FPRs) | [Click](https://drive.google.com/open?id=1SJBmHpnaW1VcbFWP6JfvbT3wWP9PsqxS)
ResCNN Softmax+Triplet trained  | [LibriSpeech](http://www.openslr.org/12/) all(*) | 2484 | 0.843 | 0.825 | 0.997 | 0.025 | [Click](https://docs.google.com/document/d/1mL0Jb8IpA7DOzFci71RT1OYTq7Kkw2DjTkI4BRpEzKc) | [Click](https://drive.google.com/file/d/1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP)

(*) all includes: dev-clean, dev-other, test-clean, test-other, train-clean-100, train-clean-360, train-other-500.

### Overview

Deep Speaker is a neural speaker embedding system that maps utterances to a hypersphere where speaker similarity is measured by cosine similarity. The embeddings generated by Deep Speaker can be used for many tasks, including speaker identification,
verification, and clustering.

## Getting started
### Install dependencies
#### Requirements
- tensorflow>=2.0
- keras>=2.3.1
```bash
pip install -r requirements.txt
```

If you see this error: `libsndfile not found`, run this: `sudo apt-get install libsndfile-dev`.

### Training

The code for training is available in this repository. It takes a bit less than a week with a GTX1070 to train the models.

System requirements for a complete training are:
- At least 300GB of free disk space on a fast SSD (250GB just for all the uncompressed + processed data)
- 32GB of memory and at least 32GB of swap (can create swap with SSD space).
- A NVIDIA GPU such as the 1080Ti.

```bash
pip uninstall -y tensorflow && pip install tensorflow-gpu
./deep-speaker download_librispeech    # if the download is too slow, consider replacing [wget] by [axel -n 10 -a] in download_librispeech.sh.
./deep-speaker build_mfcc              # will build MFCC for softmax pre-training and triplet training.
./deep-speaker build_model_inputs      # will build inputs for softmax pre-training.
./deep-speaker train_softmax           # takes ~3 days.
./deep-speaker train_triplet           # takes ~3 days.
```

NOTE: If you want to use your own dataset, make sure you follow the directory structure of librispeech. Audio files have to be in `.flac`. format. If you have `.wav`, you can use `ffmpeg` to make the conversion. Both formats are flawless (FLAC is compressed WAV).

### Test instruction using pretrained model
- Download the trained models
 

 *Model name* | *Used datasets for training* | *Num speakers* | *Model Link* | 
 | :--- | :--- | :--- | :--- |
ResCNN Softmax trained  | [LibriSpeech](http://www.openslr.org/12/) train-clean-360 | 921 | [Click](https://drive.google.com/open?id=1SJBmHpnaW1VcbFWP6JfvbT3wWP9PsqxS)
ResCNN Softmax+Triplet trained  | [LibriSpeech](http://www.openslr.org/12/) all | 2484 | [Click](https://drive.google.com/file/d/1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP)

* Run with pretrained model

```python
import random

import numpy as np

from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
from conv_models import DeepSpeakerModel
from test import batch_cosine_similarity

# Reproducible results.
np.random.seed(123)
random.seed(123)

# Define the model here.
model = DeepSpeakerModel()

# Load the checkpoint.
model.m.load_weights('ResCNN_triplet_training_checkpoint_265.h5', by_name=True)

# Sample some inputs for WAV/FLAC files for the same speaker.
# To have reproducible results every time you call this function, set the seed every time before calling it.
# np.random.seed(123)
# random.seed(123)
mfcc_001 = sample_from_mfcc(read_mfcc('samples/PhilippeRemy/PhilippeRemy_001.wav', SAMPLE_RATE), NUM_FRAMES)
mfcc_002 = sample_from_mfcc(read_mfcc('samples/PhilippeRemy/PhilippeRemy_002.wav', SAMPLE_RATE), NUM_FRAMES)

# Call the model to get the embeddings of shape (1, 512) for each file.
predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))

# Do it again with a different speaker.
mfcc_003 = sample_from_mfcc(read_mfcc('samples/1255-90413-0001.flac', SAMPLE_RATE), NUM_FRAMES)
predict_003 = model.m.predict(np.expand_dims(mfcc_003, axis=0))

# Compute the cosine similarity and check that it is higher for the same speaker.
print('SAME SPEAKER', batch_cosine_similarity(predict_001, predict_002)) # SAME SPEAKER [0.81564593]
print('DIFF SPEAKER', batch_cosine_similarity(predict_001, predict_003)) # DIFF SPEAKER [0.1419204]
```

* Commands to reproduce the test results after the training

```bash
$ export CUDA_VISIBLE_DEVICES=0; python cli.py test-model --working_dir ~/.deep-speaker-wd/triplet-training/ --
checkpoint_file checkpoints-softmax/ResCNN_checkpoint_102.h5
f-measure = 0.789, true positive rate = 0.733, accuracy = 0.996, equal error rate = 0.043
```

```bash
$ export CUDA_VISIBLE_DEVICES=0; python cli.py test-model --working_dir ~/.deep-speaker-wd/triplet-training/ --checkpoint_file checkpoints-triplets/ResCNN_checkpoint_265.h5
f-measure = 0.849, true positive rate = 0.798, accuracy = 0.997, equal error rate = 0.025
```
# Deep_speaker
