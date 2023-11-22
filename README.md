# Powder AI Tech Test
Candidate Name: Sina ALI SAMIR

# Installation
Setup environment with:

```
 pip install -r requirements.txt
```

**Note**: `ffmpeg` is needed for audio processing.

# Project description
In this project we train pytorch models to detect whale sounds. 

It consists of the following files:
- `dataset_loader`: a generic class to load a dataset.
- `whale_data_loader`: scripts to preprocess data, and make a specific dataloader for our objective. 
- `train`: train a model given the whale_data_loader.
- `eval`: evaluate the trained model.
- `ìnfer`: can classify a given wav file as whale sound or not.

# Usage


## Inference

```bash
python3 infer.py -i [path_to_wav_file] -m [path_to_trained_model]  
```

## Training 

```bash
python3 train.py -i [dir_to_aiff_files] -w [dir_to_csv_files(output)] -j [path_to_json_file(output)] -c [path_to_csv_file_labels]
```

# TODO
- get variables from whale data loader to train.py
- move files to src, and see if everything still works! (change input paths!)
- dataset_loader.py does not necessary need to get wavs_dir in its __init__
- add unit tests for train.py, ideally make a Trainer class before that
- add grid search for hyperparameter tuning
- add other (and more complex) models
