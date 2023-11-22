import os
import glob
import torch, torchaudio
from dataset_loader import DatasetLoader
import scipy.io.wavfile as wavfile
from python_speech_features import logfbank
from utils import print_bar
import pandas as pd

def get_data_loader():
    audio_dir = "data"
    wavs_dir = "whales_data_wav"
    json_path = "data.json"
    train_csv_path = "data/train.csv"
    data_loader = DatasetLoader(wavs_dir, json_path)
    if os.path.exists(json_path): # skip preprocessing and indexing if already done
        data_loader.load_from_json()
    else:
        process_audio_data(audio_dir, "aiff", wavs_dir)
        mapper = get_file_to_label_mapper(train_csv_path)
        data_loader.index_files("wav", strategy_new_item, file_to_label_mapper = mapper)
        data_loader.save_to_json()    
    data_loader.process_strategty = strategy_get_feat_and_target
    data_loader.train_keys = list(data_loader.filter_items("partition", "train").keys())
    data_loader.dev_keys = list(data_loader.filter_items("partition", "dev").keys())
    data_loader.test_keys = list(data_loader.filter_items("partition", "test").keys())
    return data_loader

def process_audio_data(audio_dir: str, audio_ext: str, wavs_dir: str) -> None:
    """Given a directory to audio files with a specific audio extension, transform all audios into wav files."""
    if not os.path.exists(wavs_dir):
        os.makedirs(wavs_dir)
    file_paths = os.path.join(audio_dir, "**", f"*.{audio_ext}")
    file_paths = glob.glob(file_paths, recursive=True)
    for i, file_path in enumerate(file_paths):
        print_bar(i + 1, len(file_paths), prefix="Transforming audio files:", length=40)
        basename = os.path.basename(file_path)
        wav_name = basename.replace(audio_ext, "wav")
        wav_path = os.path.join(wavs_dir, wav_name)
        if os.path.exists(wav_path): continue
        transform_audio(file_path, wav_path, 2000)


def transform_audio(input_path: str, output_path: str, sr: int) -> torch.Tensor:
    """Transform an audio input to .wav file 16 bit integer and a given sampling rate."""
    output_dir = os.path.dirname(output_path)
    arg = f'ffmpeg -i {input_path} -ar {sr} -ac 1 -c:a pcm_s16le -af "volume=0dB" -hide_banner -v 0 -y {output_path}'
    os.system(arg)


def feature_extraction(wav_path: str) -> torch.Tensor:
    """Extract features given a wav_path
    """
    (rate,sig) = wavfile.read(wav_path)
    feats = logfbank(sig, samplerate=rate, winlen=0.025, winstep=0.01, nfilt=40)
    feats = torch.Tensor(feats)
    feats = torch.mean(feats, 0)
    return feats

def strategy_get_feat_and_target(item: {str: str}) -> (torch.Tensor, torch.Tensor):
    """Get features and the target based on a given item dictionary
    """
    feats = feature_extraction(item["wav_path"])
    target = item["label"]
    target = torch.nn.functional.one_hot(torch.tensor(target), num_classes=7).float()
    return feats, target

def get_file_to_label_mapper(csv_path:str) -> {str: int}:
    """A mapper from the file id to the label (0 or 1)"""
    mapper = {}
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        idx = row["clip_name"].replace(".aiff", "")
        label = row["label"]
        mapper[idx] = label
    return mapper

def strategy_new_item(wav_path:str, file_to_label_mapper:{str:int}) -> {str: str}:
    """Make a new dict item based on a given wav_file.
    The dict item stores information about the wav file as a dictionary.
    """
    duration = torchaudio.info(wav_path).num_frames
    sample_rate = torchaudio.info(wav_path).sample_rate
    wav_name = os.path.basename(wav_path)
    wav_name = os.path.splitext(wav_name)[0] # removing the extension
    part = "train"
    if "test" in wav_name: part = "test"
    train_keys = list(file_to_label_mapper.keys())
    dev_keys = train_keys[25000:] # assigning a portion of the training set as development
    if wav_name in dev_keys: part = "dev"
    label = -1
    if wav_name in list(file_to_label_mapper.keys()):
        label = file_to_label_mapper[wav_name] 
    item = {
        "wav_name"   : wav_name,
        "wav_path"   : wav_path,
        "duration"   : duration,
        "rate"       : sample_rate,
        "label"      : label,
        "partition"  : part
    }
    return wav_name, item
    

if __name__ == '__main__':
    get_data_loader()
