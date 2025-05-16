import os
import random
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy.signal import butter, lfilter
import yaml


def butter_filter(audio, sample_rate, cutoff, btype, order=6):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return lfilter(b, a, audio)


def eq_high_cut(audio, sample_rate):
    return butter_filter(audio, sample_rate, cutoff=4000, btype="low")


def eq_low_cut(audio, sample_rate):
    return butter_filter(audio, sample_rate, cutoff=200, btype="high")


def eq_mid_boost(audio, sample_rate):
    from scipy.signal import sosfilt, butter

    sos = butter(
        4,
        [800 / (0.5 * sample_rate), 2000 / (0.5 * sample_rate)],
        btype="band",
        output="sos",
    )
    boosted = sosfilt(sos, audio) * 2
    return audio + boosted


def hard_compression(audio, sample_rate=None, threshold_db=-20, ratio=8):
    threshold = 10 ** (threshold_db / 20)
    compressed = np.copy(audio)
    over = np.abs(audio) > threshold
    compressed[over] = np.sign(audio[over]) * (
        threshold + (np.abs(audio[over]) - threshold) / ratio
    )
    return compressed


AUGMENTATION_FUNCS = {
    "eq_high_cut": eq_high_cut,
    "eq_low_cut": eq_low_cut,
    "eq_mid_boost": eq_mid_boost,
    "hard_compression": hard_compression,
}


class DatasetGenerator:
    def __init__(self, config_path="config.yaml"):
        print(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.output_folder = cfg.get("output_folder", "./output")
        self.input_folder = cfg.get("input_folder", "./input")
        self.chunk_sizes_seconds = cfg.get("chunk_sizes_seconds", 30)
        self.sample_rate = cfg.get("sample_rate", 48000)
        self.dataset_length_goal_minutes = cfg.get("dataset_length_goal_minutes", 1)
        self.target_lufs = cfg.get("target_lufs", -20)
        self.augmentations = cfg.get(
            "augmentations",
            ["eq_high_cut", "eq_low_cut", "eq_mid_boost", "hard_compression"],
        )
        print(f"Config loaded: output_folder={self.output_folder}, input_folder={self.input_folder}, chunk_sizes_seconds={self.chunk_sizes_seconds}, sample_rate={self.sample_rate}, dataset_length_goal_minutes={self.dataset_length_goal_minutes}, target_lufs={self.target_lufs}, augmentations={self.augmentations}")

    def list_audio_files(self):
        print(f"Listing audio files in {self.input_folder}")
        exts = (".wav", ".mp3", ".ogg", ".aif", ".aiff")
        files = [
            os.path.join(self.input_folder, f)
            for f in os.listdir(self.input_folder)
            if f.lower().endswith(exts)
        ]
        print(f"Found {len(files)} audio files with extensions {exts}.")
        return files

    def chunk_audio(self, audio, chunk_size_samples):
        total_samples = len(audio)
        chunks = [
            audio[i : i + chunk_size_samples]
            for i in range(0, total_samples, chunk_size_samples)
            if len(audio[i : i + chunk_size_samples]) == chunk_size_samples
        ]
        print(f"Chunked audio into {len(chunks)} chunks of {chunk_size_samples} samples each.")
        return chunks

    def normalize_lufs(self, audio):
        meter = pyln.Meter(self.sample_rate)
        loudness = meter.integrated_loudness(audio)
        normalized = pyln.normalize.loudness(audio, loudness, self.target_lufs)
        print(f"Normalized audio from {loudness:.2f} LUFS to {self.target_lufs} LUFS")
        return normalized

    def generate(self):
        print(f"Ensuring output folder exists: {self.output_folder}")
        os.makedirs(self.output_folder, exist_ok=True)
        wav_files = self.list_audio_files()
        chunk_size_samples = int(self.chunk_sizes_seconds * self.sample_rate)
        total_chunks_needed = int(
            (self.dataset_length_goal_minutes * 60) / self.chunk_sizes_seconds
        )
        print(f"Need total {total_chunks_needed} chunks for dataset goal.")
        all_chunks = []
        for wav_path in wav_files:
            print(f"Reading file: {wav_path}")
            audio_data = sf.read(wav_path)
            if isinstance(audio_data, tuple):
                audio, sr = audio_data[:2]
            else:
                print(f"Skipping file {wav_path}: could not read audio and sample rate.")
                continue
            if sr != self.sample_rate:
                print(f"Skipping file {wav_path}: sample rate {sr} != {self.sample_rate}")
                continue
            # Convert to mono if stereo
            if hasattr(audio, 'ndim') and audio.ndim > 1:
                print(f"Converting stereo to mono for {wav_path}")
                audio = np.mean(audio, axis=1)
            chunks = self.chunk_audio(audio, chunk_size_samples)
            all_chunks.extend(chunks)
        print(f"Total chunks collected: {len(all_chunks)}")
        random.shuffle(all_chunks)
        all_chunks = all_chunks[:total_chunks_needed]
        aug_cycle = (
            self.augmentations * ((total_chunks_needed // len(self.augmentations)) + 1)
        )[:total_chunks_needed]
        random.shuffle(aug_cycle)
        print("Starting to process and write chunk pairs...")
        for idx, (chunk, aug_name) in enumerate(zip(all_chunks, aug_cycle)):
            print(f"Processing chunk {idx+1}/{len(all_chunks)} with augmentation '{aug_name}'")
            target = self.normalize_lufs(chunk)
            aug_fn = AUGMENTATION_FUNCS[aug_name]
            src = aug_fn(target, self.sample_rate)
            src = self.normalize_lufs(src)
            target_path = os.path.join(self.output_folder, f"{idx:04d}_target.wav")
            src_path = os.path.join(self.output_folder, f"{idx:04d}_src.wav")
            sf.write(target_path, target, self.sample_rate)
            sf.write(src_path, src, self.sample_rate)
            print(f"Wrote: {target_path} and {src_path}")
        print("Dataset generation complete.")
        # Write manifest of all src/target pairs
        manifest_path = os.path.join(self.output_folder, "manifest.txt")
        with open(manifest_path, "w") as mf:
            for idx in range(len(all_chunks)):
                src = f"{idx:04d}_src.wav"
                target = f"{idx:04d}_target.wav"
                mf.write(f"{src} {target}\n")
        print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    print("Starting dataset generation...")
    generator = DatasetGenerator("config.yaml")
    generator.generate()
