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


def eq_high_cut(audio, sample_rate, randomness=0.5):
    # randomness: 0.0 = always 0dB, 1.0 = up to -24dB
    db_cut = -random.uniform(6, 24) * randomness
    gain = 10 ** (db_cut / 20)
    filtered = butter_filter(audio, sample_rate, cutoff=4000, btype="low")
    return filtered * gain

def eq_high_boost(audio, sample_rate, randomness=0.5):
    from scipy.signal import sosfilt, butter
    db_boost = random.uniform(3, 12) * randomness
    gain = 10 ** (db_boost / 20)
    # Make shelf wider by lowering cutoff to 2000 Hz
    sos = butter(4, 2000 / (0.5 * sample_rate), btype="high", output="sos")
    boosted = sosfilt(sos, audio) * gain
    return audio + boosted

def eq_low_boost(audio, sample_rate, randomness=0.5):
    from scipy.signal import sosfilt, butter
    db_boost = random.uniform(3, 12) * randomness
    gain = 10 ** (db_boost / 20)
    # Make shelf wider by raising cutoff to 800 Hz
    sos = butter(4, 800 / (0.5 * sample_rate), btype="low", output="sos")
    boosted = sosfilt(sos, audio) * gain
    return audio + boosted

def eq_high_and_low_cut(audio, sample_rate, randomness=0.5):
    db_cut_high = -random.uniform(6, 24) * randomness
    db_cut_low = -random.uniform(6, 24) * randomness
    gain_high = 10 ** (db_cut_high / 20)
    gain_low = 10 ** (db_cut_low / 20)
    low_cut = butter_filter(audio, sample_rate, cutoff=200, btype="high") * gain_low
    return butter_filter(low_cut, sample_rate, cutoff=4000, btype="low") * gain_high

def eq_high_and_low_boost(audio, sample_rate, randomness=0.5):
    high_boosted = eq_high_boost(audio, sample_rate, randomness)
    return eq_low_boost(high_boosted, sample_rate, randomness)

def eq_high_and_mid_boost(audio, sample_rate, randomness=0.5):
    high_boosted = eq_high_boost(audio, sample_rate, randomness)
    mid_boosted = eq_mid_boost(audio, sample_rate, randomness)
    return high_boosted + (mid_boosted - audio)

def eq_low_and_mid_boost(audio, sample_rate, randomness=0.5):
    low_boosted = eq_low_boost(audio, sample_rate, randomness)
    mid_boosted = eq_mid_boost(audio, sample_rate, randomness)
    return low_boosted + (mid_boosted - audio)

def eq_high_and_mid_cut(audio, sample_rate, randomness=0.5):
    high_cut = eq_high_cut(audio, sample_rate, randomness)
    from scipy.signal import sosfilt, butter
    db_cut = -random.uniform(6, 18) * randomness
    gain = 10 ** (db_cut / 20)
    sos = butter(
        4,
        [800 / (0.5 * sample_rate), 2000 / (0.5 * sample_rate)],
        btype="bandstop",
        output="sos",
    )
    mid_cut = sosfilt(sos, high_cut) * gain
    return mid_cut

def eq_low_and_mid_cut(audio, sample_rate, randomness=0.5):
    low_cut = eq_low_cut(audio, sample_rate, randomness)
    from scipy.signal import sosfilt, butter
    db_cut = -random.uniform(6, 18) * randomness
    gain = 10 ** (db_cut / 20)
    sos = butter(
        4,
        [800 / (0.5 * sample_rate), 2000 / (0.5 * sample_rate)],
        btype="bandstop",
        output="sos",
    )
    mid_cut = sosfilt(sos, low_cut) * gain
    return mid_cut

def no_change(audio, sample_rate):
    return np.copy(audio)


def eq_low_cut(audio, sample_rate, randomness=0.5):
    db_cut = -random.uniform(6, 24) * randomness
    gain = 10 ** (db_cut / 20)
    filtered = butter_filter(audio, sample_rate, cutoff=200, btype="high")
    return filtered * gain


def eq_mid_boost(audio, sample_rate, randomness=0.5):
    from scipy.signal import sosfilt, butter
    db_boost = random.uniform(3, 12) * randomness
    gain = 10 ** (db_boost / 20)
    sos = butter(
        4,
        [800 / (0.5 * sample_rate), 2000 / (0.5 * sample_rate)],
        btype="band",
        output="sos",
    )
    boosted = sosfilt(sos, audio) * gain
    return audio + boosted


def hard_compression(audio, sample_rate=None, threshold_db=-20, ratio=8):
    threshold = 10 ** (threshold_db / 20)
    compressed = np.copy(audio)
    over = np.abs(audio) > threshold
    compressed[over] = np.sign(audio[over]) * (
        threshold + (np.abs(audio[over]) - threshold) / ratio
    )
    return compressed


def low_compression(audio, sample_rate=None):
    return hard_compression(audio, sample_rate, threshold_db=-10, ratio=2)

def medium_compression(audio, sample_rate=None):
    return hard_compression(audio, sample_rate, threshold_db=-20, ratio=4)

def high_compression(audio, sample_rate=None):
    return hard_compression(audio, sample_rate, threshold_db=-30, ratio=8)

AUGMENTATION_FUNCS = {
    "eq_high_cut": lambda audio, sr, r=0.5: eq_high_cut(audio, sr, r),
    "eq_high_boost": lambda audio, sr, r=0.5: eq_high_boost(audio, sr, r),
    "eq_low_cut": lambda audio, sr, r=0.5: eq_low_cut(audio, sr, r),
    "eq_low_boost": lambda audio, sr, r=0.5: eq_low_boost(audio, sr, r),
    "eq_mid_boost": lambda audio, sr, r=0.5: eq_mid_boost(audio, sr, r),
    "eq_mid_cut": lambda audio, sr, r=0.5: audio - (eq_mid_boost(audio, sr, r) - audio),
    "eq_high_and_low_cut": lambda audio, sr, r=0.5: eq_high_and_low_cut(audio, sr, r),
    "eq_high_and_low_boost": lambda audio, sr, r=0.5: eq_high_and_low_boost(audio, sr, r),
    "eq_high_and_mid_boost": lambda audio, sr, r=0.5: eq_high_and_mid_boost(audio, sr, r),
    "eq_low_and_mid_boost": lambda audio, sr, r=0.5: eq_low_and_mid_boost(audio, sr, r),
    "eq_high_and_mid_cut": lambda audio, sr, r=0.5: eq_high_and_mid_cut(audio, sr, r),
    "eq_low_and_mid_cut": lambda audio, sr, r=0.5: eq_low_and_mid_cut(audio, sr, r),
    "no_change": lambda audio, sr, r=0.5: no_change(audio, sr),
    "low_compression": lambda audio, sr, r=0.5: low_compression(audio, sr),
    "medium_compression": lambda audio, sr, r=0.5: medium_compression(audio, sr),
    "high_compression": lambda audio, sr, r=0.5: high_compression(audio, sr),
    # "hard_compression": hard_compression,  # Uncomment if needed
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
        self.eq_randomness = cfg.get("eq_randomness", 0.5)
        print(f"Config loaded: output_folder={self.output_folder}, input_folder={self.input_folder}, chunk_sizes_seconds={self.chunk_sizes_seconds}, sample_rate={self.sample_rate}, dataset_length_goal_minutes={self.dataset_length_goal_minutes}, target_lufs={self.target_lufs}, augmentations={self.augmentations}, eq_randomness={self.eq_randomness}")

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
            target = np.copy(chunk)
            aug_fn = AUGMENTATION_FUNCS[aug_name]
            randomness = getattr(self, "eq_randomness", 0.5)
            src = aug_fn(target, self.sample_rate, randomness)
            # Ensure no clipping
            def prevent_clipping(audio):
                max_val = np.max(np.abs(audio))
                if max_val > 1.0:
                    return audio / max_val
                return audio
            target = prevent_clipping(target)
            # Match LUFS: measure target, apply to src
            meter = pyln.Meter(self.sample_rate)
            target_loudness = meter.integrated_loudness(target)
            src = pyln.normalize.loudness(src, meter.integrated_loudness(src), target_loudness)
            src = prevent_clipping(src)
            target_path = os.path.join(self.output_folder, f"{idx:04d}_target.wav")
            src_path = os.path.join(self.output_folder, f"{idx:04d}_src.wav")
            sf.write(target_path, target, self.sample_rate)
            sf.write(src_path, src, self.sample_rate)
            print(f"Wrote: {target_path} and {src_path}")
        print("Dataset generation complete.")
        # Split into train and validation sets and write .scp files with full paths
        val_pct = getattr(self, "validation_data_percentage", 0.3)
        val_count = int(len(all_chunks) * val_pct)
        indices = list(range(len(all_chunks)))
        random.shuffle(indices)
        train_indices = set(indices[val_count:])

        train_scp_path = os.path.join(self.output_folder, "train.scp")
        validate_scp_path = os.path.join(self.output_folder, "validate.scp")

        def full_path(idx, suffix):
            return os.path.abspath(os.path.join(self.output_folder, f"{idx:04d}_{suffix}.wav"))

        with open(train_scp_path, "w") as train_f, open(validate_scp_path, "w") as val_f:
            for idx in range(len(all_chunks)):
                src = full_path(idx, "src")
                target = full_path(idx, "target")
                line = f"{src} {target}\n"
                if idx in train_indices:
                    train_f.write(line)
                else:
                    val_f.write(line)
        print(f"Train SCP written to {train_scp_path}")
        print(f"Validation SCP written to {validate_scp_path}")


if __name__ == "__main__":
    print("Starting dataset generation...")
    generator = DatasetGenerator("config.yaml")
    generator.generate()
