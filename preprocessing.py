import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import librosa
import glob
import os

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(p=0.5),
])

def augment_and_save(files, category):
    for file in files:
        samples, sample_rate = librosa.load(file, sr=16000)
        augmented_samples = augment(samples=samples, sample_rate=sample_rate)
        new_file_name = f"{os.path.splitext(file)[0]}_augmented.wav"
        sf.write(new_file_name, augmented_samples, sample_rate)

healthy_files = glob.glob(os.path.join("dataset", "Normal", "*"))
laryngocele_files = glob.glob(os.path.join("dataset", "laryngozele", "*"))
vox_senilis_files = glob.glob(os.path.join("dataset", "Vox senilis", "*"))

augment_and_save(healthy_files, "Normal")
augment_and_save(laryngocele_files, "laryngozele")
augment_and_save(vox_senilis_files, "Vox senilis")
