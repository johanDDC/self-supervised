import os

import numpy as np
import pandas as pd
import torchaudio
import torchaudio.transforms as T


def resample_data(input_dir, output_dir, new_freq):
    os.makedirs(output_dir, exist_ok=True)

    speaker_ids = os.listdir(input_dir)
    speaker_ids.remove("audioMNIST_meta.txt")
    speaker_ids = np.array(speaker_ids)

    for idx in speaker_ids:
        speaker_path = os.path.join(input_dir, idx)
        out_speaker_path = os.path.join(output_dir, idx)
        speaker_records = os.listdir(speaker_path)
        os.makedirs(out_speaker_path, exist_ok=True)
        for rec in speaker_records:
            waveform, sample_rate = torchaudio.load(
                os.path.join(speaker_path, rec), normalize=True
            )
            resample = T.Resample(orig_freq=sample_rate, new_freq=new_freq)
            resampled_waveform = resample(waveform)
            torchaudio.save(os.path.join(out_speaker_path, rec), resampled_waveform, new_freq)


def form_df(speakers_ids, dir):
    paths = []
    speakers = []
    classes = []

    for idx in speakers_ids:
        speaker_records = os.listdir(f"{dir}/{idx}")
        for rec in speaker_records:
            paths.append(f"{dir}/{idx}/{rec}")
            classes.append(rec.split("_")[0])
        speakers.extend([idx] * len(speaker_records))
    df = pd.DataFrame({"paths": paths, "speaker": speakers, "y": classes})
    return df