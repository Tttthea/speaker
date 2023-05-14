import os

import pandas as pd
from pyannote.audio import Pipeline
from pydub import AudioSegment
import time

def read_path(path):
    """read files as a list under path"""
    return os.listdir(path)

def pipeDiarization(path):
    """access pyannote pipeline"""
    start_time = time.time()
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                        use_auth_token="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")
    diarization = pipeline(path)
    end_time = time.time()
    print(f'using {end_time - start_time:{.4}}s generating pipeline')
    return diarization

def to_wav(path):
    """convert other audio forms to .wav"""
    start_time = time.time()
    if path.endswith(".m4a") or path.endswith(".mp3") or path.endswith(".webm"):
        converted = os.path.splitext(os.path.basename(path))[0] + ".wav"
        AudioSegment.from_file(path).export(converted, format="wav")
    end_time = time.time()
    print(f'using {end_time-start_time:{.4}}s')
    print(f'converted wav file: {converted}')
    return converted

def out(start, end, speaker, predicted, clip_path):
    d = {}
    if predicted == 1:
        print(f'[{start}: {end}], {speaker}: male, {clip_path}')
        d = {'start': start, 'end': end, 'speaker': speaker, 'gender': 'male', 'path': clip_path}
    elif predicted == 0:
        print(f'[{start}: {end}], {speaker}: female, {clip_path}')
        d = {'start': start, 'end': end, 'speaker': speaker, 'gender': 'female', 'path': clip_path}
    return d