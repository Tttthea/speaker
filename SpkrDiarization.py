import os
import shutil
import time

from pydub import AudioSegment
from helper import pipeDiarization

def spkrTimestamp(path):
    '''
    get list of audio clip timestamp for each speaker
    :param path: path of the original audio
    :return: 'Speaker_index':[[start_1, end_2],[start_2, end_2]...]
    '''

    diarization = pipeDiarization(path)
    speakers = []
    tp_list = []
    sp_tp_list = dict()
    start_time = time.time()
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        tp = (speaker, [round(turn.start, 2), round(turn.end, 2)])
        tp_list.append(tp)
        speakers.append(speaker)
    end_time = time.time()
    print(f'using {end_time - start_time:{.4}}s iterating')
    start_time = time.time()

    for speaker in set(speakers):
        tp = []
        for e in tp_list:
            if e[0] == speaker:
                tp.append(e[1])
        sp_tp_list[speaker] = tp
    end_time = time.time()
    print(f'using {end_time - start_time:{.4}}s put into speaker list')
    print(sp_tp_list)
    return sp_tp_list

def exportAudio(path):
    '''
    generate divided audio clips
    :param path: path of the original audio
    :return: folder of the divided audio clips
    '''

    test_dir = f'{path}'.split('.wav')[0] + '_sliced/'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    sp_tp_list = spkrTimestamp(path)
    start_time = time.time()
    for index, speaker in enumerate(sp_tp_list):
        sec = sp_tp_list[speaker]
        slice_dir = test_dir + f'{speaker}_'
        for i, val in enumerate(sec):
            t1 = sec[i][0] * 1000
            t2 = sec[i][1] * 1000
            aud_outfn = slice_dir + str(t1).split('.')[0] + '_' + str(t2).split('.')[0] + '_audio.wav'
            new = AudioSegment.from_wav(path)
            new = new[t1:t2]
            new.export(aud_outfn, format='wav')
    end_time = time.time()
    print(f'exporting using {end_time - start_time:{.4}}s')
    return test_dir


