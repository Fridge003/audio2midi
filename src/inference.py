import pickle
import sys
import os
import pickle
import pretty_midi as pm
import numpy as np
import torch
import librosa
from torchaudio.transforms import MelScale

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                 '..')))
from src.models import Audio2Symb, Audio2SymbNoChord, Audio2SymbSupervised
from src.dataset import create_data_loaders, AudioMidiDataset, \
    AudioMidiDataLoaders
from src.constants import *
from src.dirs import RESULT_PATH
from src.train import train_model
from src.audio_sp_modules.pitch_shift import pitch_shift_to_spec
from src.utils import chd_to_onehot
from scripts.a2s_config import prepare_model
from pop_style_transfer.a2s_inference import segment_a_song, model_compute, audio_to_symbolic_prediction
from pop_style_transfer.chord_and_beat import analyze_chord_and_beat
from pop_style_transfer.time_stretch_song import stretch_a_song
from pop_style_transfer.perf_render import write_prediction
from midi_to_pr import midi_to_pr_with_tempo, midi_to_pr_without_tempo, tensors_to_piano_tree, tensor_to_score



PARAM_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                           '..', 'params'))
DATA_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data'))
TRANSCRIBER_PATH = os.path.join(PARAM_PATH, 'pretrained_onsets_and_frames.pt')
MODEL_PATH = os.path.join(PARAM_PATH, 'a2s-stage1.pt')
SONG_PATH = os.path.join(DATA_PATH, 'audio_stretched')
input_analysis_npy_path = None
save_analysis_npy_path = os.path.join('..', 'data', 'analysis', 'demo.npy')
# save_analysis_npy_path = os.path.join('..', 'data', 'analysis', 'analysis_pop01.npy')
normal_midi_path = 'normal_inf.mid'
classical_midi_path = 'classical_inf.mid'
# texture_input_path = 'Chopin-raindrop.mid'
texture_input_path = 'Moonlight.mid'
tempo_removed_path = 'untempoed.mid'
score_path = 'score.mid'
output_path = 'output.mid'
acc_audio_path = os.path.join('..', 'test', 'accompaniment_audio', 'demo.wav')
# acc_audio_path = os.path.join('..', 'data', 'audio_stretched', '001.wav')
model = prepare_model('a2s', stage=0, model_path=MODEL_PATH)
device = model.device
batch_size = 32

'''

analysis = analyze_chord_and_beat(acc_audio_path,
                              input_analysis_npy_path,
                                  save_analysis_npy_path)
'''



analysis = np.load(save_analysis_npy_path)
audio, sr = librosa.load(acc_audio_path)

# estimate the tempo

#onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
#audio_tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
audio_tempo = 60.0 / ((analysis[-2][0] - analysis[1][0]) / (analysis.shape[0] - 3))
print(f"Audio tempo = {audio_tempo}")



stretched_song, spb, rates = stretch_a_song(analysis[:, 0], audio)

# segment a song into 2-bar segments (batches)
audios, chords = segment_a_song(analysis, stretched_song, spb)

to_notes_func = lambda x: model.pianotree_dec. \
    grid_to_pr_and_notes(x, 60., 0., False)[1]

sym_src = midi_to_pr_with_tempo(original_midi_path=texture_input_path, output_path=output_path,
                                tempo_removed_path=tempo_removed_path, audio_tempo=audio_tempo)

sym_id, sym_src_len = 0, len(sym_src)

batch_starts = np.arange(0, len(audios), batch_size)
predictions = []
sym_used = []

for start in batch_starts:  # batching
    audio = audios[start: start + batch_size]
    chord = chords[start: start + batch_size]

    # convert chord to 36-d representation
    chord = np.stack([chd_to_onehot(c) for c in chord])
    chord = torch.from_numpy(chord).float().to(device)

    # convert audio to log mel-spectrogram.
    audio = torch.from_numpy(audio).float().to(device)

    mel_scale = MelScale(n_mels=N_MELS, sample_rate=INPUT_SR, f_min=F_MIN,
                         f_max=F_MAX, n_stft=N_FFT // 2 + 1).to(device)
    audio = pitch_shift_to_spec(audio, TGT_SR, 0,
                                n_fft=N_FFT, hop_length=HOP_LGTH,
                                mel_scale=mel_scale)

    size_0 = len(audio)
    sym = torch.zeros([size_0, 32, 128])
    for i in range(size_0):
        sym[i] = sym_src[sym_id]
        sym_id = (sym_id + 1) % sym_src_len

    sym_used.append(sym)
    pred = model.inference(audio, chord, sym_prompt=sym)
    predictions.append(pred)


predictions = np.concatenate(predictions, 0)
sym_used = torch.cat(sym_used, dim=0)
score = tensors_to_piano_tree(sym_used)
# print(score[0], score[1])
print(f'prediction shape = {predictions.shape}')


print("Rendering predictions:")
write_prediction(classical_midi_path, to_notes_func, analysis,
                   predictions, audio, sr,
                   autoregressive=False)


print("Rendering Score: ")
write_prediction(score_path, to_notes_func, analysis,
                 score, audio, sr,
                 autoregressive=False)






