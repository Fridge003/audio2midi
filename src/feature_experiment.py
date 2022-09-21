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
from midi_to_pr import midi_to_pr_with_tempo, tensors_to_piano_tree, tensor_to_score
from arrange_texture import phrase_arrange, Phrasing


PARAM_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                           '..', 'params'))
DATA_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data'))
TRANSCRIBER_PATH = os.path.join(PARAM_PATH, 'pretrained_onsets_and_frames.pt')
SONG_PATH = os.path.join(DATA_PATH, 'audio_stretched')
TEST_AUDIO_PATH = os.path.join(DATA_PATH, 'input_audio')
ANALYSIS_PATH = os.path.join(DATA_PATH, 'analysis')
CLASSICAL_MIDI_PATH = os.path.join(DATA_PATH, 'classical_midi')
INFERENCE_OUT_PATH = os.path.join(DATA_PATH, 'inference_out')
MUSIC_FORM_PATH = os.path.join(DATA_PATH, 'music_form')
input_analysis_npy_path = None

inference_midi_path = os.path.join(INFERENCE_OUT_PATH, 'classical_inf.mid')
tempo_removed_path =  os.path.join(INFERENCE_OUT_PATH, 'untempoed.mid')
score_path = os.path.join(INFERENCE_OUT_PATH, 'score.mid')
output_path = os.path.join(INFERENCE_OUT_PATH, 'output.mid')
unaligned_score_path = os.path.join(INFERENCE_OUT_PATH, 'score_unaligned.mid')

classical_input = 'Pathetique'
audio_input = 'demo'
audio_input_alt = '001'
model_stage = 2
MODEL_PATH = os.path.join(PARAM_PATH, 'a2s-stage3a.pt')
restrict_ratio = None
texture_input_path = os.path.join(CLASSICAL_MIDI_PATH, classical_input + '.mid')
classical_music_form_path = os.path.join(MUSIC_FORM_PATH, classical_input + '.pkl')

save_analysis_npy_path = os.path.join(ANALYSIS_PATH, audio_input + '.npy')
analysis_alt_path = os.path.join(ANALYSIS_PATH, audio_input_alt + '.npy')
acc_audio_path = os.path.join(TEST_AUDIO_PATH, audio_input + '.wav')
acc_audio_path_alt = os.path.join(TEST_AUDIO_PATH, audio_input_alt + '.wav')
demo_music_form_path = os.path.join(MUSIC_FORM_PATH, audio_input + '.pkl')


model = prepare_model('a2s', stage=model_stage, model_path=MODEL_PATH)
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
audio_tempo = 60.0 / ((analysis[-2][0] - analysis[1][0]) / (analysis.shape[0] - 3))
print(f"Audio tempo = {audio_tempo}")

stretched_song, spb, rates = stretch_a_song(analysis[:, 0], audio)

# segment a song into 2-bar segments (batches)
audios, chords = segment_a_song(analysis, stretched_song, spb)


# only pick the elements with even index in audios and chords, so that segments picked are not overlapped
audios, chords = audios[::2], chords[::2]


# pick another song as an alternate audio input
analysis_alt = np.load(analysis_alt_path)
audio2, _ = librosa.load(acc_audio_path_alt)
stretched_song2, spb2, rates2 = stretch_a_song(analysis_alt[:, 0], audio2)
audios2, _ = segment_a_song(analysis_alt, stretched_song2, spb2)
audios2 = audios2[::2]

sym_src = midi_to_pr_with_tempo(original_midi_path=texture_input_path, output_path=output_path,
                                tempo_removed_path=tempo_removed_path, audio_tempo=audio_tempo, restrict_ratio=restrict_ratio)

sym_src = phrase_arrange(sym_src, classical_music_form_path, demo_music_form_path)
# tensor_to_score(sym_src, unaligned_score_path)
sym_id, sym_src_len = 0, len(sym_src)
batch_starts = np.arange(0, len(audios), batch_size)

audio_option_list = ['alt', 'random', 'zero', None]
features = ['bass', 'int', 'mel', None]
feature_option_list = ['zero', 'one']
audio_len, audio_alt_len = len(audios), len(audios2)
alt_shift = 2
audios_alt = audios2[:audio_len] if audio_alt_len >= audio_len \
                else np.concatenate((audios2, audios2[:audio_len - audio_alt_len]), axis=0)

result_folder = os.path.join(INFERENCE_OUT_PATH, f'try_feature_stage_{model_stage+1}')
if not os.path.exists(result_folder):
    os.mkdir(result_folder)
if os.path.exists(score_path):
    os.remove(score_path)

for feature in features:
    for feature_opt in feature_option_list:
        for audio_opt in audio_option_list:
            #if audio_opt != 'alt':
            #    continue
            predictions = []
            sym_used = []

            # constructing the file path
            filename = 'audio-'
            filename += audio_opt if audio_opt else 'unchanged'
            filename += '__'
            filename += 'feature-'
            if feature:
                filename += f'{feature}_set_to_{feature_opt}.mid'
            else:
                filename += 'unchanged.mid'
            result_midi_path = os.path.join(result_folder, filename)
            print(f'Processing {filename}...')
            for start in batch_starts:  # batching
                audio = audios[start: start + batch_size]
                audio_alt = audios_alt[start: start + batch_size]
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

                audio_alt = torch.from_numpy(audio_alt).float().to(device)
                audio_alt = pitch_shift_to_spec(audio_alt, TGT_SR, 0,
                                            n_fft=N_FFT, hop_length=HOP_LGTH,
                                            mel_scale=mel_scale)

                size_0 = len(audio)
                sym = torch.zeros([size_0, 32, 128])
                for i in range(size_0):
                    sym[i] = sym_src[sym_id]
                    sym_id = (sym_id + 1) % sym_src_len

                sym_used.append(sym)
                pred = model.inference(audio, chord, sym_prompt=sym, audio_opt=audio_opt, feature=feature,
                                       feature_opt=feature_opt, audio_alt=audio_alt)
                predictions.append(pred)


            predictions = np.concatenate(predictions, 0)
            sym_used = torch.cat(sym_used, dim=0)

            score = tensors_to_piano_tree(sym_used)
            # print(score[0], score[1])
            #print(f'prediction shape = {predictions.shape}')

            to_notes_func = lambda x: model.pianotree_dec. \
                grid_to_pr_and_notes(x, 60., 0., False)[1]


            print("Rendering predictions......\n")
            write_prediction(result_midi_path, to_notes_func, analysis,
                             predictions, audio, sr,
                             autoregressive=False, bars_overlapped=False)

            if not os.path.exists(score_path):
                print("Rendering Score......\n")
                write_prediction(score_path, to_notes_func, analysis,
                                 score, audio, sr,
                                 autoregressive=False, bars_overlapped=False)






