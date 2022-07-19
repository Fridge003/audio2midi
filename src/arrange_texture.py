import numpy as np
import sys
import os
import pickle

import torch

from midi_to_pr import midi_to_pr_with_tempo, tensor_to_score, pick_bar_from_tensors

DATA_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data'))
ANALYSIS_PATH = os.path.join(DATA_PATH, 'analysis')
CLASSICAL_MIDI_PATH = os.path.join(DATA_PATH, 'classical_midi')
INFERENCE_OUT_PATH = os.path.join(DATA_PATH, 'inference_out')
MUSIC_FORM_PATH = os.path.join(DATA_PATH, 'music_form')
save_analysis_npy_path = os.path.join(ANALYSIS_PATH, 'demo.npy')
demo_music_form_path = os.path.join(MUSIC_FORM_PATH, 'demo.pkl')
pathetique_music_form_path = os.path.join(MUSIC_FORM_PATH, 'pathetqiue.pkl')
texture_input_path = os.path.join(CLASSICAL_MIDI_PATH, 'Pathetique.mid')
output_path =  os.path.join(INFERENCE_OUT_PATH, 'output.mid')
tempo_removed_path =  os.path.join(INFERENCE_OUT_PATH, 'untempoed.mid')



class Phrasing():
    '''
    Here, should explain the design of phrase_timeline & phrase_dict
    '''
    def __init__(self, phrases):
        if isinstance(phrases, list):
            # feed in a time series representing the music form
            self.phrase_timeline = phrases
            self.phrase_dict = self._get_phrase_dic()
        elif isinstance(phrases, dict):
            self.phrase_dict = phrases
            self.phrase_timeline = self._get_phrase_timeline()
        else:
            self.phrase_dict = {}
            self.phrase_timeline = []


    def _get_phrase_dic(self):
        dic = {}
        pointer = 0 # calculate the position of phrase during the loop
        for p in self.phrase_timeline:
            phrase, length = p[0], p[1]
            if phrase in dic:
                assert dic[phrase]['length'] == length, "The length of the same phrase must be the same !!!"
                dic[phrase]['locations'].append([pointer, pointer + length])
            else:
                dic[phrase] = {}
                dic[phrase]['length'] = length
                dic[phrase]['locations'] = [[pointer, pointer + length]]
            pointer += length
        return dic

    def _get_phrase_timeline(self):
        collect, lst = [], []
        for k, v in self.phrase_dict.items():
            for id, interval in enumerate(v['locations']):
                collect.append([k, interval, id])
        collect.sort(key=lambda x:x[1][0])
        for phrase in collect:
            lst.append([phrase[0], phrase[1][1]-phrase[1][0], phrase[2]])
        return lst


    @staticmethod
    def construct_phrase_mapping(audio_phrase, texture_phrase):
        texture_phrase_dict = texture_phrase.phrase_dict
        audio_phrase_dict = audio_phrase.phrase_dict

        matching_result = {}
        phrase_to_be_matched = audio_phrase_dict.keys()
        classical_phrase_for_choice = texture_phrase_dict.keys()
        visited = set() # store the phrases that have been used in classical input

        # search for the appropriate texture phrase in a heuristic way
        for query in phrase_to_be_matched:
            max_score, best_match = -1, 'None'
            for key in classical_phrase_for_choice:
                matching_score = Phrasing.heuritic_matching_score(query, key, audio_phrase_dict[query],
                                                                  texture_phrase_dict[key])
                if key not in visited: # prefer the phrase that hasn't been used
                    matching_score += 1
                if matching_score > max_score:
                    max_score, best_match = matching_score, key

            matching_result[query] = best_match
            visited.add(best_match)

        return matching_result

    @staticmethod
    def is_offcut(phrase):
        return phrase == 'intro' or phrase == 'outro' or phrase.upper()[0] == 'X'

    @staticmethod
    def heuritic_matching_score(query, key, audio_phrase_info, texture_phrase_info):

        score = 0
        # prefer phrases with same or larger length
        if audio_phrase_info['length'] == texture_phrase_info['length']:
            score += 1
        elif audio_phrase_info['length'] < texture_phrase_info['length']:
            score += 0.5

        # prefer the phrase that has repeated for the same times
        if len(audio_phrase_info['locations']) == len(texture_phrase_info['locations']):
            score += 1

        if Phrasing.is_offcut(query) and Phrasing.is_offcut(key):
            score += 3

        if (not Phrasing.is_offcut(query)) and (not Phrasing.is_offcut(key)):
            score += 3

        return score








demo_phrase = [['intro', 4, 0], ['A', 8, 0], ['B', 8, 0], ['C', 8, 0], ['X1', 2, 0],
               ['A', 8, 1], ['B', 8, 1], ['C', 8, 1], ['X2', 8, 0], ['D', 6, 0], ['C', 8, 2], ['outro', 2, 0]]


pat_phrase = {'A1':{'length':8, 'locations':[[0, 8], [16, 24], [56, 64]]},
              'A2':{'length':8, 'locations':[[8, 16], [24, 32], [64, 72]]},
              'B1':{'length':8, 'locations':[[32, 40]]},
              'B2':{'length':4, 'locations':[[40, 44]]},
              'X1':{'length':8, 'locations':[[44, 52]]},
              'X2':{'length':4, 'locations':[[52, 56]]}}



def combine_bars(tensors):
    # input a list of 16*128 tensors, each tensor represents a bar of texture input

    if len(tensors) % 2 == 1:
        # if the number of bars is odd, pad the last bar as zeros
        tensors.append(torch.zeros([16, 128]))

    output = []
    seg_num = len(tensors) // 2
    for i in range(seg_num):
        t1, t2 = tensors[2 * i], tensors[2 * i + 1]
        output.append(torch.cat([t1, t2], dim=0))

    return output


# phrase, length, used, theme/transition


def phrase_arrange(tensors, texture_music_form_path, audio_music_form_path):

    with open(audio_music_form_path, 'rb') as f:
        audio_stucture = pickle.load(f)

    with open(texture_music_form_path, 'rb') as f:
        texture_structure = pickle.load(f)

    mapping = Phrasing.construct_phrase_mapping(audio_stucture, texture_structure)
    # mapping = {'intro':'X2', 'A':'A1', 'B':'A2', 'C':'B1', 'X1':'X2', 'X2':'X1', 'D':'X1', 'outro':'B2'}

    arranged_tensors = []
    audio_phrase_timeline = audio_stucture.phrase_timeline
    texture_phrase_dict = texture_structure.phrase_dict

    for phrase, length, idx in audio_phrase_timeline:
        picked_texture = mapping[phrase]
        texture_length =  texture_phrase_dict[picked_texture]['length']
        texture_locations = texture_phrase_dict[picked_texture]['locations']
        picked_location = texture_locations[idx % len(texture_locations)]
        l, r = picked_location[1] - min(length, texture_length), picked_location[1]
        for i in range(l, r):
            arranged_tensors.append(pick_bar_from_tensors(tensors, bar_idx=i))

    return combine_bars(arranged_tensors)




if __name__ == "__main__":

    with open(demo_music_form_path, 'wb') as f:
        p = Phrasing(demo_phrase)
        pickle.dump(p, f)

    with open(pathetique_music_form_path, 'wb') as f:
        p = Phrasing(pat_phrase)
        pickle.dump(p, f)
