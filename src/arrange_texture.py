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
revo_music_form_path = os.path.join(MUSIC_FORM_PATH, 'Revolutionary.pkl')
output_path =  os.path.join(INFERENCE_OUT_PATH, 'output.mid')
tempo_removed_path =  os.path.join(INFERENCE_OUT_PATH, 'untempoed.mid')

# TODO: 1. Design an algorithm that prefers neighboring phrases when picking reference
#  2. Write a script that takes in a string representing a music form, converts it into a Phrasing instance,
#  and saves it under the input path.


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


    # Given two phrases: p1, p2, this function counts how many times does
    # p2 directly follow p1 along time axis
    # e.g.: if the phrase_timeline of this song is [A, B, C, A, B]
    # then score('A','B') will be 2, since 'B' comes after 'A' twice.
    # Similarly, score('B', 'C') = 1, score('A', 'C') = 0
    def consecutive_score(self, p1, p2):
        p_num = len(self.phrase_timeline)
        cnt = 0
        for i in range(p_num - 1):
            if self.phrase_timeline[i][0] == p1 and self.phrase_timeline[i + 1][0] == p2:
                cnt += 1
        return cnt

    # Given a phrase p, return all the phrases that has been prior to p in timeline, in the form of a dictionary
    def find_priors(self, p):
        if not p in self.phrase_dict.keys():
            return None
        p_num = len(self.phrase_timeline)
        prior_dict = {}
        for i in range(1, p_num):
            if self.phrase_timeline[i][0] == p:
                prior = self.phrase_timeline[i-1][0]
                if prior not in prior_dict.keys():
                    prior_dict[prior] = 1
                else:
                    prior_dict[prior] += 1
        return prior_dict

    @staticmethod
    def construct_phrase_mapping(audio_phrase, texture_phrase):
        texture_phrase_dict = texture_phrase.phrase_dict
        audio_phrase_dict = audio_phrase.phrase_dict

        matching_result = {}
        phrase_to_be_matched = audio_phrase_dict.keys()
        classical_phrase_for_choice = texture_phrase_dict.keys()
        visited_ref = set() # store the phrases that have been used in classical input

        # search for the appropriate texture phrase in a heuristic way
        for query in phrase_to_be_matched:
            print(f"Query = {query}:")
            max_score, best_match = -1, 'None'
            for ref in classical_phrase_for_choice:
                matching_score = Phrasing.heuritic_matching_score(query, ref, audio_phrase_dict[query],
                                                                  texture_phrase_dict[ref])
                if ref not in visited_ref: # prefer the phrase that hasn't been used
                    matching_score += 1

                # Computing the "consecutive bonus": if two phrases are consecutive along the timeline
                prior_dict = audio_phrase.find_priors(query)
                for prior_query, cnt in prior_dict.items():
                    if prior_query not in matching_result.keys():
                        continue
                    prior_ref = matching_result[prior_query]
                    matching_score += cnt * min(cnt, texture_phrase.consecutive_score(prior_ref, ref))
                print(f"ref {ref}: score={matching_score}")
                if matching_score > max_score:
                    max_score, best_match = matching_score, ref
            print(f"Finally pick {best_match} \n\n")
            matching_result[query] = best_match
            visited_ref.add(best_match)

        return matching_result


    @staticmethod
    def is_intro(phrase):
        return phrase[0] == 'i'

    @staticmethod
    def is_outro(phrase):
        return phrase[0] == 'o'

    @staticmethod
    def is_transitional(phrase):
        return phrase.upper()[0] == 'X'

    # whether a phrase is a theme, or a structural phrase such as intro, outro or transitional segments
    @staticmethod
    def is_theme(phrase):
        return not (phrase == 'intro' or phrase == 'outro' or phrase.upper()[0] == 'X')

    @staticmethod
    def heuritic_matching_score(query, ref, audio_phrase_info, texture_phrase_info):

        score = 0
        # prefer phrases with same or larger length
        if audio_phrase_info['length'] == texture_phrase_info['length']:
            score += 1
        elif audio_phrase_info['length'] < texture_phrase_info['length']:
            score += 0.5
        else:
            score -= 100 # in any case, the reference phrase should be no shorter than query phrase

        # prefer the phrase that has repeated for the same times
        if len(audio_phrase_info['locations']) == len(texture_phrase_info['locations']):
            score += 1

        # if query and reference are the same type of phrase, the reference will be strongly preferred
        if Phrasing.is_intro(query) and Phrasing.is_intro(ref):
            score += 3
        if Phrasing.is_outro(query) and Phrasing.is_outro(ref):
            score += 3
        if Phrasing.is_transitional(query) and Phrasing.is_transitional(ref):
            score += 3
        if Phrasing.is_intro(query) and Phrasing.is_transitional(ref):
            score += 1
        if Phrasing.is_outro(query) and Phrasing.is_transitional(ref):
            score += 1
        if Phrasing.is_theme(query) and Phrasing.is_theme(ref):
            score += 3

        return score




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
    print(audio_stucture.phrase_timeline, texture_structure.phrase_timeline)
    mapping = Phrasing.construct_phrase_mapping(audio_stucture, texture_structure)
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

    demo_phrase = [['intro', 4, 0], ['A', 8, 0], ['B', 8, 0], ['C', 8, 0], ['X1', 2, 0],
                   ['A', 8, 1], ['B', 8, 1], ['C', 8, 1], ['X2', 8, 0], ['D', 6, 0], ['C', 8, 2], ['outro', 2, 0]]


    pat_phrase = {'A':{'length':8, 'locations':[[0, 8], [16, 24], [56, 64]]},
                  'B':{'length':8, 'locations':[[8, 16], [24, 32], [64, 72]]},
                  'C':{'length':8, 'locations':[[32, 40]]},
                  'D':{'length':4, 'locations':[[40, 44]]},
                  'X1':{'length':8, 'locations':[[44, 52]]},
                  'X2':{'length':4, 'locations':[[52, 56]]}}

    revo_phrase = [['intro', 8, 0], ['X', 2, 0], ['A', 8, 0], ['X', 2, 1], ['B', 8, 0], ['C', 8, 0], ['D', 4, 0],
                   ['intro', 8, 1], ['X', 2, 2], ['A', 8, 1], ['X', 2, 3], ['B', 8, 1], ['E', 8, 0], ['outro', 8, 0]]



    walk_phrase = [['intro', 8, 0], ['A1', 8, 0], ['A2', 8, 0], ['A3', 8, 0],
                       ['B', 8, 0], ['A1', 8, 1], ['A2', 8, 1], ['A3', 8, 1], ['X', 8, 0], ['outro', 4, 0]]

    with open(os.path.join(MUSIC_FORM_PATH, 'walk_tonight.pkl'), 'wb') as f:
         p = Phrasing(walk_phrase)
         pickle.dump(p, f)
