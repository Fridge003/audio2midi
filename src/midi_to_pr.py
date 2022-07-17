import math

import pretty_midi as pm
import numpy as np
import torch
from mido import MidiFile
from math import ceil

# remove the tempo curve of a midi file, thus setting its tempo to the default 120
def remove_tempo(midi_path, save_path):
    midi = MidiFile(midi_path)
    track = midi.tracks[0]
    for msg in list(track):
        if msg.is_meta and msg.type == "set_tempo":
            # print(msg)
            track.remove(msg)
    midi.save(save_path)



def preprocess_bars(beats, downbeats):
    '''
    Cutting midi into bars by processing the position of downbeats,
    the returned list is in the length of bar numbers, each of its element is a tertiary tuple containing
    downbeat position, number of beats in this bar, and the position of beats in this bar
    '''
    eps = 1e-4
    bars = []
    num_beats, num_downbeats = len(beats), len(downbeats)
    beat_pointer, downbeat_pointer = 0, 0
    while downbeat_pointer < num_downbeats - 1:
        cnt, bar_beat = 0, []
        while beats[beat_pointer] < downbeats[downbeat_pointer + 1] - eps:
            cnt += 1
            bar_beat.append(beats[beat_pointer])
            beat_pointer += 1
        bars.append((downbeats[downbeat_pointer], cnt, bar_beat))
        downbeat_pointer += 1
    bars.append((downbeats[-1], num_beats - beat_pointer, list(beats[beat_pointer:])))
    return bars


def bars_to_segments(bars, terminal_time):
    '''
    Cutting bars into 8-beat segments
    '''
    segments = []
    bar_id, bar_num = 0, len(bars)
    while bar_id < bar_num:
        quota = 8
        segment_start = bars[bar_id][0]
        track_bar = bar_id
        segment_beats = []
        while quota > 0:
            quota -= bars[track_bar][1]
            segment_beats += bars[track_bar][2]
            track_bar += 1
            if track_bar == bar_num:
                break

        if track_bar == bar_num: # reach the end of music, no need to take total 8 beats
            segment_end = terminal_time
            segments.append((segment_start, segment_end, segment_beats))
            break
        elif quota == 0:
            segment_end = bars[track_bar][0]
            bar_id = track_bar
            segments.append((segment_start, segment_end, segment_beats))
        else:
            bar_id += 1
    return segments

def get_midi_tempo(midi_path):
    '''
    input the file path of a midi file
    output the average tempo of the midi file through computing the average value of tempo curve
    '''
    midi = pm.PrettyMIDI(midi_path)
    tempo_onsets, tempo_vals = midi.get_tempo_changes()
    terminal_time = midi.get_end_time()
    tempo_onsets = np.append(tempo_onsets, terminal_time)
    tempo_intervals = np.diff(tempo_onsets) / terminal_time
    return sum(tempo_vals * tempo_intervals)


def get_segment_tempos(midi_path):
    '''
    Given a midi file with tempo curves, cut it into 2-bar segments, and return the
    average tempo of each segment
    '''
    eps = 1e-4
    midi = pm.PrettyMIDI(midi_path)
    terminal_time = midi.get_end_time()
    tempo_onsets, tempo_vals = midi.get_tempo_changes()
    tempo_change_num = len(tempo_onsets)
    tempo_intervals = []

    # first get the affecting intervals of each tempo in the tempo set
    for id in range(tempo_change_num - 1):
        tempo_intervals.append((tempo_onsets[id], tempo_onsets[id + 1]))
    tempo_intervals.append((tempo_onsets[-1], terminal_time)) # the last tempo change goes to the final end


    downbeats = midi.get_downbeats()
    beats = midi.get_beats()
    bars = preprocess_bars(beats=beats, downbeats=downbeats)
    segments = bars_to_segments(bars, terminal_time)

    segment_tempos = []

    tempo_id = 0
    collect_tempo = [tempo_vals[0]]

    for segment in segments:
        seg_start, seg_end, _ = segment
        while tempo_intervals[tempo_id][1] < seg_start - eps:
            tempo_id += 1
        while tempo_intervals[tempo_id][1] < seg_end - eps:
            tempo_id += 1
            collect_tempo.append(tempo_vals[tempo_id])
        segment_tempos.append(sum(collect_tempo) / len(collect_tempo))
        collect_tempo = collect_tempo[:1] # only retain the nearest tempo


    # print(segment_tempos)
    return segment_tempos



def tensors_to_piano_tree(tensors):
    score = []
    for tensor in tensors:
        # each tensor is in size (32, 128),
        # we know want to convert it into pianotree format with size (32, 15, 6)
        segment = []
        pitch_num_count = np.zeros(32, dtype=int)
        pitches = 129 * torch.ones([32, 15, 1])
        durations = torch.zeros([32, 15, 5])
        note_positions = torch.nonzero(tensor) # each nonzero element in tensor represents a note
        for note in note_positions:
            t, pitch = int(note[0]), note[1]
            dur, pitch_id = int(tensor[t][pitch]), pitch_num_count[t]
            pitches[t][pitch_id][0] = pitch
            while dur > 0:
                binary_pointer = 4
                if dur % 2 == 1:
                    durations[t][pitch_id][binary_pointer] = 1
                    dur = (dur - 1) / 2
                else:
                    dur = dur / 2
                binary_pointer -= 1
            pitch_num_count[t] += 1

        pt = torch.cat([pitches, durations], dim=-1)
        score.append(torch.unsqueeze(pt, dim=0))


    score = torch.cat(score, dim=0)
    return score.numpy().astype(int)



def tensor_to_score(tensors, output_path):
    # output the notes in tensors to a midi in a cleaner format
    output = pm.PrettyMIDI(initial_tempo=120)
    velocity = 100 # or any value you like
    inst = pm.Instrument(program=0, is_drum=False, name='piano')
    output.instruments.append(inst)
    for i, tensor in enumerate(tensors):
        segment_start_time = 4 * i
        for step in range(32):
            note_start = segment_start_time + 0.125 * step
            for pitch in range(128):
                if tensor[step][pitch] > 0:
                    dur = tensor[step][pitch].item()
                    inst.notes.append(pm.Note(velocity=velocity, pitch=pitch, start=note_start, \
                                              end=note_start + dur * 0.125))
    output.write(output_path)


def midi_to_pr_original(original_midi_path, output_path, tempo_removed_path):

    eps = 1e-1

    default_tempo = 120
    remove_tempo(original_midi_path, tempo_removed_path)

    midi = pm.PrettyMIDI(tempo_removed_path)

    tensors = []

    # if left hand and right hand are seperated, then track_num = 2, else track_num = 1
    track_num = len(midi.instruments)
    assert track_num <= 2
    two_track = (track_num == 2)

    tempos = get_segment_tempos(original_midi_path)
    downbeats = midi.get_downbeats()
    beats = midi.get_beats()
    terminal_time = midi.get_end_time()


    notes = midi.instruments[0].notes
    notes_iter = iter(notes)
    next_note = next(notes_iter)
    if two_track:
        notes_extra = midi.instruments[1].notes
        notes_extra_iter = iter(notes_extra)
        next_note_extra = next(notes_extra_iter)


    bars = preprocess_bars(beats=beats, downbeats=downbeats)
    segments = bars_to_segments(bars, terminal_time)

    assert len(tempos) == len(segments)
    # extracting notes from 2-bar segments and producing tensors
    for seg_id, segment in enumerate(segments):

        tensor = torch.zeros([32, 128])
        seg_start, seg_end, seg_beats = segment

        # since prior bars might be skipped, keep iterating until the note is later than segment_start
        while(next_note and next_note.start < seg_start - eps):
            next_note = next(notes_iter, None)

        if two_track:
            while(next_note_extra and next_note_extra.start < seg_start - eps):
                next_note_extra = next(notes_extra_iter, None)

        unit_time = 60 / (4 * default_tempo) # taking a quarter of beat as the time unit
        while next_note and next_note.start <= seg_end - eps:
            idx = int(round((next_note.start - seg_start) / unit_time , 0))
            if idx >= 32:
                break
            dur = round((next_note.end - next_note.start) / unit_time , 0)
            tensor[idx][next_note.pitch] = dur
            next_note = next(notes_iter, None)

        if two_track:
            while next_note_extra and next_note_extra.start <= seg_end - eps:
                idx = int(round((next_note_extra.start - seg_start) / unit_time , 0))
                if idx >= 32:
                    break
                dur = round((next_note_extra.end - next_note_extra.start) / unit_time , 0)
                tensor[idx][next_note_extra.pitch] = dur
                next_note_extra = next(notes_extra_iter, None)

        tensors.append(tensor)

    tensor_to_score(tensors, output_path)
    return tensors


def midi_to_pr_without_tempo(original_midi_path, output_path, tempo_removed_path):

    eps = 1e-1

    default_tempo = 120
    remove_tempo(original_midi_path, tempo_removed_path)

    midi = pm.PrettyMIDI(tempo_removed_path)

    tensors = []

    downbeats = midi.get_downbeats()
    beats = midi.get_beats()
    terminal_time = midi.get_end_time()

    note_collector = []
    for inst in midi.instruments:
        track = inst.notes
        for note in track:
            note_collector.append({'start':note.start, 'end':note.end, 'pitch':note.pitch})

    # sort all the notes depending on starting time
    note_collector.sort(key=lambda x:x['start'])
    note_iter = iter(note_collector)
    next_note = next(note_iter, None)

    bars = preprocess_bars(beats=beats, downbeats=downbeats)
    segments = bars_to_segments(bars, terminal_time)

    # extracting notes from 2-bar segments and producing tensors
    for seg_id, segment in enumerate(segments):

        tensor = torch.zeros([32, 128])
        seg_start, seg_end, seg_beats = segment

        # since prior bars might be skipped, keep iterating until the note is later than segment_start
        while(next_note and next_note['start'] < seg_start - eps):
            next_note = next(note_iter, None)

        unit_time = 60 / (4 * default_tempo) # taking a quarter of beat as the time unit
        while next_note and next_note['start'] <= seg_end - eps:
            idx = int(round((next_note['start'] - seg_start) / unit_time , 0))
            if idx >= 32:
                break
            dur = round((next_note['end'] - next_note['start']) / unit_time , 0)
            tensor[idx][next_note['pitch']] = dur
            next_note = next(note_iter, None)

        tensors.append(tensor)

    tensor_to_score(tensors, output_path)
    return tensors

def estimate_ratio(audio_tempo, texture_tempo):
    log_specific_ratio = math.log2(audio_tempo / texture_tempo)
    rounded = round(log_specific_ratio, 0)
    return 2 ** rounded


def stretch_note(note_collector, segments, stretch_ratio=1.0):

    eps = 1e-1
    stretched_notes, stretched_segments = [], []
    note_iter = iter(note_collector)
    next_note = next(note_iter, None)
    stretched_seg_start, stretched_seg_end = 0.0, 0.0

    # distort the timestamps of each note according to the tempo ratio
    for segment in segments:

        seg_start, seg_end, _ = segment
        stretched_seg_end = stretched_seg_start + (seg_end - seg_start) * stretch_ratio

        # keep iterating until the note is later than segment_start
        while(next_note and next_note['start'] < seg_start - eps):
            next_note = next(note_iter, None)

        while next_note and next_note['start'] <= seg_end - eps:
            stretched_note_start = stretched_seg_start +  (next_note['start'] - seg_start) * stretch_ratio
            stretched_note_end = stretched_note_start + (next_note['end'] - next_note['start']) * stretch_ratio
            stretched_notes.append({'start':stretched_note_start,
                                    'end':stretched_note_end,
                                    'pitch':next_note['pitch']})
            next_note = next(note_iter, None)

        stretched_segments.append((stretched_seg_start, stretched_seg_end))
        stretched_seg_start = stretched_seg_end

    # print(stretched_notes, stretched_segments)
    return stretched_notes, stretched_segments





def midi_to_pr_with_tempo(original_midi_path, output_path, tempo_removed_path, audio_tempo):

    eps = 1e-1

    default_tempo = 120
    remove_tempo(original_midi_path, tempo_removed_path)

    midi = pm.PrettyMIDI(tempo_removed_path)

    tensors = []

    texture_tempo = get_midi_tempo(original_midi_path)
    print(f"Input texture tempo = {texture_tempo}")
    # tempos = get_segment_tempos(original_midi_path)
    downbeats = midi.get_downbeats()
    beats = midi.get_beats()
    terminal_time = midi.get_end_time()

    note_collector = []
    for inst in midi.instruments:
        track = inst.notes
        for note in track:
            note_collector.append({'start':note.start, 'end':note.end, 'pitch':note.pitch})

    # sort all the notes depending on starting time
    note_collector.sort(key=lambda x:x['start'])
    bars = preprocess_bars(beats=beats, downbeats=downbeats)
    segments = bars_to_segments(bars, terminal_time)

    # do time-to-time stretching to collected notes
    stretch_ratio = estimate_ratio(audio_tempo, texture_tempo)
    print(f"stretch ratio = {stretch_ratio}")
    # stretch_ratio = 1.0
    stretched_notes, stretched_segments = stretch_note(note_collector, segments, stretch_ratio)
    note_iter = iter(stretched_notes)
    next_note = next(note_iter, None)
    num_of_tensors = int(ceil(stretched_segments[-1][1] / 4.0))
    unit_time = 60 / (4 * default_tempo) # taking a quarter of beat as the time unit


    # each tensor also represents a two-bar segment under bpm=120
    # however, the onsets and durations of notes are distorted, thus unaligned to the beats
    for tensor_id in range(num_of_tensors):

        tensor = torch.zeros([32, 128])
        seg_start, seg_end = 4 * tensor_id, 4 * (tensor_id + 1)

        while next_note and next_note['start'] <= seg_end - eps:
            idx = int(round(float(next_note['start'] - seg_start) / unit_time , 0))
            if idx >= 32:
                break
            # dur = round((next_note['end'] - next_note['start']) / unit_time, 0)
            dur = ceil((next_note['end'] - next_note['start']) / unit_time)
            tensor[idx][next_note['pitch']] = dur
            next_note = next(note_iter, None)

        tensors.append(tensor)
    tensor_to_score(tensors, output_path)
    return tensors


if __name__ == "__main__":
    MIDI_PATH, OUTPUT_PATH, UNTEMPO_PATH = 'Pathetique.mid', 'output.mid', 'untempoed.mid'
    midi_to_pr_with_tempo(MIDI_PATH, OUTPUT_PATH, UNTEMPO_PATH, audio_tempo=62)
