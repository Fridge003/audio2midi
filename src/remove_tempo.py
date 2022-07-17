# import muspy
import mido
from tqdm import tqdm
import os
from os.path import join
import sys


def remove_tempo(midi: mido.MidiFile):
    track = midi.tracks[0]
    for msg in list(track):
        if msg.is_meta and msg.type == "set_tempo":
            # print(msg)
            track.remove(msg)


if __name__ == "__main__":
    directory = sys.argv[1]
    print(directory)
    new_dir = join(directory, "lmd_no_tempo")

    list_dpath = []
    for root, dirs, files in os.walk(directory):
        if "lmd_paired" in root or "piano_2t" in root or "lmd_no_tempo" in root:
            continue
        good = False
        for file in files:
            if file.endswith(".mid"):
                good = True
                break
        if good:
            # print(root)
            # assert len(root) == 26
            list_dpath.append(root)
    # print(list_dpath)
    print(len(list_dpath))

    for dpath in tqdm(list_dpath):
        new_dpath = join(new_dir, dpath.replace(directory + "/", ""))
        if os.path.exists(new_dpath):
            continue
        os.makedirs(new_dpath)
        for fname in tqdm(os.listdir(dpath)):
            fpath = join(dpath, fname)
            new_fpath = join(new_dpath, fname)
            if os.path.exists(new_fpath):
                continue
            try:
                midi = mido.MidiFile(fpath)
            except BaseException:
                # print("read ehh")
                continue
            remove_tempo(midi)
            try:
                midi.save(new_fpath)
            except BaseException:
                # print(f"{fpath} {err}")
                continue
