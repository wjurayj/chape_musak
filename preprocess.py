import tensorflow as tf
import numpy as np
import pypianoroll
import os


def get_data(train_dir):
    f = open("ex_tracks", "w")

    train_rolls = []
    for filename in os.listdir(train_dir):
        if filename.endswith('.mid'):
            multi = pypianoroll.Multitrack(os.path.join(train_dir, filename), name=filename[:-4])
            for track in multi.tracks:
                nonzeros = np.nonzero(tf.math.reduce_sum(track.pianoroll, axis=1))[0]
                roll = track.pianoroll[nonzeros[0]:nonzeros[-1],0:128]
                roll = np.where(roll > 0, 1, 0)
                train_rolls.append(roll)
                f.write(str(roll))
    f.close()
    return train_rolls


def main():
    """
    this is just to test the shape of the data, and shouldn't ever be actually run
    """
    hey_joe = "clean_midi/Jimi Hendrix/Hey Joe.mid"
    bach_sinfonia = "clean_midi/Bach Johann Sebastian/Sinfonia.mid"
    fp = bach_sinfonia
    multitrack = pypianoroll.Multitrack(fp)
    print(multitrack)
    for track in multitrack.tracks:
        print(track.name, 'length:', len(track.pianoroll), 'num_notes:', np.sum(track.pianoroll))
    pass


if __name__ == '__main__':
    main()
