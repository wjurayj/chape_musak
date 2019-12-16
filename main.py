import tensorflow as tf
import numpy as np
from preprocess import get_data
import pypianoroll
import os
from matplotlib import pyplot as plt

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.window_size = 15

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.99, epsilon=1e-8)

        self.note_range = 128

        self.batch_size = 64

        self.rnn_size = 128
        self.hidden_size = 64


        #Stacked LSTM layers
        self.lstm_1 = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=True, return_state=True, dtype=tf.float32)
        self.lstm_2 = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=True, return_state=True)

        self.dense_1 = tf.keras.layers.Dense(units=self.note_range, activation=tf.nn.relu, use_bias=True)
        self.dense_2 = tf.keras.layers.Dense(units=self.note_range, activation=tf.nn.sigmoid, use_bias=False)


    def call(self, inputs, initial_state):

        inputs = tf.cast(inputs, tf.float32)

        lstm_out_1, last_output_1, last_state_1 = self.lstm_1(inputs, initial_state)
        lstm_out_2, last_output_2, last_state_2 = self.lstm_2(inputs, (last_output_1, last_state_1))

        dense_out_1 = self.dense_1(tf.concat([lstm_out_1, lstm_out_2], axis=2))

        notes = self.dense_2(dense_out_1)

        return notes, (last_output_2, last_state_2)


    def loss(self, logits, labels):
        """so the labels here are probably just the next note, right?
        """

        l = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, logits))
        return l


def train(model, train_inputs, train_labels):
    w = model.window_size

    n = len(train_inputs)
    print("n", n)
    num_batches = n // model.batch_size

    print("num batches", num_batches)
    indices = tf.random.shuffle(np.arange(n))
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)


    perps = []
    batch_losses = []

    for i in range(num_batches):

        losses = []

        inputs = train_inputs[i*model.batch_size:(i+1)*model.batch_size]
        labels = train_labels[i*model.batch_size:(i+1)*model.batch_size]

        with tf.GradientTape() as tape:
            probs = model.call(inputs, None)[0]
            loss = model.loss(probs, labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        losses.append(loss)

        if i % (num_batches // 15) == 0:
            print("probs shape", probs.shape)
            p = test(model, inputs, labels)
            print('batch perplexity:', p)
            perps.append(p)
            batch_loss = np.sum(losses)/len(losses)
            print("batch loss", batch_loss)
            batch_losses.append(batch_loss)

    return perps, batch_losses

def test(model, test_inputs, test_labels):
    loss = model.loss(model.call(test_inputs, None)[0],test_labels)
    return tf.math.exp(loss).numpy()


def make_musak(model, starting_notes, length):
    print("composing...")

    previous_state = None
    previous_state_u = None

    #song composed by taking just the argmax of predited notes at each time setp
    song = np.asarray(starting_notes)
    #song composed by including all notes with sigmoid over a certain threshold
    song_u = np.asarray(starting_notes)

    note = song[-1]
    note_u = song[-1]

    threshold = 0.8
    volume = 60

    print('length', length)
    for i in range(length+1):
        logits, previous_state = model.call(np.expand_dims(song[-50: -1], axis=0), previous_state)
        logits_u, previous_state_u = model.call(np.expand_dims(song_u[-50: -1], axis=0), previous_state_u)

        choice = np.argmax(logits[-1][0].numpy())
        note = np.zeros(model.note_range)
        note[choice] = volume

        note_u = np.where(logits_u[-1][0] > threshold, volume, 0)

        song = np.append(song, [note], axis = 0)
        song_u = np.append(song_u, [note_u], axis = 0)
    song = song[-len(starting_notes):-1]
    song_u = song_u[-len(starting_notes):-1]

    print("song", song.shape)
    print(song)
    t = pypianoroll.Track(song)
    multi_1 = pypianoroll.Multitrack(name="song", tracks = [t], beat_resolution=4)
    multi_1.write('./song.mid')

    t = pypianoroll.Track(song_u)
    multi_2 = pypianoroll.Multitrack(name="song", tracks = [t], beat_resolution=4)
    multi_2.write('./song_u.mid')

    fig, ax = multi_1.plot()
    plt.show()

    fig, ax = multi_2.plot()
    plt.show()


def main():
    data = get_data("clean_midi/classical")
    train_data = data

    model = Model()

    train_inputs =[]
    train_labels = []
    for track in train_data:
        for i in range(0, len(track)-model.window_size, model.window_size):
            train_inputs.append(track[i:i+model.window_size])
            train_labels.append(track[i+1:i+1+model.window_size])

    train_inputs = tf.convert_to_tensor(np.asarray(train_inputs))
    train_labels = tf.convert_to_tensor(np.asarray(train_labels))

    perps = []
    losses = []


    print("training...")
    num_epochs = 10

    for ep in range(num_epochs):
        p_arr, l_arr = train(model, train_inputs, train_labels)

        print("EPOCH", ep)
        print("ave p: ", sum(p_arr)/len(p_arr))
        print("ave loss: ", sum(l_arr)/len(l_arr))
        perps += p_arr
        losses += l_arr

    print("ave p:", sum(perps)/len(perps))
    print("ave loss:", sum(losses)/len(losses))

    x = np.arange(len(perps))
    y = np.asarray(perps)
    plt.plot(x,y)
    plt.show()

    x = np.arange(len(losses))
    y = np.asarray(losses)
    plt.plot(x,y)
    plt.show()

    output_length = 300
    starting = train_data[0][-50:-1]

    print("starting", starting)
    make_musak(model, starting, output_length)


if __name__ == '__main__':
    main()
