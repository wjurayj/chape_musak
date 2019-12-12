import tensorflow as tf
import numpy as np
from preprocess import get_data
import pypianoroll
import os


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.window_size = 15
        # TODO:
        # 1) Define any hyperparameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-8)

        self.note_range = 128

        # TODO: change back to reasonable batch size when we fix data
        self.batch_size = 50

        # self.embedding_size = 50
        self.rnn_size = 128
        self.hidden_size = 64

        # 2) Define embeddings, encoder, decoder, and feed forward layers
        #self.note_embedding = tf.keras.layers.Embedding(self.note_range, self.embedding_size, input_length=self.window_size)

        self.lstm_1 = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=True, return_state=True, dtype=tf.float32)
        self.lstm_2 = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=True, return_state=True)

        self.dense_1 = tf.keras.layers.Dense(units=self.note_range, activation=tf.nn.relu, use_bias=True)
        self.dense_2 = tf.keras.layers.Dense(units=self.note_range, activation=tf.nn.sigmoid, use_bias=True)


    def call(self, inputs, initial_state):

        inputs = tf.cast(inputs, tf.float32)
        #print("in call, inputs: " + str(inputs.shape))
        #embedding = self.note_embedding(inputs)
        #print('inputs:', inputs.shape)
        #print("embedding:" + str(embedding.shape))
        lstm_out_1, last_output_1, last_state_1 = self.lstm_1(inputs, initial_state)
        #print("lstm out 1", lstm_out_1.shape)
        #not 100% sure abt the inital state stuff/if it even makes sense to pass between LSTM layers
        lstm_out_2, last_output_2, last_state_2 = self.lstm_2(inputs, (last_output_1, last_state_1)) #, initial_sate=(last_output, last_state))
        #print("lstm out 2", lstm_out_2.shape)

        dense_out_1 = self.dense_1(tf.concat([lstm_out_1, lstm_out_2], axis=2))
        #print("dense out 1", dense_out_1.shape)
        #notes = self.dense_2(dense_out_1)
        #print("dense out 2", notes.shape)
        notes = self.dense_2(dense_out_1)

        #print("notes shape", notes.shape)
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
    #indices = tf.random.shuffle(np.arange(n))
    #train_inputs = tf.gather(train_inputs, indices)
    #train_labels = tf.gather(train_labels, indices)

    for i in range(num_batches):
        inputs = train_inputs[i*model.batch_size:(i+1)*model.batch_size]
        labels = train_labels[i*model.batch_size:(i+1)*model.batch_size]
        # inputs = inputs[0]
        # labels = labels[0]
        with tf.GradientTape() as tape:
            probs = model.call(inputs, None)[0]
            loss = model.loss(probs, labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if i % (num_batches // 10) == 0:
            print("probs shape", probs.shape)
            print('batch perplexity:', test(model, inputs, labels))
            print("loss", loss)

def test(model, test_inputs, test_labels):
    loss = model.loss(model.call(test_inputs, None)[0],test_labels)
    return tf.math.exp(loss).numpy()


def make_musak(model, starting_notes, length):
    #reverse_vocab = {idx:word for word, idx in vocab.items()}
    print("composing...")

    previous_state = None
    previous_state_n = None
    previous_state_u = None

    song = np.asarray(starting_notes)
    print("song shape", song.shape)
    song_n = np.asarray(starting_notes)
    song_u = np.asarray(starting_notes)

    note = song[-1]
    note_n = song[-1]
    note_u = song[-1]
    threshold = 0.8
    mu = 0.85
    sigma = 0.05
    volume = 60

    print('length', length)
    for i in range(length+1):
        #print("note shape", note.shape)
        #print(np.expand_dims(np.expand_dims(note, axis=0), axis=0).shape)
        # expanded = np.expand_dims(np.expand_dims(note, axis=0), axis=0)
        logits, previous_state = model.call(np.expand_dims(song[-model.window_size: -1], axis=0), previous_state)
        # logits_n, previous_state_n = model.call(song_n[np.newaxis,:], previous_state_n)
        logits_u, previous_state_u = model.call(np.expand_dims(song_u[-model.window_size: -1], axis=0), previous_state_u)
        # out_index = np.argmax(np.array(logits[0][0]))

        #print("note 2", note.shape)
        # song.append(out_index)
        # print(out_index)
        print("logits", logits.shape)
        #print(logits[0])

        choice = np.argmax(logits[-1][0].numpy())
        print("max val of logits", np.max(logits[-1][0].numpy()))
        note = np.zeros(model.note_range)
        note[choice] = volume


        # note_n = np.where(logits_n[-1][0].numpy() > np.random.normal(mu, sigma, size=model.note_range), volume, 0)
        note_u = np.where(logits_u[-1][0] > threshold, volume, 0)

        print("note:", np.nonzero(note), "note_u:", np.nonzero(note_u)) #, "note_n:", np.nonzero(note_n))

        #next_input = tf.expand_dims(note, axis=0)
        song = np.append(song, [note], axis = 0)
        # song_n = np.append(song_n, [note_n], axis = 0)
        song_u = np.append(song_u, [note_u], axis = 0)

    #song = np.asarray(song)

    print("song", song.shape)
    print(song)
    t = pypianoroll.Track(song)
    multi = pypianoroll.Multitrack(name="song", tracks = [t])
    #pypianoroll.save("./song.midi", multi)
    multi.write('./song.mid')
    #print(" ".join(song))
    # t = pypianoroll.Track(song_n)
    # multi = pypianoroll.Multitrack(name="song", tracks = [t])
    # #pypianoroll.save("./song.midi", multi)
    # multi.write('./song_n.mid')
    t = pypianoroll.Track(song_u)
    multi = pypianoroll.Multitrack(name="song", tracks = [t])
    #pypianoroll.save("./song.midi", multi)
    multi.write('./song_u.mid')


def main():
    data = get_data("clean_midi/classical")
    model = Model()

    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    train_data = data

    train_inputs =[]
    train_labels = []
    for track in train_data:
        for i in range(0, len(track)-model.window_size, model.window_size):
            train_inputs.append(track[i:i+model.window_size])
            train_labels.append(track[i+1:i+1+model.window_size])

    train_inputs = np.asarray(train_inputs)
    train_labels = np.asarray(train_labels)
    #train_labels = np.where(train_labels > 0, 1, 0)

    train_inputs = tf.convert_to_tensor(train_inputs)
    train_labels = tf.convert_to_tensor(train_labels)
    test = False
    if test:
        print("restoring previous checkpoint instead of training")
        checkpoint.restore(manager.latest_checkpoint)
    else:
        print("training...")
        num_epochs = 1
        for ep in range(num_epochs):
            train(model, train_inputs, train_labels)
        manager.save()

    output_length = 300
    starting = train_data[0][0:model.window_size]
    first_note = np.zeros(model.note_range)
    first_note[64] = 60
    print("starting", starting)
    make_musak(model, starting, output_length)


if __name__ == '__main__':
    #parameterize this func to handle multiple tracks.
    main()
