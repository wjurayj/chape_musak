import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):
    def __init__(self, note_range, window_size):
        super(Model, self).__init__()

        self.window_size = window_size
        # TODO:
        # 1) Define any hyperparameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-8)

        self.note_range = note_range
        self.batch_size = 100
        self.embedding_size = 50
        self.rnn_size = 64
        self.hidden_size = 64

        # 2) Define embeddings, encoder, decoder, and feed forward layers
        self.note_embedding = tf.keras.layers.Embedding(note_range, self.embedding_size)

        self.lstm_1 = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=True, return_state=True)
        self.lstm_2 = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=False, return_state=False)
        # self.dense1 = tf.keras.layers.Dense(units=self.dense1_size, activation=tf.nn.relu, use_bias=True)
        self.dense_1 = tf.keras.layers.Dense(units=self.note_range, activation=tf.nn.relu, use_bias=True)
        self.dense_2 = tf.keras.layers.Dense(units=self.note_range, activation=tf.nn.softmax, use_bias=True)


    def call(self, inputs, initial_state):
        embedding = self.note_embedding(note_range)
        lstm_out_1, last_output, last_state = self.lstm_1(embedding, initial_state)
        #not 100% sure abt the inital state stuff/if it even makes sense to pass between LSTM layers
        lstm_out_2 = self.lstm_2(lstm_out_1) #, initial_sate=(last_output, last_state))

        dense_out_1 = self.dense_1(lstm_out_2)
        notes = self.dense_2(dense_out_1)
        return notes


    def loss(self, logits, labels):
        """so the labels here are probably just the next note, right?
        """
        l = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=logits))
        return l



def train(model, train_inputs, train_labels):
    w = model.window_size
    #assumes offset of 1 between inputs and labels--potentially not the move?
    inputs = [train_inputs[i:i+w] for i in range(0, len(train_inputs)-w, w)]
    labels = [train_labels[i:i+w] for i in range(0, len(train_labels)-w, w)]

    n = len(train_inputs)

    #indices = tf.random.shuffle(list(range(n)))
    #train_labels = tf.gather(train_labels, indices, axis=0)

    num_batches = n // model.batch_size

    for i in range(num_batches):
        inputs = train_inputs[i*model.batch_size:(i+1)*model.batch_size]
        labels = train_labels[i*model.batch_size:(i+1)*model.batch_size]

        with tf.GradientTape() as tape:
            probs = model.call(inputs, None)[0]
            loss = model.loss(probs, labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # if i % (num_batches // 10) == 0:
        #     print('batch accuracy:', test(model, inputs, labels))

def test(model, test_inputs, test_labels):
    loss = model.loss(model.call(test_inputs, None),test_labels)
    return tf.math.exp(loss).numpy()

def main():
    model = Model(100, 20)
    #need some data for the model, I'll work on this tm
    pass


if __name__ == '__main__':
    #parameterize this func to handle multiple tracks.
    main()
