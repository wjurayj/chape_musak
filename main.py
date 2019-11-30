import tensorflow as tf
import numpy as np
from preprocess import get_data
import pypianoroll


class Model(tf.keras.Model):
    def __init__(self, note_range, window_size):
        super(Model, self).__init__()

        self.window_size = window_size
        # TODO:
        # 1) Define any hyperparameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-8)

        self.note_range = note_range

        # TODO: change back to reasonable batch size when we fix data
        self.batch_size = 1

        self.embedding_size = 50
        self.rnn_size = 64
        self.hidden_size = 64

        # 2) Define embeddings, encoder, decoder, and feed forward layers
        self.note_embedding = tf.keras.layers.Embedding(self.note_range, self.embedding_size, input_length=self.window_size)

        self.lstm_1 = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=True, return_state=True)
        self.lstm_2 = tf.keras.layers.LSTM(units=self.rnn_size, return_sequences=False, return_state=False)

        self.dense_1 = tf.keras.layers.Dense(units=self.note_range, activation=tf.nn.relu, use_bias=True)
        self.dense_2 = tf.keras.layers.Dense(units=self.note_range, activation=tf.nn.softmax, use_bias=True)


    def call(self, inputs, initial_state):
        # replaced input from note_range to inputs -- not sure what note range was?
        embedding = self.note_embedding(inputs)
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
    # i think we can just handle this in main when seperating labels?
    #inputs = [train_inputs[i:i+w] for i in range(0, len(train_inputs)-w, w)]
    #labels = [train_labels[i:i+w] for i in range(0, len(train_labels)-w, w)]

    n = len(train_inputs)
    num_batches = n // model.batch_size

    indices = tf.random.shuffle(np.arange(n))
    #train_inputs = tf.gather(train_inputs, indices)
    #train_labels = tf.gather(train_labels, indices)

    for i in range(num_batches):
        inputs = train_inputs[i*model.batch_size:(i+1)*model.batch_size]
        labels = train_labels[i*model.batch_size:(i+1)*model.batch_size]

        with tf.GradientTape() as tape:
            probs = model.call(inputs, None)[0]
            loss = model.loss(probs, labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if i % (num_batches // 10) == 0:
            print('batch accuracy:', test(model, inputs, labels))

def test(model, test_inputs, test_labels):
    loss = model.loss(model.call(test_inputs, None),test_labels)
    return tf.math.exp(loss).numpy()

def main():
    data = get_data("clean_midi/Bach Johann Sebastian")
    print(len(data))
    print(data[0])


    model = Model(100, 10)

    #TODO: need to seperate into train and test
    train_data = data

    train_inputs =[]
    train_labels = []
    for i in range(0, len(train_data)-model.window_size, model.window_size):
        train_inputs.append(train_data[i:i+model.window_size])
        train_labels.append(train_data[i+1:i+1+model.window_size])

    train_inputs = np.asarray(train_inputs)
    train_labels = np.asarray(train_labels)


    #train_inputs = np.expand_dims(train_inputs, -1)
    #train_labels = np.expand_dims(train_labels, -1)

    print("inputs:")
    print(train_inputs.shape)
    print("labels:")
    print(train_labels.shape)
    print(train_inputs[0:10])
    train_inputs = tf.convert_to_tensor(train_inputs)
    train_labels = tf.convert_to_tensor(train_labels)

    train(model, train_inputs, train_labels)




if __name__ == '__main__':
    #parameterize this func to handle multiple tracks.
    main()
