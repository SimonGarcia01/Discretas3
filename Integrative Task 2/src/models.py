#Here we replicate the models from the different ipynb files
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout, LSTM, Activation, SimpleRNN
from tensorflow.keras.optimizers import Adam, RMSprop

def build_models(max_words=4000, input_length=44, embedding_dim=50):

    # DNN model
    dnn_model = Sequential([
        Input(shape=(input_length,)),
        #Added the embedding layer like i promised hehe
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=input_length),
        Flatten(),
        Dense(256, activation='tanh'),
        Dense(128, activation='tanh'),
        Dense(64, activation='tanh'),
        Dense(32, activation='tanh'),
        Dense(1, activation='sigmoid')
    ])
    #This was the best optimizer we found for the DNN
    optimizer = RMSprop(learning_rate=0.0005)

    #Compile the DNN model 
    dnn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Here make the rnnn model
    rnn_model = Sequential([
        Input(shape=[input_length]),
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=input_length),
        SimpleRNN(1),
        Activation('sigmoid'),
        Dense(1),
    ])


    #Compile it
    rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Here make the lstm model
    lstm_model = Sequential([
        Input(shape=(input_length,)),
        # LSTM embedding requires 128 dimensions, this based on research
        Embedding(input_dim=max_words, output_dim=128, input_length=input_length),
        LSTM(32),
        Dense(32),
        Activation('tanh'),
        Dropout(0.3),
        Dense(1),
        Activation('sigmoid')
    ])

    #This was the best optimizer we found for the LSTM
    opt = RMSprop(learning_rate=0.0005)

    #Compile it
    lstm_model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    # Compile all the models
    models = [dnn_model, rnn_model, lstm_model]

    return dnn_model, rnn_model, lstm_model