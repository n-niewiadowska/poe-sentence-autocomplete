from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.layers import LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History, ModelCheckpoint, EarlyStopping


# training the LSTM model
def train_lstm(X, y):
	filepath="models/lstm-model-{epoch:02d}-{loss:.4f}.keras"
	history = History()
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [history, checkpoint]

	model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.2),
    LSTM(256),
    Dropout(0.2),
    Dense(y.shape[1], activation='softmax')
	])
	# model = Sequential()
	# model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
	# model.add(Dropout(0.2))
	# model.add(Dense(y.shape[1], activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.fit(X, y, epochs=50, batch_size=128, callbacks=callbacks_list)
  
	return model, history