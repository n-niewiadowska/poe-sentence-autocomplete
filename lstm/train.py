from keras.models import Sequential, load_model
from keras.layers import Dense, Input, Bidirectional, LSTM, Dropout
from keras.callbacks import History, ModelCheckpoint


# LSTM model
def create_model(X, y):
	model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.2),
    LSTM(256),
    Dropout(0.2),
    Dense(y.shape[1], activation='softmax')
	])

	# model = load_model("./models/lstm-model-65-3.8402.keras")

	return model

def train_lstm(X, y):
	model = create_model(X, y)
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	checkpoint = ModelCheckpoint("models/lstm-model-{epoch:02d}-{loss:.4f}.keras", monitor='loss', verbose=1, save_best_only=True, mode='min')
	# model.fit(X, y, epochs=66, initial_epoch=65, batch_size=128, callbacks=[checkpoint])
	model.fit(X, y, epochs=66, batch_size=128, callbacks=callbacks_list)
