import sys
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from lstm.preprocessing import tokenize_text, prepare_dataset
from lstm.train import create_model

def prepare_generator(file_path):
	raw_text = open(file_path, 'r', encoding='utf-8').read().lower()
	tokenized_text, tokens = tokenize_text(raw_text)
	_, X, y, tok_to_int, int_to_tok, dataX, n_token_vocab = prepare_dataset(tokenized_text, tokens, 50)

	model = create_model(X, y)
	model.load_weights("./models/lstm-model-64-3.8821.keras")
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	return model, tok_to_int, int_to_tok, dataX, n_token_vocab

def generate_random_prompt(dataX, int_to_tok):
	start = np.random.randint(0, len(dataX)-1)
	pattern = dataX[start]
	print("Prompt:")
	print("\"", ' '.join([int_to_tok[value] for value in pattern]), "\"")

	return pattern

def generate_text_for_prompt(model, pattern, int_to_tok, n_token_vocab):
	print("Generated text:")
	print("\"")
	for i in range(60):
		x = np.reshape(pattern, (1, len(pattern), 1))
		x = x / float(n_token_vocab)
		prediction = model.predict(x, verbose=0)
		index = np.argmax(prediction)
		result = int_to_tok[index]
		seq_in = [int_to_tok[value] for value in pattern]
		sys.stdout.write(result+" ")
		pattern.append(index)
		pattern = pattern[1:len(pattern)]
	print("\n\"")

def run_app_lstm(file_path):
  model, tok_to_int, int_to_tok, dataX, n_token_vocab = prepare_generator(file_path)

  print("Enter 'exit' to close the app.")

  while True:
    option = int(input("Do you want to enter your own prompt (1) or to generate one (2)?   "))
    if option == 1:
      prompt = input("Enter your prompt here:   ")
			
      if prompt.lower() == "exit":
        print("Closing the app...")
        break

      prompt = [tok_to_int[word] for word in wordpunct_tokenize(prompt.lower()) if word in tok_to_int]
    else:
      prompt = generate_random_prompt(dataX, int_to_tok)
	
    generate_text_for_prompt(model, prompt, int_to_tok, n_token_vocab)
