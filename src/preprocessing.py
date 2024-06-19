import os, string
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.utils import to_categorical
import numpy as np

def combine_files_into_one(dataset, output_file):
  # If the file exists and is not empty, we skip this part
  if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    return
    
  with open(output_file, 'w', encoding='utf-8') as outfile:
    for filename in os.listdir(dataset):
      if filename.endswith('.txt'):
        file_path = os.path.join(dataset, filename)
        with open(file_path, 'r', encoding='utf-8') as infile:
          outfile.write(infile.read())
          outfile.write("\n\n")

  print(f"All tales combined into one file.")


def prepare_text(text):
  # words_to_delete = [ "———", "‡", "†", "¶" ]
  # stop_words = stopwords.words('english')
  words_to_delete = ["'s", "“", "’", "(", ")", "———", "‡", "†", "¶"]
  # stop_words.extend(additional_stop_words)
  words_to_delete.extend(list(string.punctuation))

  tokenized_text = wordpunct_tokenize(text)
  tokenized_text = [ token for token in tokenized_text if token not in words_to_delete ]
  tokens = sorted(list(dict.fromkeys(tokenized_text)))
  tok_to_int = dict((c, i) for i, c in enumerate(tokens))
  n_tokens = len(tokenized_text)
  n_token_vocab = len(tokens)
  print("Total Tokens: ", n_tokens)
  print("Unique Tokens (Token Vocab): ", n_token_vocab)

  seq_length = 100
  dataX = []
  dataY = []

  for i in range(0, n_tokens - seq_length, 1):
    seq_in = tokenized_text[i:i + seq_length]
    seq_out = tokenized_text[i + seq_length]
    dataX.append([tok_to_int[tok] for tok in seq_in])
    dataY.append(tok_to_int[seq_out])

  n_patterns = len(dataX)
  X = np.reshape(dataX, (n_patterns, seq_length, 1))
  X = X / float(n_token_vocab)
  y = to_categorical(dataY)

  return tokens, X, y


def preprocessing(tales_file):
  tales = open(tales_file, 'r', encoding='utf-8').read()
  tokens, X, y = prepare_text(tales.lower())

  return tokens, X, y
