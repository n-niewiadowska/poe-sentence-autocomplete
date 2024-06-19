# main file that performs everything
import os, argparse
from src.preprocessing import preprocessing, combine_files_into_one
from src.train import train_lstm
from src.statistics import get_statistics

def main():
  DATASET = "./eapoe-data"
  tales_file = "./poe-tales.txt"
  os.makedirs('models', exist_ok=True)

  parser = argparse.ArgumentParser(description="Autocomplete Suggestions - Edgar Allan Poe")
  parser.add_argument("-t", "--train", action="store_true", help="Train the model")
  parser.add_argument("-r", "--run", action="store_true", help="Run the app")
  args = parser.parse_args()

  print("Nevermore...")
  print("This is a Sentence Autocomplete Suggestions application.")
  print("It is based on Edgar Allan Poe's tales.")

  if args.train:
    print("Preparing data...")
    combine_files_into_one(DATASET, tales_file)
    tokens, X, y = preprocessing(tales_file)
    print("Tales prepared.")

    print("Training the model...")
    model, history = train_lstm(X, y)
    get_statistics(history, tokens)
    print("LSTM model is trained and saved in /models directory!")
    print("||   You can find model plots in directory /plots.")
  elif args.run:
    print("App in the making")
  else:
    print("You can run the program with following options:")
    print("->  -t, --train  - performs data preprocessing, trains the model and shows important statistics")
    print("->  -r, --run    - runs the application that uses this model")


if __name__ == "__main__":
  main()