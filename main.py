import os, argparse
from lstm.preprocessing import preprocessing, combine_files_into_one
from lstm.train import train_lstm
from lstm.app import run_app_lstm
from gpt.train import train_gpt
from gpt.app import run_app_gpt
from word_statistics import words_statistics

def main():
  DATASET = "./eapoe-data"
  tales_file = "./poe-tales.txt"
  gpt_model_path = "./models/gpt-model"
  os.makedirs('models', exist_ok=True)
  os.makedirs('plots', exist_ok=True)

  parser = argparse.ArgumentParser(description="Autocomplete Suggestions - Edgar Allan Poe")
  parser.add_argument("-t", "--train", action="store_true", help="Train the model")
  parser.add_argument("-r", "--run", action="store_true", help="Run the app")
  parser.add_argument("-p", "--plot", action="store_true", help="Get the additional statistics")
  args = parser.parse_args()

  print("This is a Sentence Autocomplete Suggestions application based on Edgar Allan Poe's tales.")

  if args.train:
    option = int(input("Would you like to train LSTM model (1) or GPT-2 model (2)?   "))
    if option == 1:
      print("Preparing data...")
      combine_files_into_one(DATASET, tales_file)
      tokens, X, y = preprocessing(tales_file)
      print("Tales prepared.")

      print("Training the model...")
      train_lstm(X, y)
      print("LSTM model is trained and saved in /models directory!")
    else:
      print("Training the pre-trained GPT model...")
      train_gpt(tales_file)
      print("GPT-2 model is trained and saved in /models directory!")
  elif args.run:
    option = int(input("Do you want to use LSTM generator (1) or GPT-2 (2)?   "))
    if option == 1:
      run_app_lstm(tales_file)
    else:
      run_app_gpt(gpt_model_path)
  elif args.plot:
    print("Preparing statistics...")
    combine_files_into_one(DATASET, tales_file)
    tokens, _, _ = preprocessing(tales_file)
    words_statistics(tokens)
    print("You can find the statistics in /plots directory.")
  else:
    print("You can run the program with following options:")
    print("->  -t, --train  - performs data preprocessing, trains the model and shows important statistics")
    print("->  -r, --run    - runs the application that uses this model")


if __name__ == "__main__":
  main()