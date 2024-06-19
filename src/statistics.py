from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import wordcloud, string, os

# word cloud and top 15 words
def words_statistics(tokens):
  lemmatizer = WordNetLemmatizer()
  # stop_words = stop_words.words('english')
  # additional_stop_words = ["'s", "“", "”", "’"]
  # stop_words.extend(additional_stop_words)
  # stop_words.extend(list(string.punctuation))

  # words = [word for word in words if word not in stop_words]
  lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

  document = " ".join(lemmatized_words)
  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform([document])
  word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))
  sorted_word_freq = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
  top_15_words = sorted_word_freq[:15]

  plt.bar(*zip(*top_15_words))
  plt.xlabel('Words')
  plt.ylabel('Count')
  plt.title('Top 15 most common words')
  plt.xticks(rotation=45)
  plt.savefig('plots/word-count.png')

  word_cloud = wordcloud.WordCloud(width=1000, height=500).generate(document)
  plt.figure(figsize=(15,8))
  plt.imshow(word_cloud)
  plt.axis('off')
  plt.savefig('plots/wordcloud.png')

# learning curves
def plot_learning_curves(history):
  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.plot(history.history['accuracy'], label='Training Accuracy')
  plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.grid(True, linestyle='--', color='grey')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.grid(True, linestyle='--', color='grey')
  plt.legend()

  plt.tight_layout()
  plt.savefig('plots/learning_curves.png')

def get_statistics(history, tokens):
  os.makedirs('../plots', exist_ok=True)
  words_statistics(tokens)
  plot_learning_curves(history)
  
  print("Plots saved in /plots directory.")