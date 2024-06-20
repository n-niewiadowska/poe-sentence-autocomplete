from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import wordcloud, nltk

# nltk.download("wordnet")
# nltk.download("stopwords")

# word cloud and top 15 words
def words_statistics(tokens):
  lemmatizer = WordNetLemmatizer()
  stop_words = stopwords.words('english')
  stop_words.extend(["le"])

  words = [word for word in tokens if word not in stop_words]
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
