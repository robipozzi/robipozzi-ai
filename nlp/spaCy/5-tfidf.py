import spacy
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("############################")
print("########## TF-IDF ##########")
print("############################")
print("")

print("This time around, rather than using a short toy corpus, let's use a larger dataset. scikit-learn has a **datasets** module with utilties to load datasets of our own as well as fetch popular reference datasets online.")
print("https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets")
print("")
print("We'll use the **20 newsgroups** dataset, which is a collection of 18,000 newsgroup posts across 20 topics.")
print("https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset")
print("")
print("List of datasets available:https://scikit-learn.org/stable/datasets.html#datasets")

corpus = fetch_20newsgroups(categories=['sci.space'], remove=('headers', 'footers', 'quotes'))
print("===== Printing corpus type ...")
print(type(corpus))

print("===== Number of posts in our dataset.")
print(len(corpus.data))

print("===== View first two posts.")
print(corpus.data[:2])

print('************************************')
print('***** Creating TF-IDF features *****')
print('************************************')
nlp = spacy.load('en_core_web_sm')

# We don't need named-entity recognition nor dependency parsing for
# this so these components are disabled. This will speed up the
# pipeline. We do need part-of-speech tagging however.
unwanted_pipes = ["ner", "parser"]

# For this exercise, we'll remove punctuation and spaces (which includes newlines), 
# filter for tokens consisting of alphabetic characters, and return the lemma (which require POS tagging).
def spacy_tokenizer(doc):
  with nlp.disable_pipes(*unwanted_pipes):
    return [token.lemma_ for token in nlp(doc) if \
            not token.is_punct and \
            not token.is_space and \
            token.is_alpha]
# Like the classes to create raw frequency and binary bag-of-words vectors, scikit-learn includes a similar class 
# called TfidfVectorizer to create TF-IDF vectors from a corpus.
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

# The usage pattern is similar in that we call fit_transform on the corpus which generates 
# 1. the vocabulary dictionary (fit step) 
# 2. the TF-IDF vectors (transform step).
# Use the default settings of TfidfVectorizer.
vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer)
features = vectorizer.fit_transform(corpus.data)
print("===== Printing the number of unique tokens ...")
print(len(vectorizer.get_feature_names_out()))
print("===== Printing The dimensions of our feature matrix : X rows (documents) by Y columns (tokens) ...")
print(features.shape)
print("===== Printing the first document encoded ...")
print(features[0])
print("")

print('*****************************')
print('***** Querying the data *****')
print('*****************************')
queryString = "lunar orbit"
print("===== Querying corpus for '" + queryString + "'")
print("===== Transform the query into a TF-IDF vector (using TfidfVectorizer with spacy_tokenizer).")
query = [queryString]
query_tfidf = vectorizer.transform(query)
print("Calculate the cosine similarities between the query and each document.")
print("We're calling flatten() here becaue cosine_similarity returns a list")
print("of lists and we just want a single list.")
cosine_similarities = cosine_similarity(features, query_tfidf).flatten()
print("===== Printing Cosine similarities ...")
print(cosine_similarities)
print("Now that we have our list of cosine similarities, we can use this utility function to return the indices of the top k documents \
with the highest cosine similarities.")

# ================================
# ============= TODO =============
# ================================
import numpy as np

# numpy's argsort() method returns a list of *indices* that
# would sort an array:
# https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
#
# The sort is ascending, but we want the largest k cosine_similarites
# at the bottom of the sort. So we negate k, and get the last k
# entries of the indices list in reverse order. There are faster
# ways to do this using things like argpartition but this is
# more succinct.
def top_k(arr, k):
  kth_largest = (k + 1) * -1
  return np.argsort(arr)[:kth_largest:-1]

# So for our query above, these are the top five documents.
top_related_indices = top_k(cosine_similarities, 5)
print(top_related_indices)

# Let's take a look at their respective cosine similarities.
print(cosine_similarities[top_related_indices])

# Top match.
print(corpus.data[top_related_indices[0]])

# Second-best match.
print(corpus.data[top_related_indices[1]])

# Try a different query
query = ["satellite"]
query_tfidf = vectorizer.transform(query)

cosine_similarities = cosine_similarity(features, query_tfidf).flatten()
top_related_indices = top_k(cosine_similarities, 5)

print(top_related_indices)
print(cosine_similarities[top_related_indices])

print(corpus.data[top_related_indices[0]])

