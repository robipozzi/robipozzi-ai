import spacy
# Modules for Bag Of Words
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

print("##################################")
print("########## Bag Of Words ##########")
print("##################################")
print("")
print('****************************************')
print('***** Plain frequency Bag Of Words *****')
print('****************************************')
# A corpus of sentences.
corpus = [
  "Red Bull drops hint on F1 engine.",
  "Honda exits F1, leaving F1 partner Red Bull.",
  "Hamilton eyes record eighth F1 title.",
  "Aston Martin announces sponsor."
]
print("===== Printing Corpus ...")
print(corpus)
print("===== Vectorizing Corpus using CountVectorizer fit_transform function  ...")
print("# The fit_transform method does two things:")
print("#   1. It learns a vocabulary dictionary from the corpus.")
print("#   2. It returns a matrix where each row represents a document and each column represents a token (i.e. term).")
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(corpus)

# We can take a look at the features and vocabulary dictionary. 
# Notice the CountVectorizer took care of tokenization for us. It also removed punctuation and lower-cased everything.
print("===== View features (tokens), using CountVectorizer ...")
print(vectorizer.get_feature_names_out())
print("===== View vocabulary dictionary, using CountVectorizer ...")
print(vectorizer.vocabulary_)
print("===== Printing Bag of words from the corpus ...")
print("# If we look at the raw structure, we'll see tuples where the first element represents the document, ")
print("# and the second element represents a token ID. ")
print("# It's then followed by a count of that token. So in the second document (index 1), token 8 (\"f1\") occurs twice.")
print(bow)
print("")

print('*****************************************************')
print('***** Binary Bag Of Words with custom tokenizer *****')
print('*****************************************************')
# Create a tokenizer callback using spaCy under the hood. Here, we tokenize
# the passed-in text and return the tokens, filtering out punctuation.
def spacy_tokenizer(doc):
  return [token.text for token in nlp(doc) if not token.is_punct]

vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, lowercase=False, binary=True)
binaryBow = vectorizer.fit_transform(corpus)
print("===== Printing Binary Bag of words from the corpus ...")
print(binaryBow)

print("===== View features (tokens), using CountVectorizer (with spacy_tokenizer) ...")
print(vectorizer.get_feature_names_out())
print("===== View vocabulary dictionary, using CountVectorizer (with spacy_tokenizer) ...")
print(vectorizer.vocabulary_)
print("")

print('*****************************')
print('***** Cosine Similarity *****')
print('*****************************')
# The cosine method expects array_like inputs, so we need to generate
# arrays from our sparse matrix.
print("===== Reprinting Corpus ...")
print(corpus)
print("===== Cosine similarity on Bag of words from the corpus (using cosine function)")
print("===== The cosine method expects array_like inputs, so we need to generate arrays from bow sparse matrix.")
#print(bow)
doc1_vs_doc2 = 1 - spatial.distance.cosine(bow[0].toarray()[0], bow[1].toarray()[0])
doc1_vs_doc3 = 1 - spatial.distance.cosine(bow[0].toarray()[0], bow[2].toarray()[0])
doc1_vs_doc4 = 1 - spatial.distance.cosine(bow[0].toarray()[0], bow[3].toarray()[0])

print(f"Doc 1 vs Doc 2: {doc1_vs_doc2}")
print(f"Doc 1 vs Doc 3: {doc1_vs_doc3}")
print(f"Doc 1 vs Doc 4: {doc1_vs_doc4}")

# cosine_similarity can take either array-likes or sparse matrices.
print("===== Cosine similarity on Bag of words, using cosine_similarity function (that can take arrays or sparse matrices)")
similarityMatrix = cosine_similarity(bow)
print(similarityMatrix)
print("===== Accessing matrix with indices [0,1] ...")
print(similarityMatrix[0,1])

print("===== Cosine similarity on Binary Bag of words from the corpus (using cosine function)")
print("===== The cosine method expects array_like inputs, so we need to generate arrays from bow sparse matrix.")
#print(binaryBow)
doc1_vs_doc2 = 1 - spatial.distance.cosine(binaryBow[0].toarray()[0], binaryBow[1].toarray()[0])
doc1_vs_doc3 = 1 - spatial.distance.cosine(binaryBow[0].toarray()[0], binaryBow[2].toarray()[0])
doc1_vs_doc4 = 1 - spatial.distance.cosine(binaryBow[0].toarray()[0], binaryBow[3].toarray()[0])

print(f"Doc 1 vs Doc 2: {doc1_vs_doc2}")
print(f"Doc 1 vs Doc 3: {doc1_vs_doc3}")
print(f"Doc 1 vs Doc 4: {doc1_vs_doc4}")

print("===== Cosine similarity on Binary Bag of words, using cosine_similarity function (that can take arrays or sparse matrices)")
binarySimilarityMatrix = cosine_similarity(binaryBow)
print(cosine_similarity(binarySimilarityMatrix))

print("")

print('*******************')
print('***** N-Grams *****')
print('*******************')
print("===== Reprinting Corpus ...")
print(corpus)
print("===== Reprinting Corpus ...")
vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, lowercase=False, binary=True, ngram_range=(1,2))
bigrams = vectorizer.fit_transform(corpus)
print('Number of features: {}'.format(len(vectorizer.get_feature_names_out())))
print(vectorizer.get_feature_names_out())
print(vectorizer.vocabulary_)

print("")

print("#")
print("# EXERCISE: Create a spacy_tokenizer callback which takes a string and returns")
print("# a list of tokens (each token's text) with punctuation filtered out.")
print("#")
corpus = [
  "Students use their GPS-enabled cellphones to take birdview photographs of a land in order to find specific danger points such as rubbish heaps.",
  "Teenagers are enthusiastic about taking aerial photograph in order to study their neighbourhood.",
  "Aerial photography is a great way to identify terrestrial features that aren’t visible from the ground level, such as lake contours or river paths.",
  "During the early days of digital SLRs, Canon was pretty much the undisputed leader in CMOS image sensor technology.",
  "Syrian President Bashar al-Assad tells the US it will 'pay the price' if it strikes against Syria."
]
print("===== Reprinting Corpus ...")
print(corpus)

def my_spacy_tokenizer(doc):
  return [token.text for token in nlp(doc) if not token.is_punct]

myvectorizer = CountVectorizer(tokenizer=spacy_tokenizer, lowercase=False, binary=True)
#myvectorizer = CountVectorizer()
mybinaryBow = myvectorizer.fit_transform(corpus)
print("===== Printing Binary Bag of words from the corpus ...")
print(mybinaryBow)