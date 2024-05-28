import spacy

nlp = spacy.load("en_core_web_sm")

print("##########################################")
print("########## Basic pre-processing ##########")
print("##########################################")
phrase = "He told Dr. Lovato that he was done with the tests. He would post the results shortly."
print("Phrase '" + phrase + "'")
doc = nlp(phrase)
print("")

print('************************')
print('***** Case folding *****')
print('************************')
print("===== Print tokens lowercase ...")
print([token.lower_ for token in doc])
# Skip case-folding if a token is the start of a sentence.
print('===== Skip case-folding if a token is the start of a sentence ...')
print([token.lower_ if not token.is_sent_start else token for token in doc])
print("")

print('*****************************')
print('***** Stop word removal *****')
print('*****************************')
print("===== spaCy's default stop word list")
print(nlp.Defaults.stop_words)
print(len(nlp.Defaults.stop_words))
print("")

print("===== Print tokens if not stop word")
print([token for token in doc if not token.is_stop])
print("")

print('*************************')
print('***** Lemmatization *****')
print('*************************')
print('===== Cycle over tokens and print the following:')
print('TOKEN - TOKEN LOWERCASE - TOKEN LEMMA - TOKEN POS - TOKEN IS ALPHA')
for token in doc:
    print((token.text, token.lower_, token.lemma_, token.pos_, token.is_alpha))