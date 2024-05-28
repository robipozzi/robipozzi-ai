import spacy

nlp = spacy.load("en_core_web_sm")

print("##################################")
print("########## Tokenization ##########")
print("##################################")
phrase = "He didn't want to pay $20 for this book."
print("Phrase '" + phrase + "'")
doc = nlp(phrase)
print('===== Print doc.text ...')
print(doc.text)
print('===== Tokenize phrase and printing tokens ...')
print([token.text for token in doc])
print('===== Print number of tokens in phrase ...')
tokens = [token for token in doc]
print("Number of tokens = ", len(tokens))
print('===== Cycle over tokens and print the following:')
print('TOKEN - TOKEN ENTITY TYPE - IS_CURRENCY, IS_DIGIT, IS_PUNCTUATION')
for token in doc:
    print((token, token.ent_type_, token.is_currency, token.is_digit, token.is_punct))
print("")

# We can view an individual token by indexing into the Doc object.
index = 0
print("===== View individual token at index " + str(index) + " ...")
print(doc[0])
print("")

# Slicing a Doc object returns a Span object.
index_final = 3
print("===== Slicing a Doc object between indexes " + str(index) + " and " + str(index_final) + " ...")
print(doc[index:index_final])
print("")

# Access a token's index in a sentence.
print("===== Access a token's index in a sentence ...")
print([(token.text, token.i) for token in doc])
print("")

print("# Iterate through the tokens to check whether there's a currency symbol.")
print("# If there is, and the currency label is followed by a number, print both the symbol and the number.")
print("# Look through https://spacy.io/api/token#attributes on how to check whether a token is a currency symbol or a number.")
print("# Expected output: \"$20\".")
amount = ""
for token in doc:
    if token.is_currency:
        index = token.i
        amount = token.text
        index = index + 1
        if doc[index].is_digit:
            amount = amount + doc[index].text
print(amount)