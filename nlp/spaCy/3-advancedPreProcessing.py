import spacy
from spacy import displacy
# The general Matcher is one of multiple matcher objects included with spaCy.
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

print("#############################################")
print("########## Advanced pre-processing ##########")
print("#############################################")
phrase = "John watched an old movie at the cinema."
print("Phrase '" + phrase + "'")
doc = nlp(phrase)
print("")

print('**********************************')
print('***** Part-of-Speech tagging *****')
print('**********************************')
print("===== Print course-grained tags with token.pos_ ...")
for token in doc:
    print([token.text, token.pos_])
print("===== Get a description for a POS tag with 'spacy.explain'")
print("PROPN is " + spacy.explain("PROPN"))
print("===== Print fine-grained tags with tag_ attribute. Fine-grained tags provide more detailed information about a token such as its tense and, if a word is a pronoun, what specific type of pronoun it is.")
for token in doc:
    print([token.text, token.tag_])
print("NNP is " + spacy.explain("NNP"))
print("VBD is " + spacy.explain("VBD"))
print("")

print('************************************')
print('***** Named Entity Recognition *****')
print('************************************')
phrase = "Volkswagen is developing an electric sedan which could potentially come to America next fall."
print("Phrase '" + phrase + "'")
doc = nlp(phrase)
print("===== There are multiple ways to access named entities. One way is through the ent_type_ attribute.")
for token in doc:
    print([token.text, token.ent_type_])
print("ORG is " + spacy.explain("ORG"))
print("GPE is " + spacy.explain("GPE"))
print("DATE is " + spacy.explain("DATE"))
print("===== You can also check if a token is an entity before printing it by checking whether the ent_type (note the lack of trailing underscore) attribute is non-zero.")
print([(t.text, t.ent_type_) for t in doc if t.ent_type != 0])
print("===== Another way is through the ents property of the Doc object. Here, we iterate through ents and print the entity itself and its label.")
print([(ent.text, ent.label_) for ent in doc.ents])
print("===== You can also access the positions of entities:")
print([(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents])

# We need to set the 'jupyter' variable to True in order to output
# the visualization directly. Otherwise, you'll get raw HTML.
### displacy.serve(doc, style='ent', auto_select_port=True)
print("")

print('*******************')
print('***** Parsing *****')
print('*******************')
phrase = "She enrolled in the course at the university."
print("Phrase '" + phrase + "'")
doc = nlp(phrase)
# Note the 'style' argument is assigned a 'dep' flag this time around.
### displacy.serve(doc, style='dep', auto_select_port=True)
print("===== The dependency labels can be accessed through the dep_ attribute.")
for token in doc:
    print([token.text, token.dep_])
print("===== You can use spacy.explain to get information on a particular annotation.")
print("nsubj is " + spacy.explain("nsubj"))
print("ROOT is " + spacy.explain("ROOT"))
print("prep is " + spacy.explain("prep"))
print("det is " + spacy.explain("det"))
print("pobj is " + spacy.explain("pobj"))
print("===== The labels above don't show how the words are related to each other (the arcs). To get a better idea, you can print the head of each dependency.")
for token in doc:
    print([token.text, token.dep_, token.head.text])
print("")

print('************************************************')
print('***** Using spaCy Matcher to find patterns *****')
print('************************************************')
phrase = "I want to book a hotel room."
print("Phrase '" + phrase + "'")
doc = nlp(phrase)
print("===== Tokenize and print tokens ...")
tokens = [tokens.lower_ for tokens in doc]
print(tokens)

# We initialize the Matcher with the spaCy vocab object, which contains
# words along with their labels and entities.
matcher = Matcher(nlp.vocab)

# Patterns are expressed as an ordered sequence. Here, we're looking
# to match occurrences starting with a 'book' string followed by
# a determiner (DET) POS tag, then a noun POS tag.
# The OP key marks the match as optional in some way.

# Here, the DET POS (marked with '?') will match 0 or 1 times, and
# the NOUN POS (marked with '+') will match 1 or more times.
# See this link for more information:
# https://spacy.io/usage/rule-based-matching#quantifiers
pattern = [
  {'TEXT': 'book'},
  {'POS': 'DET', 'OP': '?'},
  {'POS': 'NOUN', 'OP': '+'},
]
print("===== Match with this pattern", pattern)

# We give our pattern a label and pass it to the matcher.
matcher.add('USER_INTENT', [pattern])

# Run the matcher over the doc.
#[token.lower_ for token in doc]
matches = matcher(doc)

# For each match, the matcher returns a tuple specifying a match id, start, 
# and end of the match.
print("===== Matches:", [doc[start:end].text for match_id, start, end in matches])

phrase = "I loved dogs but now I love cats more and love is always on top of my mind."
print("Phrase '" + phrase + "'")
doc = nlp(phrase)
pattern = [
    {"LEMMA": "love", "POS": "VERB"},
    {"POS": "NOUN"}
]
print("===== Match with this pattern", pattern)
matcher.add('USER_INTENT', [pattern])
matches = matcher(doc)
for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)
#
# EXERCISE: using doc.ents, identify and print the dates in this sentence.
# Expected output: ['Feb 13th', 'Feb 24th']
#
phrase = "We'll be in Osaka on Feb 13th and leave on Feb 24th."
doc = nlp(phrase)
print("===== Identify and print the dates in the following sentence:")
print(phrase)
print("--> Expected output: ['Feb 13th', 'Feb 24th']")
print("=== Printing dates within the phrase ...")
entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
for entity in entities:
    if entity[1] == "DATE":
        print(entity[0])

#
# EXERCISE: Read about spaCy's PhraseMatcher
# https://spacy.io/usage/rule-based-matching#phrasematcher
#
# Expected output: [(0, 2), (15, 17)]
#
phrase = "Caesar Augustus was the founder of the Roman Principate (the first phase of the Roman Empire)."
doc = nlp(phrase)
print("===== Using PhraseMatcher, find start and end index of all occurrences of 'Caesar Augustus' and 'Roman Empire' (case-insensitive).")
print(phrase)
print("--> Expected output: [(0, 2), (15, 17)]")
print("=== Printing start and end indexes ...")

matcher = PhraseMatcher(nlp.vocab)
terms = ["Caesar Augustus", "Roman Empire"]
# Only run nlp.make_doc to speed things up
patterns = [nlp.make_doc(text) for text in terms]
matcher.add("TerminologyList", patterns)

matches = matcher(doc)
for match_id, start, end in matches:
    span = doc[start:end]
    print(span.text + " - (" + str(start) + ", " + str(end) + ")")