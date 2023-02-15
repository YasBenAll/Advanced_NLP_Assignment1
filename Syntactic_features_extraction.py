import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
print("Please enter your text here:")
text = input()
doc = nlp(text)

print("------------------FULL CONSTITUENTS---------------")
print('Noun phrases:')
for chunk in doc.noun_chunks:
    print('Noun Phrase:', chunk.text, ',', 'NP Head:', chunk.root.text)

print("------------------TOKENS---------------")
for token in doc:
    print("Token:", token.text, ',', "Lemma:", token.lemma_, ','"POS tag:", token.pos_, ',', "Token Dependency:",
          token.dep_, ',', "Token Head:", token.head)
print("-----------------ENTITIES---------------")
for ent in doc.ents:
    print('Entity:', ent.text, ',', 'Label:', ent.label_)
options = {"collapse_punct": False, "add_lemma": True, "compact": True}
displacy.serve(doc, style="dep", options=options)
