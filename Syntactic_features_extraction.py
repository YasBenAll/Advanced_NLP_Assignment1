import spacy
from spacy import displacy
import nltk

nlp = spacy.load("en_core_web_sm")
print("Please enter your text here:")
text = input()
doc = nlp(text)

print("------------------FULL CONSTITUENTS---------------")
print('Noun phrases:')
for chunk in doc.noun_chunks:
    print('Noun Phrase:', chunk.text, ',', 'NP Head:', chunk.root.text)

print("------------------TOKENS---------------")
tagged_text = []
for token in doc:
    tagged_text.append((token.text, token.pos_))
    print("Token:", token.text, ',', "Lemma:", token.lemma_, ','"POS tag:", token.pos_, ',', "Token Dependency:",
          token.dep_, ',', "Token Head:", token.head)
print("-----------------ENTITIES---------------")
for ent in doc.ents:
    print('Entity:', ent.text, ',', 'Label:', ent.label_)

grammar = r"""
  NP: {<DET|ADJ|NOUN.*>+}       
  PP: {<ADP><NP>}               
  VP: {<VERB.*><NP|PP|CLAUSE>+$} 
  CLAUSE: {<NP><VP>}           
  """
cp = nltk.RegexpParser(grammar, loop=2)
result = cp.parse(tagged_text)
result.draw()
options = {"collapse_punct": False, "add_lemma": True}
displacy.serve(doc, style="dep", options=options)
