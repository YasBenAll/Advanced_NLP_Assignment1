import spacy
import benepar
from nltk import ParentedTree

benepar.download('benepar_en3')
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
print("Please enter your text here:")
text = input()
doc = nlp(text)

# Extracting all Noun Phrases and head of each NP
print("------------------NOUN PHRASES---------------\n")
for chunk in doc.noun_chunks:
    print('Noun Phrase:', chunk.text, ',', 'NP Head:', chunk.root.text, "\n")

# Word Tokenization and tokens' features
print("------------------TOKENS---------------\n")
tok_id = []
tok_text = []
for token in doc:
    tok_id.append(token.i)
    tok_text.append(token.text)
    print("Token ID", token.i, ',', "Token:", token.text, ',', "Lemma:", token.lemma_, ','"POS tag:", token.pos_, ',',
          "Token Dependency Tag:",
          token.dep_, ',', "Token Head:", token.head, "\n")

# Extracting token dependents
print("------------------TOKEN DEPENDENTS---------------\n")
for index in tok_id:
    tok_dep = []
    for tokens in doc:
        if tok_text[index] == tokens.text:
            continue
        if index == tokens.head.i:
            tok_dep.append(tokens.text)
    print("Token ID:", index, ',', "Dependents:", tok_dep, "\n")

# Entity Recognition
print("-----------------ENTITIES---------------\n")
for ent in doc.ents:
    print('Entity:', ent.text, ',', 'Label:', ent.label_, "\n")

# Constituency Tree
sent = list(doc.sents)[0]
print("-----------------CONSTITUENCY TREE---------------\n")
print(sent._.parse_string)
parse_tree = ParentedTree.fromstring('(' + sent._.parse_string + ')')
print(parse_tree.pretty_print())

# Dependencies Tree
options = {"collapse_punct": False, "add_lemma": True}
displacy.serve(doc, style="dep", options=options)

