## Code for extracting syntactic features of an English text.

Use an IDE like VScode or Pycharm to run the code. Jupyter Notebook does not work*

## Running Instructions:
Step 1: Libraries installation.
>pip install -r requirements.txt

Step 2: Spacy English package installation.
>python -m spacy download en_core_web_sm

Step 3: The code is ready to run. In order to view the dependency tree follow http://localhost:5000/ link.


### The features able to be extracted are:

1. Noun Phrases of the text and the head of each noun phrase.
2. Tokens,token's lemma, token's head, token's part-of-speech (POS) tag, token's dependency tag, and token's dependents.
3. Constituency Tree of the input text
4. Dependency Tree of the input text
