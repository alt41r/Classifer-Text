import nltk
import string


def tokenize(text):
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()

    for token in nltk.word_tokenize(text):
        if token in string.punctuation: continue
        yield stem.stem(token)


corpus = ["The elephant sneezed at the sight of potatoes.",
          "Bats can see via echolocation. See the bat sight sneeze!",
          "Wondering, she opened the door to the studio.", ]
