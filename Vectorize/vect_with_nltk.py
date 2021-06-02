from collections import defaultdict
import nltk
import string
from nltk.text import TextCollection


def tokenize(text):
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()

    for token in nltk.word_tokenize(text):
        if token in string.punctuation: continue
        yield stem.stem(token)


corpus = ["The elephant sneezed at the sight of potatoes.",
          "Bats can see via echolocation. See the bat sight sneeze!",
          "Wondering, she opened the door to the studio.", ]


def nltk_frequency_vectorize(corpus):
    # The NLTK frequency vectorize method
    from collections import defaultdict

    def vectorize(doc):
        features = defaultdict(int)

        for token in tokenize(doc):
            features[token] += 1

        return features

    return map(vectorize, corpus)


def nltk_one_hot_vectorize(corpus):
    # The NLTK one hot vectorize method
    def vectorize(doc):
        return {
            token: True
            for token in tokenize(doc)
        }

    return map(vectorize, corpus)


def nltk_tfidf_vectorize(corpus):

    from nltk.text import TextCollection

    corpus = [list(tokenize(doc)) for doc in corpus]
    texts = TextCollection(corpus)

    for doc in corpus:
        yield {
            term: texts.tf_idf(term, doc)
            for term in doc
        }




print(nltk_tfidf_vectorize)