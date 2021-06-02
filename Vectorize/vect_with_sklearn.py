from sklearn.feature_extraction.text import CountVectorizer
import nltk
import string
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize(text):
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()

    for token in nltk.word_tokenize(text):
        if token in string.punctuation: continue
        yield stem.stem(token)


corpus = ["The elephant sneezed at the sight of potatoes.",
          "Bats can see via echolocation. See the bat sight sneeze!",
          "Wondering, she opened the door to the studio.", ]


def sklearn_frequency_vectorize(corpus):

    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(corpus)


def sklearn_one_hot_vectorize(corpus):


    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import Binarizer

    freq = CountVectorizer()
    vectors = freq.fit_transform(corpus)

    print(len(vectors.toarray()[0]))

    onehot = Binarizer()
    vectors = onehot.fit_transform(vectors.toarray())

    print(len(vectors[0]))


def sklearn_tfidf_vectorize(corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(corpus)

print(sklearn_tfidf_vectorize(corpus))
