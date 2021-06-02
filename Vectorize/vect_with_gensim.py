import nltk
import string
import gensim
import numpy as np


def tokenize(text):
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()

    for token in nltk.word_tokenize(text):
        if token in string.punctuation: continue
        yield stem.stem(token)


corpus = ["The elephant sneezed at the sight of potatoes.",
          "Bats can see via echolocation. See the bat sight sneeze!",
          "Wondering, she opened the door to the studio.", ]


def gensim_frequency_vectorize(corpus):
    # The Gensim frequency vectorize method

    tokenized_corpus = [list(tokenize(doc)) for doc in corpus]
    id2word = gensim.corpora.Dictionary(tokenized_corpus)
    return [id2word.doc2bow(doc) for doc in tokenized_corpus]


def gensim_one_hot_vectorize(corpus):
    # The Gensim one hot vectorize method

    corpus = [list(tokenize(doc)) for doc in corpus]
    id2word = gensim.corpora.Dictionary(corpus)

    corpus = np.array([
        [(token[0], 1) for token in id2word.doc2bow(doc)]
        for doc in corpus
    ])

    return corpus


def gensim_tfidf_vectorize(corpus):
    corpus = [list(tokenize(doc)) for doc in corpus]
    lexicon = gensim.corpora.Dictionary(corpus)

    tfidf = gensim.models.TfidfModel(dictionary=lexicon, normalize=True)
    vectors = [tfidf[lexicon.doc2bow(vector)] for vector in corpus]

    lexicon.save_as_text('test.txt')
    tfidf.save('tfidf.pkl')

    return vectors


def gensim_doc2vec_vectorize(corpus):
    from gensim.models.doc2vec import TaggedDocument, Doc2Vec

    corpus = [list(tokenize(doc)) for doc in corpus]
    docs = [
        TaggedDocument(words, ['d{}'.format(idx)])
        for idx, words in enumerate(corpus)
    ]
    model = Doc2Vec(docs,  min_count=0)
    return model.docvecs


print(gensim_doc2vec_vectorize(corpus)[0])
