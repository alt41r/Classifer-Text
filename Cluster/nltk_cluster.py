import nltk.cluster.util
from nltk.cluster import KMeansClusterer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
import unicodedata
import nltk
import nltk.corpus
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import wordnet as wn
from nltk import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from Preprocessing.PickledCorpusReader_custom import PickledCorpusReader


class KMeansClusters(BaseEstimator, TransformerMixin):
    def __init__(self, k=7):
        self.k = k
        self.distance = nltk.cluster.util.cosine_distance
        self.model = KMeansClusters(self.k)

    def fit(self, documents, labels=None):
        return self

    def transform(self, documnets):
        return self.model.cluster(documnets, assign_clusters=True)


class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='english'):
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def normalize(self, document):
        return [
            self.lemmatize(token, tag).lower()
            for paragraph in document
            for sentence in paragraph
            for (token, tag) in sentence
            if not self.is_punct(token) and not self.is_stopword(token)
        ]

    def lemmatize(self, token, pos_tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        return [' '.join(self.normalize(doc)) for doc in documents]


class OneHotVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = CountVectorizer(binary=True)

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        freqs = self.vectorizer.fit_transform(documents)
        return [freq.toarray()[0] for freq in freqs]


corpus = PickledCorpusReader('sample')  # path
docs = corpus.docs(categories=['news'])

model = Pipeline([
    ('norm', TextNormalizer()),
    ('vect', OneHotVectorizer()),
    ('clusters', KMeansClusters(k=7))
])
clusters = model.fit_transform(docs)
pickles = list(corpus.fileids(categories=['news']))
for idx, cluster in enumerate(clusters):
    print('Document "{}" assigned to cluster {}.'.format(pickles[idx], cluster))
