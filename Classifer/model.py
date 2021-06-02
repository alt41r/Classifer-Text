from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn_models.TextNormalizer_custom import TextNormalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from Preprocessing.PickledCorpusReader_custom import PickledCorpusReader
import nltk
from sklearn.metrics import classification_report
import tabulate
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
nltk.download('stopwords')
import time

class CorpusLoader(object):

    def __init__(self, reader, folds=12, shuffle=True, categories=None):
        self.reader = reader
        self.folds  = KFold(n_splits=folds, shuffle=shuffle)
        self.files  = np.asarray(self.reader.fileids(categories=categories))

    def fileids(self, idx=None):
        if idx is None:
            return self.files
        return self.files[idx]

    def documents(self, idx=None):
        for fileid in self.fileids(idx):
            yield list(self.reader.docs(fileids=[fileid]))

    def labels(self, idx=None):
        return [
            self.reader.categories(fileids=[fileid])[0]
            for fileid in self.fileids(idx)
        ]

    def __iter__(self):
        for train_index, test_index in self.folds.split(self.files):
            X_train = self.documents(train_index)
            y_train = self.labels(train_index)

            X_test = self.documents(test_index)
            y_test = self.labels(test_index)

            yield X_train, X_test, y_train, y_test



def identity(words):
    return words


def create_pipeline(estimator, reduction=False):
    steps = [
        ('normalize', TextNormalizer()),
        ('vectorize', TfidfVectorizer(
            tokenizer=identity, preprocessor=None, lowercase=False))
    ]
    if reduction:
        steps.append((
            'reduction', TruncatedSVD(n_components=10000)

        ))
    # добавить обьект оценки
    steps.append(('classifier', estimator))
    return Pipeline(steps)
def identity(words):
    return words


def create_pipeline(estimator, reduction=False):
    steps = [
        ('normalize', TextNormalizer()),
        ('vectorize', TfidfVectorizer(
            tokenizer=identity, preprocessor=None, lowercase=False))
    ]
    if reduction:
        steps.append((
            'reduction', TruncatedSVD(n_components=10000)

        ))
    # добавить обьект оценки
    steps.append(('classifer', estimator))
    return Pipeline(steps)


models = []
for form in (LogisticRegression, SGDClassifier):
    models.append(create_pipeline(form(), True))
    models.append(create_pipeline(form(), False))

models.append(create_pipeline(MultinomialNB(), False))
models.append(create_pipeline(GaussianNB(), True))

reader = PickledCorpusReader('/content/drive/MyDrive/sample')
labels = ['books', 'cinema', 'cooking', 'gaming', 'sports', 'tech']
loader = CorpusLoader(reader, 5, shuffle=True, categories=labels)
fields = ['model','precision','recall','accuracy','f1']
table = []
for model in models:

        name = model.named_steps['classifier'].__class__.__name__
        if 'reduction' in model.named_steps:
            name += " (TruncatedSVD)"

        scores = {
            'model': str(model),
            'name': name,
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'time': [],
        }

        for X_train, X_test, y_train, y_test in loader:
            start = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            scores['time'].append(time.time() - start)
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred, average='weighted'))
            scores['recall'].append(recall_score(y_test, y_pred, average='weighted'))
            scores['f1'].append(f1_score(y_test, y_pred, average='weighted'))
            row = [str(model)]
            for field in fields[1:]:
                row.append(np.mean(scores[field]))

            table.append(row)
table.sort(key=lambda row: row[-1], reverse=True)
print(tabulate.tabulate(table, headers=fields))
import pickle
from datetime import datetime, time

date = datetime.now().strftime("%Y-%m-%d")
path  = 'hobby-calssifer-{0}'.format(date)
with open(path,'wb') as f:
    pickle.save(model,f)