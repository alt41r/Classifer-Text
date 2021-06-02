import pickle
from Preprocessing.HTMLCorpusReader_custom import HTMLCorpusReader
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
CAT_PATTERN = r'([a-z_\s]+)/.*'

class PickledCorpusReader(HTMLCorpusReader):
    def __init__(self, root, fileids=PKL_PATTERN, **kwargs):
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids)

    def docs(self, fileids=None, categories=None):
        fileids = self.resolve(fileids, categories)

        # Загружать документы впамять по одному.
        for path in self.abspaths(fileids):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def paras(self, fileids=None, categories=None):
        for doc in self.docs(fileids, categories):
            for para in doc:
                yield para

    def sents(self, fileids=None, categories=None):
        for para in self.paras(fileids, categories):
            for sent in para:
                yield sent

    def tagged(self,fileids=None,categories=None):
        for sent in self.sents(fileids,categories):
            for tagged_token in sent:
                yield tagged_token

    def words(self, fileids=None, categories=None):
        for tagged in self.tagged(fileids,categories):
            yield tagged[0]
