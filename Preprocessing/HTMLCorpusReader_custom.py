import codecs
import bs4
import nltk
from nltk import sent_tokenize
from nltk import wordpunct_tokenize
from nltk import pos_tag
import time
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from readability.readability import Unparseable
from readability.readability import Document as Paper
import logging
import os
import pickle

CAT_PATTERN = r'([a-z_\s]+)/.*'
DOC_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.json'
TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'p', 'li']
tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'p', 'li']


class HTMLCorpusReader(CategorizedCorpusReader, CorpusReader):
    """    Объект чтения корпуса сHTML-документами для получения
         возможности дополнительной предварительной обработки.    """

    def __init__(self, root, fileids=DOC_PATTERN, encoding='utf8', tags=TAGS, **kwargs):
        """Инициализирует объект чтения корпуса.Аргументы, управляющие классификацией (``cat_pattern``, ``cat_map``
        и``cat_file``), передаются в конструктор ``CategorizedCorpusReader``. остальные аргументы передаются
        вконструктор ``CorpusReader``."""
        # Добавить шаблон категорий, если он не был передан в класс явно
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

            # Инициализировать объекты чтения корпуса из NLTK
            CategorizedCorpusReader.__init__(self, kwargs)
            CorpusReader.__init__(self, root, fileids, encoding)

            # Сохранить теги, подлежащие извлечению
            self.tags = tags

    def resolve(self, fileids, categories):
        """Возвращает список идентификаторов файлов или названий категорий,которые передаются каждой
        внутренней функции объекта чтения корпуса. Реализована по аналогии с``CategorizedPlaintextCorpusReader``
        в NLTK. """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")
        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids=None, categories=None):
        """        Возвращает полный текст HTML-документа, закрывая его         по завершении чтения.        """
        # Получить список файлов для чтени
        fileids = self.resolve(fileids, categories)

        # Cоздать    генератор, загружающий    документы    в память    по    одному
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as f:
                yield f.read()

    def sizes(self, fileids=None, categories=None):
        """Возвращает список кортежей, идентификатор файла иего размер.Эта функция используется для выявления
        необычно больших файлов в корпусе. """

        # Получить список файлов
        fileids = self.resolve(fileids, categories)

        # Создать генератор, возвращающий имена иразмеры файлов
        for path in self.abspaths(fileids):
            yield path, os.path.getsize(path)

    def html(self, fileids=None, categories=None):
        """Возвращает содержимое HTML каждого документа, очищая его
        с помощью библиотеки readability-lxml."""

        for doc in self.docs(fileids, categories):
            try:
                yield Paper(doc).summary()
            except Unparseable as e:
                print("Could not parse HTML: {}".format(e))
            continue

        log = logging.getLogger("readability.readability")
        log.setLevel('WARNING')

    def paras(self, fileids=None, categories=None):
        for html in self.html(fileids, categories):
            soup = bs4.BeautifulSoup(html, 'lxml')
            for element in soup.find_all(tags):
                yield element.text
            soup.decompose()

    def sents(self, fileids=None, categories=None):
        """Использует встроенный механизм для выделения предложений из
        абзацев. Обратите внимание, что для парсинга разметки HTML
        этот метод использует BeautifulSoup."""
        for paragraph in self.paras(fileids, categories):
            for sentence in sent_tokenize(paragraph):
                yield sentence

    def words(self, fileids=None, categories=None):
        """Использует встроенный механизм для выделения слов из предложений.
        Обратите внимание, что для парсинга разметки HTML
        этот метод использует BeautifulSoup"""
        for sentence in self.sents(fileids, categories):
            for token in wordpunct_tokenize(sentence):
                yield token

    def tokenize(self, fileids=None, categories=None):
        """Сегментирует, лексемизирует и маркирует документ в корпусе."""
        for paragraph in self.paras(fileids=fileids):
            yield [
                pos_tag(wordpunct_tokenize(sent))
                for sent in sent_tokenize(paragraph)
            ]

    def descibe(self, fileids=None, categories=None):
        """Выполняет обход содержимого корпуса ивозвращает
        словарь сразнообразными оценками, описывающими
        состояние корпуса."""
        started = time.time()

        # Структуры для подсчета
        counts = nltk.FreqDist()
        tokens = nltk.FreqDist()

        # Выполнить обход абзацев, выделить лексемы иподсчитать их
        for para in self.paras(fileids, categories):
            counts['para'] += 1

            for sent in para:
                counts['sents'] += 1
                for word, tag in sent:
                    counts['words'] += 1
                    tokens[word] += 1

        # Определить число файлов и категорий в корпусе
        n_fileids = len(self.resolve(fileids, categories) or self.fileids())
        n_topics = len(self.categories(self.resolve(fileids, categories)))

        # Вернуть структуру данных с информацией
        return {'files': n_fileids,
                'topics': n_topics,
                'paras': counts['paras'],
                'sents': counts['sents'],
                'words': counts['words'],
                'vocab': len(tokens),
                'lexdiv': float(counts['words']) / float(len(tokens)),
                'ppdoc': float(counts['paras']) / float(n_fileids),
                'sppar': float(counts['sents']) / float(counts['paras']),
                'secs': time.time() - started, }


class Preprocessor(object):
    """Обертывает `HTMLCorpusReader` ивыполняет лексемизацию
        смаркировкой частями речи."""

    def __init__(self, corpus, target=None, **kwargs):
        self.corpus = corpus
        self.target = target

    def fileids(self, fileids=None, categories=None):
        fileids = self.corpus.resolve(fileids, categories)
        if fileids:
            return fileids
        return self.corpus.fileids()

    def abspath(self, fileid):
        # Найти путь к каталогу относительно корня исходного корпуса.
        parent = os.path.relpath(
            os.path.dirname(self.corpus.abspath(fileid)), self.corpus.root
        )
        # Выделить части пути для реконструирования
        basename = os.path.basename(fileid)
        name, ext = os.path.splitext(basename)

        # Сконструировать имя файла срасширением .pickle
        basename = name + '.pickle'

        # Вернуть путь кфайлу относительно корня целевого корпуса.
        return os.path.normpath(os.path.join(self.target, parent, basename))

    def tokenize(self, fileids=None, categories=None):
        """Сегментирует, лексемизирует и маркирует документ в корпусе."""
        for paragraph in self.paras(fileids=fileids):
            yield [
                pos_tag(wordpunct_tokenize(sent))
                for sent in sent_tokenize(paragraph)
            ]

    def process(self, fileid):
        """Вызывается для одного файла, проверяет местоположение на диске,
            чтобы гарантировать отсутствие ошибок, использует +tokenize()+ для
            предварительной обработки и записывает трансформированный документ
            в виде сжатого архива в заданное место."""

        # Определить путь кфайлу для записи результата.
        target = self.abspath(fileid)
        parent = os.path.dirname(target)

        # Убедиться всуществовании каталога
        if not os.path.exists(parent):
            os.makedirs(parent)

        # Убедиться, что parent— это каталог, а не файл
        if not os.path.isdir(parent):
            raise ValueError(
                'Please supply a directory to write preprocessed data to. '
            )
        # Создать структуру данных для записи вархив
        document = list(self.tokenize(fileid))

        # Записать данные вархив на диск
        with open(target, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)

        # Удалить документ из памяти
        del document

        # Вернуть путь кцелевому файлу
        return target

    def transform(self, fileids=None, categories=None):
        # Создать целевой каталог, если его еще нет
        if not os.path.exists(self.target):
            os.makedirs(self.target)

        # Получить имена файлов для обработки
        for fileid in self.fileids(fileids, categories):
            yield self.process(fileid)
