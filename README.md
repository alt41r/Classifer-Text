# Classifer-Text
Классификация корпуса документов

В данном репозитории реализовано:
1)Создание корпуса данных путем метода HTMLCorpusReader_custom
2)Считывание корпуса путем метода PickledCorpusReader
3)Удаление стопслов,пунктуации и лемматизация текста в методе TextNormalizer_custom
4)Обучение моделей MultinomialNB,GaussianNB,LogisticRegression,SGDClassifier
5)К каждой модели так же применилось сингулярное разложение
Оценка f1 = 0.816619 для SGDClassifier
