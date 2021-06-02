
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from TextNormalizer_custom import TextNormalizer
from main_gensim import GensimVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ('normalizer',TextNormalizer()),
    ('vectorizer', GensimVectorizer()),
    ('bayes', MultinomialNB())
])
search = GridSearchCV(model,param_grid={
    'count_analyzer':['word','char','char_wb'],
    'count_ngram_range':[(1,1), (1,2), (1,3), (1,4), (1,5), (2,3)],
    'onehot__threshold': [0.0, 1.0, 2.0, 3.0],
    'bayes__alpha': [0.0, 1.0],
})


