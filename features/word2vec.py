from __future__ import division, print_function
from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
from sklearn.feature_extraction.text import VectorizerMixin
from preprocess import ark_tweet_tokenizer
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
import gensim.models

__all__ = [' Word2VecFeatures']

class Word2VecFeatures(BaseEstimator, VectorizerMixin, TransformerMixin):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 analyzer='word', dtype=np.float32, model_name=None):

        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.dtype = dtype
        self.model_name = model_name
        # TODO: add the max and min document frequency

    def get_feature_names(self):
        return np.array(['word_emb_' + i for i in range(self.num_features_)])

    def _words(self, tokens, stop_words=None):
        """Turn tokens into a sequence of words after stop words filtering"""
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        return tokens

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        if callable(self.analyzer):
            return self.analyzer

        preprocess = self.build_preprocessor()

        if self.analyzer == 'word':
            stop_words = self.get_stop_words()
            tokenize = self.build_tokenizer()

            return lambda doc: self._words(
                tokenize(preprocess(self.decode(doc))), stop_words)

        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             self.analyzer)

    def make_feature_vec(self, words):

        # Function to average all of the word vectors in a given
        # paragraph
        #
        # Pre-initialize an empty numpy array (for speed)
        feature_vec = np.zeros((self.num_features_,), dtype=self.dtype)
        #
        nwords = 0.

        # Loop over each word in the review and, if it is in the model's
        # vocaublary, add its feature vector to the total
        for word in words:
            if word in self.index2word_set_:
                nwords = nwords + 1.
                feature_vec = np.add(feature_vec, self.model_[word])
        #
        # Divide the result by the number of words to get the average
        if nwords == 0.:
            nwords = nwords + 1.
        feature_vec = np.divide(feature_vec, nwords)
        return feature_vec

    def fit(self, documents, y=None):
        # TODO: implement word2vec training
        if self.model_name:
            print("Loading Word2Vec")
            self.model_ = gensim.models.KeyedVectors.load(self.model_name)
            self.num_features_ = self.model_.syn0.shape[1]
            # Index2word is a list that contains the names of the words in
            # the model's vocabulary. Convert it to a set, for speed
            self.index2word_set_ = set(self.model_.index2word)
            print ("Done Loading vectors")
        else:
            # TODO: implement word2vec training
            print ("Building Word2Vec")
            #sentences = []
            #for doc in documents:
            #    sentences.append(ark_tweet_tokenizer(doc.content))
            #self.model_ = word2vec.Word2Vec(sentences, size=300, window=5, min_count=5, workers=4, iter=1000)
            #self.model_.save('C:\\Users\\Niloofar Safi\\Desktop\\Kaggle.model')
            self.model_ = gensim.models.KeyedVectors.load('/home/niloofar/PycharmProjects/AdvancedNLP/Assign1/BlogWordModel.model')
            #/home/niloofar/Shahryar_Niloofar/WikiWordModelMain.model
            # / home / niloofar / Shahryar_Niloofar / AskWordModelMain.model
            self.num_features_ = self.model_.wv.syn0.shape[1]
            self.index2word_set_ = set(self.model_.wv.index2word)
            print (len(self.index2word_set_))
            print ("Done Loading vectors")
            pass

        return self

    def transform(self, documents):
        analyze = self.build_analyzer()
        # Preallocate a 2D numpy array, for speed
        doc_feature_vecs = np.zeros((len(documents), self.num_features_), dtype=self.dtype)
        #
        # Loop through the reviews
        for i, doc in enumerate(documents):
            #
            # Print a status message every 1000th review
            if i % 1000. == 0.:
                print("Document %d of %d" % (i, len(documents)))
            #
            # Call the function (defined above) that makes average feature vectors
            doc_feature_vecs[i] = self.make_feature_vec(analyze(doc))

        return doc_feature_vecs
