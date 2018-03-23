# -*- coding: utf-8 -*-
from __future__ import print_function
import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    DATA = os.path.join(basedir,'resources','Data_05_16_2016.csv')
    OUT_DIRECTORY=os.path.join(basedir,'output','analysis')
    CELERY_BROKER_URL = "amqp://guest:guest@localhost:5672//"
    CELERY_BACKEND_URL = "amqp://guest:guest@localhost:5672//"
    CHART_DIR=os.path.join(basedir,'charts')




    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    FEATURES = [
         'unigram',
         # 'bigram',
         # 'trigram',
         # 'binary_unigram',
         # 'binary_bigram',
         # 'binary_trigram',
         # 'char_3',
         # 'char_4',
         # 'char_5',
        # 'google_word_emb',
        #  'embedding',
        #  'wrd'
    ]

    VECTORS='.'



class TestingConfig(Config):
    TESTING = True


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
