import pandas as pd
from config import config
from features import create_feature
from sklearn.utils import shuffle
import os
import joblib
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation, grid_search
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder

app = config[os.getenv('CYBERBULLYING_CONFIG') or 'default']
DUMPED_VECTOR_DIR = "extracted_features"

def load_data(path):
    X, Y, doc_id = [], [], []
    df = pd.read_csv(path, encoding='utf-8')
    for index, row in df.iterrows():
        X.append(row['Document'])
        Y.append(row['Label'])
        doc_id.append(row['File_name'].replace('.txt', '.xml'))
    return X, Y, doc_id

def load_data2(path):
    X, Y, doc_id = [], [], []
    df = pd.read_csv(path, encoding='utf-8')
    for index, row in df.iterrows():
        X.append(row['Document'])
        # Y.append(row['Label'])
        doc_id.append(row['File_name'].replace('.txt', '.xml'))
    return X,doc_id


def extract_features():
    for feature in app.FEATURES:
        print ("Feature : %s"%feature)
        f_name, feature_obj = create_feature(feature)
        X, Y, id = load_data('train_data_preprocessed.csv')
        X1, id = load_data2('twitter_test_data_preprocessed.csv')
        X2, id = load_data2('blog_test_data_preprocessed.csv')
        X, Y = shuffle(X, Y,random_state=1234)
        x_train = X[:11000]
        x_test = X[11000:]
        y_train = Y[:11000]
        X_train_features = feature_obj.fit_transform(x_train, y_train)
        X_test_features = feature_obj.transform(x_test)
        print ("Shape of X_train: {}".format(X_train_features.shape))
        print ("Shape of X_dev: {}".format(X_test_features.shape))
        X_features = feature_obj.fit_transform(X, Y)
        X_test1_features = feature_obj.transform(X1)
        X_test2_features = feature_obj.transform(X2)
        print ("Shape of X_train: {}".format(X_features.shape))
        print ("Shape of X_test1: {}".format(X_test1_features.shape))
        print ("Shape of X_test2: {}".format(X_test2_features.shape))
        joblib.dump(X_train_features, os.path.join(DUMPED_VECTOR_DIR, feature + '_train' + '.pkl'))
        joblib.dump(X_test_features, os.path.join(DUMPED_VECTOR_DIR, feature + '_test' + '.pkl'))
        joblib.dump(X_features, os.path.join(DUMPED_VECTOR_DIR, feature + '.pkl'))
        joblib.dump(X_test1_features, os.path.join(DUMPED_VECTOR_DIR, feature + '_test1' + '.pkl'))
        joblib.dump(X_test2_features, os.path.join(DUMPED_VECTOR_DIR, feature + '_test2' + '.pkl'))
        print("Feature %s done!!"%f_name)


def combination(feature_list):
    return np.concatenate((feature_list), axis=1)


def get_features(suffix):
    feature_list = []
    for feature in app.FEATURES:

        print ("Feature : %s" % feature)
        path = os.path.join(DUMPED_VECTOR_DIR, feature + suffix + '.pkl')
        loaded_feature = joblib.load(path)

        if not isinstance(loaded_feature, np.ndarray):
            loaded_feature = loaded_feature.toarray()

        feature_list.append(loaded_feature)
    return feature_list

def gridsearch(clf, X, Y, parameters):
    """
    perform a grid search for best value of parameters
    :param clf:
    :param X:
    :param Y:
    :param parameters:
    :return:
    """
    gs = grid_search.GridSearchCV(clf, parameters, cv=3, scoring='f1_weighted', n_jobs=1)
    gs.fit(X, Y)
    best_clf = gs.best_estimator_
    print("Best Classifier")
    print(best_clf)
    return best_clf


def train(test_set, classifier):

    if test_set == 'test':
        labels = ['female', 'male']

        X, Y, id = load_data('train_data_preprocessed.csv')
        X, Y = shuffle(X, Y, random_state=1234)
        x_train = X[:11000]
        x_test = X[11000:]

        train_features = get_features('_train')
        test_features = get_features('_test')

        X_train = combination(train_features)
        X_test = combination(test_features)

        le = LabelEncoder()
        le.fit(Y)
        Y = le.fit_transform(Y)
        y_train = Y[:11000]
        y_test = Y[11000:]
        # if sp.issparse(X_train):
        #     X_train_all_features = sp.vstack((X_train, X_test))
        # else:
        #     X_train_all_features = np.vstack((X_train, X_test))
        # print ("Shape of X_train + X_dev: {}".format(X_train_all_features.shape))

        # Create classifiers
        if classifier =='LR':
            clf = LogisticRegression(C=10)
        elif classifier == 'SVM':
            clf = LinearSVC()

        # 10 fold cross validation
        # cv = cross_validation.KFold(n=len(X_train_all), n_folds=10, shuffle=True, random_state=1234)

        train_score, test_score = [], []
        Y_actual, Y_predicted = [], []
        clf.fit(X_train, y_train)

        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        output = clf.predict(X_test)
        # output = clf.predict(X_test)
        Y_actual.extend(y_test)
        Y_predicted.extend(output)

        # i += 1
        print("Done with experiment")
        # print evaluation metrics

        report(le, Y_actual, Y_predicted)
        # print("Fold  %d  done .." % (i + 1))

        # print("Running Experiment")
        # print (len(X),len(Y))
        # parameters = {'C': [0.1, 1.0, 10, 100, 1000]}
        # i = 0

        # for train, test in cv:
        #     X_train, X_test = X_train_all[train], X_train_all[test]
        #     y_train, y_test = Y[train], Y[test]
        #
        #     print (len(X_train), len(y_train))
        #     #best_clf = gridsearch(clf, X_train, y_train, parameters)
        #     clf.fit(X_train, y_train)
        #
        #     train_score = clf.score(X_train, y_train)
        #     test_score = clf.score(X_test, y_test)
        #
        #     output = clf.predict(X_test)
        #     # output = clf.predict(X_test)
        #     Y_actual.extend(y_test)
        #     Y_predicted.extend(output)
        #
        #
        #     i += 1
        #     print("Done with experiment")
        #     # print evaluation metrics
        #
        #     report(le,  Y_actual, Y_predicted)
        #     print("Fold  %d  done .." % (i + 1))

    if test_set == 'test1':
        if classifier =='LR':
            clf = LogisticRegression(C=10)
        elif classifier == 'SVM':
            clf = LinearSVC(C=0.1)
        # clf = LinearSVC(class_weight='balanced')
        # clf = LogisticRegression(class_weight='balanced')
        parameters = {'C': [1e-1, 1, 10, 100, 1000]}
        train_features = get_features('')
        test_features = get_features('_test1')
        X1, id1 = load_data2('twitter_test_data_preprocessed.csv')
        X, Y, id = load_data('train_data_preprocessed.csv')
        X, Y_train, id = shuffle(X, Y, id, random_state=1234)

        # print(Y_train[:10])
        class_le = LabelEncoder()
        Y = class_le.fit_transform(Y_train)
        # print(Y[:10])

        # train_features1 = get_features('_train')
        # dev_features = get_features('_test')
        # X_train = combination(train_features1)
        # X_dev = combination(dev_features)

        X_test = combination(test_features)
        X_train_final = combination(train_features)

        # if sp.issparse(X_train):
        #     X_train_all_features = sp.vstack((X_train, X_dev))
        # else:
        #     X_train_all_features = np.vstack((X_train, X_dev))
        # print ("Shape of X_train + X_dev: {}".format(X_train_all_features.shape))
        #
        # ps = PredefinedSplit(test_fold=[-1] * X_train.shape[0] + [0] * X_dev.shape[0])
        #
        # grid = GridSearchCV(estimator=clf, param_grid=parameters, cv=ps, scoring='f1_weighted')
        # grid.fit(X_train_all_features, Y)
        #
        # print("Best score: %0.3f" % grid.best_score_)
        # print("Best parameters set:")
        # best_parameters = grid.best_estimator_.get_params()
        # for param_name in sorted(parameters.keys()):
        #     print("\t%s: %r" % (param_name, best_parameters[param_name]))
        #
        # for params, mean_score, scores in grid.grid_scores_:
        #     print("%0.3f+/-%0.2f %r"
        #           % (mean_score, scores.std() / 2, params))
        #
        # best_clf = grid.best_estimator_.fit(X_train_final, Y)
        #
        # print("[INFO] Training + Validations set instances %d and labels %d" % (
        #     X_train_all_features.shape[0], Y.shape[0]))
        best_clf = clf.fit(X_train_final, Y)
        print("Training Accuracy =%.3f" % best_clf.score(X_train_final, Y))

        # print("[INFO] Test Set  instances %d and labels %d" % (
        #     X_test.shape[0], y_test.shape[0]))

        y_predicted = best_clf.predict(X_test)

        prediction = []
        ids = []
        for i in range(len(y_predicted)):
            ids.append(id1[i])
            if y_predicted[i] == 0:
                prediction.append("female")
            if y_predicted[i] == 1:
                prediction.append("male")

        twitter_test_data = {}
        twitter_test_data["File_name"] = ids
        twitter_test_data["Prediction"] = prediction
        df = pd.DataFrame(twitter_test_data, columns=['File_name', 'Prediction'])
        fname = 'result/twitter_' + '-'.join([feature for feature in app.FEATURES]) + '_' + classifier + '.csv'
        print(fname)
        df.to_csv(fname, index=False, header=False, encoding='utf-8')

    if test_set == 'test2':
        if classifier =='LR':
            clf = LogisticRegression(C=1)
        elif classifier == 'SVM':
            clf = LinearSVC(C=0.1)
        # clf = LinearSVC(class_weight='balanced')
        # clf = LogisticRegression(class_weight='balanced')
        parameters = {'C': [1e-1, 1, 10, 100, 1000]}
        train_features = get_features('')
        test_features = get_features('_test2')
        X1, id1 = load_data2('blog_test_data_preprocessed.csv')
        X, Y, id = load_data('train_data_preprocessed.csv')
        X, Y_train, id = shuffle(X, Y, id, random_state=1234)


        class_le = LabelEncoder()
        Y = class_le.fit_transform(Y_train)


        # train_features1 = get_features('_train')
        # dev_features = get_features('_test')
        # X_train = combination(train_features1)
        # X_dev = combination(dev_features)

        X_test = combination(test_features)
        X_train_final = combination(train_features)

        # if sp.issparse(X_train):
        #     X_train_all_features = sp.vstack((X_train, X_dev))
        # else:
        #     X_train_all_features = np.vstack((X_train, X_dev))
        # print ("Shape of X_train + X_dev: {}".format(X_train_all_features.shape))
        #
        # ps = PredefinedSplit(test_fold=[-1] * X_train.shape[0] + [0] * X_dev.shape[0])
        #
        # grid = GridSearchCV(estimator=clf, param_grid=parameters, cv=ps, scoring='f1_weighted')
        # grid.fit(X_train_all_features, Y)
        #
        # print("Best score: %0.3f" % grid.best_score_)
        # print("Best parameters set:")
        # best_parameters = grid.best_estimator_.get_params()
        # for param_name in sorted(parameters.keys()):
        #     print("\t%s: %r" % (param_name, best_parameters[param_name]))
        #
        # for params, mean_score, scores in grid.grid_scores_:
        #     print("%0.3f+/-%0.2f %r"
        #           % (mean_score, scores.std() / 2, params))
        #
        # best_clf = grid.best_estimator_.fit(X_train_final, Y)
        #
        # print("[INFO] Training + Validations set instances %d and labels %d" % (
        #     X_train_all_features.shape[0], Y.shape[0]))
        #
        # # print("Training Accuracy =%.3f" % best_clf.score(X_train_final, Y))
        #
        # # print("[INFO] Test Set  instances %d and labels %d" % (
        # #     X_test.shape[0], y_test.shape[0]))

        best_clf = clf.fit(X_train_final, Y)
        print("Training Accuracy =%.3f" % best_clf.score(X_train_final, Y))
        y_predicted = best_clf.predict(X_test)

        prediction = []
        ids = []
        for i in range(len(y_predicted)):
            ids.append(id1[i])
            if y_predicted[i] == 0:
                prediction.append("female")
            if y_predicted[i] == 1:
                prediction.append("male")

        twitter_test_data = {}
        twitter_test_data["File_name"] = ids
        twitter_test_data["Prediction"] = prediction
        df = pd.DataFrame(twitter_test_data, columns=['File_name', 'Prediction'])
        fname = 'result/blog_' + '-'.join([feature for feature in app.FEATURES]) + '_' + classifier + '.csv'
        print(fname)
        df.to_csv(fname, index=False, header=False, encoding='utf-8')


def report(le, y_test, y_pred):
        """
        prints the precision, recall, f-score
        print confusion matrix
        print accuracy

        """
        print('---------------------------------------------------------')
        print
        print("Classifation Report")
        print

        target_names = le.classes_
        class_indices = {cls: idx for idx, cls in enumerate(le.classes_)}

        print(metrics.classification_report(y_test, y_pred, target_names=target_names,
                                            labels=[class_indices[cls] for cls in target_names]))

        print("============================================================")
        print("Confusion matrix")
        print("============================================================")
        print(target_names)
        print
        print(confusion_matrix(
            y_test,
            y_pred,
            labels=[class_indices[cls] for cls in target_names]))

        print

        precisions_micro, recalls_micro, fscore_micro, _ = precision_recall_fscore_support(y_test, y_pred,
                                                                                           average='micro',
                                                                                           pos_label=None)
        precisions_macro, recalls_macro, fscore_macro, _ = precision_recall_fscore_support(y_test, y_pred,
                                                                                           average='macro',
                                                                                           pos_label=None)
        precisions_weighted, recalls_weighted, fscore_weighted, _ = precision_recall_fscore_support(y_test, y_pred,
                                                                                                    average='weighted',
                                                                                                    pos_label=None)
        #
        # print
        print('Test Accuracy: %.3f' % accuracy_score(y_test, y_pred))
        #
        #
        # print("Macro Precision Score, %f, Micro Precision Score, %f, Weighted Precision Score, %f" % (
        #     precisions_macro, precisions_micro, precisions_weighted))
        #
        # print("Macro Recall score, %f, Micro Recall Score, %f, Weighted Recall Score, %f" % (
        #     recalls_macro, recalls_micro, recalls_weighted))
        #
        print("Macro F1-score, %f, Micro F1-Score, %f, Weighted F1-Score, %f" % (
            fscore_macro, fscore_micro, fscore_weighted))

        # # print('Misclassified samples: %d' % (y_test != y_pred).sum())
        #
        # print('ROC AUC: %.3f' % roc_auc_score(y_true=y_test, y_score=y_pred))
        #
        # print("============================================================")

if __name__ == "__main__":
    # extract_features()
    # embedding_dict = gensim.models.KeyedVectors.load('/home/niloofar/PycharmProjects/AdvancedNLP/Assign1/BlogWordModel.model')
    # embedding_dict.save_word2vec_format('/home/niloofar/PycharmProjects/AdvancedNLP/Assign1/BlogWordModel' + ".bin", binary=True)
    train('test1', classifier='LR')