import os
from os import listdir
from bs4 import BeautifulSoup
import pandas as pd

TRAIN_PATH = 'Data/advanced_nlp_2017/blogs_train'
TEST_PATH = 'Data/advanced_nlp_2017/blogs_test'
TWITTER_PATH = 'Data/advanced_nlp_2017/twitter_test'

def extract_documents(input_path):
    path = []
    file_names = []
    documents = []
    labels = []
    j = 0
    for f in listdir(input_path):
        path.append(os.path.join(input_path, f))
        new_path = []
        i = 0
        for f in listdir(path[j]):
            file_names.append(f.replace('.xml', '.txt'))
            new_path.append(os.path.join(path[j], f))
            with open(new_path[i], "r") as f:
                doc = ''
                contents = f.read()
                soup = BeautifulSoup(contents, 'xml')
                texts = soup.find_all('post')
                for tx in texts:
                    doc = doc + tx.text.rstrip().lstrip()
            f.close()
            documents.append(doc.rstrip().lstrip())
            if "female" in new_path[i]:
                labels.append("female")
            else:
                labels.append("male")
            i = i + 1
            print i
        j = j + 1
    return file_names, documents, labels

def extract_blog_test(input_path):
    path = []
    documents = []
    file_names = []
    j = 0
    for f in listdir(input_path):
        file_names.append(f.replace('.xml', '.txt'))
        path.append(os.path.join(input_path, f))
        with open(path[j], "r") as f:
            doc = ''
            contents = f.read()
            soup = BeautifulSoup(contents,'xml')
            texts = soup.find_all('post')
            for tx in texts:
                doc = doc + tx.text.rstrip().lstrip()
        f.close()
        documents.append(doc.rstrip().lstrip())
        print j
        j = j + 1
    return file_names, documents

def extract_twitter_test(input_path):
    path = []
    documents = []
    file_names = []
    j = 0
    for f in listdir(input_path):
        file_names.append(f.replace('.xml', '.txt'))
        path.append(os.path.join(input_path, f))
        with open(path[j], "r") as f:
            doc = ''
            contents = f.read()
            soup = BeautifulSoup(contents,'xml')
            texts = soup.find('documents').find_all('document')
            for tx in texts:
                doc = doc + tx.text.rstrip().lstrip()
        f.close()
        documents.append(doc.rstrip().lstrip())
        print j
        j = j + 1
    return file_names, documents


#train_data = {}
#fname, X_train, Y_train = extract_documents(TRAIN_PATH)
#train_data["File_name"] = fname
#train_data["Document"] = X_train
#train_data["Label"] = Y_train
#df = pd.DataFrame(train_data, columns=['File_name', 'Document', 'Label'])
#df.to_csv('train_data.csv', index=False, encoding='utf-8')


test_data = {}
fname, X_test = extract_blog_test(TEST_PATH)
test_data["File_name"] = fname
test_data["Document"] = X_test
df = pd.DataFrame(test_data, columns=['File_name', 'Document'])
df.to_csv('blog_test_data.csv', index=False, encoding='utf-8')

twitter_test_data = {}
fname, X_twitter_test = extract_twitter_test(TWITTER_PATH)
twitter_test_data["File_name"] = fname
twitter_test_data["Document"] = X_twitter_test
df = pd.DataFrame(twitter_test_data, columns=['File_name', 'Document'])
df.to_csv('twitter_test_data.csv', index=False, encoding='utf-8')

#X_test, Y_test = extract_documents(TEST_PATH)
#raw_data["Document"] = X_train
#raw_data["Label"] = Y_train
#print(len(X_train))
#print(len(Y_train))
#raw_data["Document"] = X_test
#raw_data["Label"] = Y_test
#print(len(X_test))
#print(len(Y_test))
#df = pd.DataFrame(raw_data, columns=['Document', 'Label'])
#df.to_csv('test_data.csv', index=False, encoding='utf-8')

# print(X[10])
# print(Y[10])
# print(X[-1])
# print(Y[-1])

# infile = open(path,"r")
# document = ''
# contents = infile.read()
# soup = BeautifulSoup(contents,'xml')
# titles = soup.find_all('post')
# for title in titles:
#     # print title.text.rstrip()
#     document = document + title.text.rstrip().lstrip() + " "
#
# print(document)
