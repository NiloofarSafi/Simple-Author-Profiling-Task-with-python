import pandas as pd

import string
import re
from nltk.corpus import stopwords
import enchant
from preprocess import ark_tweet_tokenizer

EN_DICT = enchant.Dict("en_US")

def check_english(string):
    english_words = []
    tokens = ark_tweet_tokenizer(string)
    for token in tokens:
        if EN_DICT.check(token):
            english_words.append(token)
    return " ".join(english_words)


X, Y, doc_id = [], [], []
dict = enchant.Dict("en_US")
df = pd.read_csv('/home/niloofar/PycharmProjects/AdvancedNLP/Assign1/blog_test_data.csv', encoding='utf-8')
for index, row in df.iterrows():
    lower_case = row['Document'].lstrip().rstrip().lower()
    for c in string.punctuation:
        lower_case = lower_case.replace(c,'')
    normal = re.sub(r'[0-9]',"",lower_case)
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    normal = pattern.sub('', normal)
    normal = check_english(normal)
    X.append(normal)
    # Y.append(row['Label'])
    doc_id.append(row['File_name'])
    print(index)

# train_data = {}
# train_data["File_name"] = doc_id
# train_data["Document"] = X
# train_data["Label"] = Y
# df = pd.DataFrame(train_data, columns=['File_name', 'Document', 'Label'])
# df.to_csv('train_data_preprocessed.csv', index=False, encoding='utf-8')


test_data = {}
test_data["File_name"] = doc_id
test_data["Document"] = X
df = pd.DataFrame(test_data, columns=['File_name', 'Document'])
df.to_csv('blog_test_data_preprocessed.csv', index=False, encoding='utf-8')
#


X, Y, doc_id = [], [], []
dict = enchant.Dict("en_US")
df = pd.read_csv('/home/niloofar/PycharmProjects/AdvancedNLP/Assign1/twitter_test_data.csv', encoding='utf-8')
for index, row in df.iterrows():
    lower_case = row['Document'].lstrip().rstrip().lower()
    for c in string.punctuation:
        lower_case = lower_case.replace(c,'')
    normal = re.sub(r'[0-9]',"",lower_case)
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    normal = pattern.sub('', normal)
    normal = check_english(normal)
    X.append(normal)
    # Y.append(row['Label'])
    doc_id.append(row['File_name'])
    print(index)



twitter_test_data = {}
twitter_test_data["File_name"] = doc_id
twitter_test_data["Document"] = X
df = pd.DataFrame(twitter_test_data, columns=['File_name', 'Document'])
df.to_csv('twitter_test_data_preprocessed.csv', index=False, encoding='utf-8')
