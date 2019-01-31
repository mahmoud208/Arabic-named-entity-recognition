
# coding: utf-8

from pathlib import Path
from datetime import datetime as dt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import  accuracy_score
from sklearn import metrics
import sys

path="ANERCorp.xlsx"


xlsx = pd.ExcelFile(path)

df=pd.read_excel(xlsx, header=None)
df = df.drop(2, 1)
df = df.rename(columns={0: 'text', 1: 'label'})



train, test = train_test_split(df, test_size = 0.2)
train_arr = []
test_arr = []
train_lbl = []
test_lbl = []

train_arr=train['text'].astype(str)
train_lbl=train['label'].astype(str)
test_arr=test['text'].astype(str)
test_lbl=test['label'].astype(str)


vectorizer = CountVectorizer()
vectorizer.fit(train_arr)
train_mat = vectorizer.transform(train_arr)


tfidf = TfidfTransformer()
tfidf.fit(train_mat)
train_tfmat = tfidf.transform(train_mat)

test_mat = vectorizer.transform(test_arr)
test_tfmat = tfidf.transform(test_mat)
del df
del test_arr
del train_arr


train_tfmat

lsvm=LinearSVC()
lsvm.fit(train_tfmat,train_lbl)

y_pred_lsvm=lsvm.predict(test_tfmat)

test=['ألمانيا']
test_str = vectorizer.transform(test)
test_tfstr = tfidf.transform(test_str)
test_tfstr.shape
c= lsvm.predict(test_tfstr.toarray())[0]

print (c)


print("accuracy:", metrics.accuracy_score(test_lbl, y_pred_lsvm))

phrase="شاهد أحمد مباراة فرنسا"
arr=phrase.split()


y=[]
token=[]
for x in arr:
    x=[x]
    test_str = vectorizer.transform(x)
    test_tfstr = tfidf.transform(test_str)
    test_tfstr.shape
    token.append(x)
    v= y.append(lsvm.predict(test_tfstr.toarray())[0])

df=pd.DataFrame(list(zip(token,y)),columns=['token','entity_type'])
print (df)


