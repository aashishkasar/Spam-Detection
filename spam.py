# import library
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')

#data cleaning

data = pd.read_csv(r"D:\Data Sets\Kaggle\spam.csv", encoding='latin-1')
data.sample()
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis='columns',inplace=True)
data.sample()
data.rename(columns={'v1':'category','v2':'text'},inplace=True)
data.sample(5)

#label encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

# missing values
data.isnull().sum()

# check for duplicate values
data.duplicated().sum()

# remove duplicates
data = data.drop_duplicates(keep='first')
data.duplicated().sum()
data['category'].value_counts()
# Ham --> 0, spam --> 1

# class lable feature variable
f_v=data[["text"]]
c_l=data[["category"]]

f_v.sample()
c_l.sample()

data.shape

#EDA
#distribution spam/non-spam plots

"""count_Class=pd.value_counts(data["category"], sort= True)
count_Class.plot(kind= 'bar', color= ["blue", "orange"])
plt.title('Bar chart')
plt.show()

#In these data set we have more counts of ham which show in 0 label blue color column sms and less number of spam sms shown in lable 1 orange column.

count_Class.plot(kind = 'pie',  autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('')
plt.show()"""

#there is we seen the data 87% of sms is ham or 13% are spam.

#Text Data Preprocessing
stp=stopwords.words("english")

lf=[]
for sent in data["text"]:
    l=[]
    ls=sent.lower()
    ls=re.sub("[^a-zA-Z ]","",ls)
    for word in word_tokenize(ls):
        if word in stp and len(word)<3:
            pass
        else:
            stems=PorterStemmer().stem(word)
            l.append(stems)
    lf.append(" ".join(l))


cd=pd.DataFrame({"cleaned_data":lf})

#Splitting into train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(cd,c_l,train_size=0.8)
x_train.shape

#text to vector
from sklearn.feature_extraction.text import CountVectorizer
CV=CountVectorizer()
dtm=CV.fit_transform(x_train["cleaned_data"])
dtm=pd.DataFrame(dtm.toarray(),columns=sorted(CV.vocabulary_.keys()))

#Multinomial nb algorithm apply
from sklearn.naive_bayes import MultinomialNB  ##multinomial NB is works very well in text
mnb=MultinomialNB()
model=mnb.fit(dtm,y_train)

x_test1=CV.transform(x_test["cleaned_data"])
predicts=model.predict(x_test1)
from sklearn.metrics import accuracy_score
model.predict(CV.transform(cd["cleaned_data"]))

#model building
import joblib
joblib.dump(model,r'D:\Data Sets\ML Models\spamq.joblib')


joblib.dump(CV,r'D:\Data Sets\ML Models\spamcv.joblib')