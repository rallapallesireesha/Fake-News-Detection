#!/usr/bin/env python
# coding: utf-8

# # Fake News Detection

# In[87]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


# ## Importing Data

# In[88]:


df = pd.read_csv("train.csv")


# In[89]:


df.head()


# The dataset contains 20800 rows and 5 dimensions

# In[90]:


df.shape


# We have missing values on the title, author, and text columns. We will drop those null values

# In[91]:


df.isnull().sum()


# In[92]:


df.dropna(inplace = True)


# In[93]:


df.shape


# ## Cheking Imbalance

# In[94]:


def create_distribution(feature):
    return sns.countplot(df[feature])


# To visualize the distribution of the label values we need to check the data types of it. As seen below, it is encoded as integer but we will turn it into string

# In[95]:


df.dtypes


# In[96]:


df['label'] = df['label'].astype(str)


# In[97]:


df.dtypes


# In[98]:


create_distribution('label')


# We can see the deleted rows are no longer in the dataframe, for example id 6 is missing. 

# In[99]:


df.head(20)


# We will create a copy of that dataframe and reset the index value.

# In[100]:


news = df.copy()


# In[101]:


news.reset_index(inplace = True)


# In[102]:


news.head(10)


# In[103]:


news.drop(['index', 'id'], axis=1, inplace=True)


# In[104]:


news.head()


# For natural language processing, we need to remove special characters.

# In[105]:


data = news['title'][0]
data


# The below code removes everything except than a-z and A-Z remove with regex

# In[106]:


re.sub('[^a-zA-Z]', ' ', data)


# We need to apply lower case operation

# In[107]:


data = data.lower()
data


# And split the sentence by space

# In[108]:


list = data.split()
list


# ## Applying NLP Techniques

# We will remove the **stopwords** from our word list which are the English words that does not add much meaning to a sentence. 
# 
# They can safely be ignored without sacrificing the meaning of the sentence. 
# 
# For example, the words like the, he, have etc.

# In[109]:


ps = PorterStemmer()


# In[110]:


review = []
for word in list:
    if word not in set(stopwords.words('english')):
        review.append(ps.stem(word))
review


# In[111]:


' ' .join(review)


# As we will apply those operations on each row, we should put them together and apply them to all the values of the title column.

# In[112]:


corpus = []
sentences = []
for i in range(0, len(news)):
    review = re.sub('[^a-zA-Z]', ' ', news['title'][i])
    review = review.lower()
    list = review.split()
    review = [ps.stem(word) for word in list if word not in set(stopwords.words('english'))]
    sentences = ' '.join(review)
    corpus.append(sentences)


# In[113]:


corpus[0]


# In[114]:


corpus


# In[115]:


len(corpus)


# ### Bag of words 

# In[116]:


cv = CountVectorizer(max_features = 5000, ngram_range = (1,3))


# In[117]:


X = cv.fit_transform(corpus).toarray()


# In[118]:


X.shape


# In[119]:


X


# We can see all the words down below

# In[120]:


news.columns


# ## Creating Model

# In[121]:


y = news['label']


# In[122]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)


# In[123]:


X_test


# In[124]:


X_train.shape


# In[125]:


X_test.shape


# In[126]:


y_train.shape


# In[127]:


y_test.shape


# ### Model Training

# In[128]:


classifier = MultinomialNB()


# Train the classifier with training data

# In[129]:


classifier.fit(X_train, y_train)


# Get the predicted values of the classifier for testing data

# In[130]:


pred = classifier.predict(X_test)
pred


# ### Accuracy

# If we look at the accuracy of our model, which is defined as the number of classifications a model correctly predicts divided by the total number of predictions made, it is 89%.

# In[131]:


metrics.accuracy_score(y_test, pred)


# ### Confusion Matrix

# We can use a confusion matrix to visualize and summarize the performance of our classification algorithm.

# In[132]:


cm = metrics.confusion_matrix(y_test, pred)
cm


# In[133]:


cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])


# With the confusion matrix, we can see how many of the values were predicted correctly and how many of them were labeled wrong.

# In[134]:


cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# **Passive Aggressive Classifier** belongs to the category of online learning algorithms in machine learning. It works by responding as passive for correct classifications and responding as aggressive for any miscalculation.
