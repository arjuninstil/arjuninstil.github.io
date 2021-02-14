---
layout: post
title: "Basic Topic Modelling"
subtitle: "Basic topic modelling using 20 newsgroups dataset"
date: 2020-01-10 
background: '/img/posts/Basic-Topic-Modelling/Head.jpg'
---

# Basic Topic Modelling #


## Topic detection ##

Topic detection is a way to extract relevant information from texts. The topic is a set of words (from the text) having particular relevance in terms of probability. They apper to be words that characterize the topics (one or more) discussed in the documents.

**Definitions:**

* Document: A single text, paragraph or even tweet to be classified
* Word/Term: a single component of a document
* Topic: a set of words describing a group (cluster) of documents

**each document usually is as a mixture of several topics**

### Mixture of topics ###
Suppose you have the following set of sentences:

I like to eat broccoli and bananas.
I ate a banana and spinach smoothie for breakfast.
Chinchillas and kittens are cute.
My sister adopted a kitten yesterday.
Look at this cute hamster munching on a piece of broccoli.
A model such as LDA will produce an classification such as the following:

* Sentences 1 and 2: 100% Topic A
* Sentences 3 and 4: 100% Topic B
* Sentence 5: 60% Topic A, 40% Topic B

Topic A: 30% broccoli, 15% bananas, 10% breakfast, 10% munching, … (at which point, you could interpret topic A to be about food)

Topic B: 20% chinchillas, 20% kittens, 20% cute, 15% hamster, … (at which point, you could interpret topic B to be about cute animals)

### Methodologies ###

* latent dirichlet allocation (lda) 
* Non negative matrix factorization
* Clustering 

**latent dirichlet allocation (lda)** 

It's a complex mathematical model (based on Bayesian statistics and Dirichlet and Multinomial distributions) to establish the words in a set of documents that are the most representative. The starting point is definining a 

* fixed number of topics K
* to each topic k we associate a probability p = p(k,w) i.e. the probability of seeing the topic k given the set of words w in the document d
* to each topic k we associate a probability s = s(k,d) i.e. the probability of a k topic belonging to the document d. The distribution s represents the mixture of topics related to d
* A word in the document is picked by randomly extracting from a topic and from a document according to s and p distributions
* An optimization is performed fitting the s,p distributions to the actual distribution of words in the documents.

**Non negative matrix factorization**

* **V**  is the matrix representing all documents
* **H** is the matrix representing documents given the topics
* **W** is the matrix representing the topics


the factorization is made using objective functions such as *Frobenius Norm*

### Main features ###

**LDA**

* Slow method
* Quite accurate for large corpora where each document is a mixture of topics
* Most adopted 

**NMF**

* Fast method
* Accurate with small corpora (i.e. tweets) or tweets with no mixture of topics
* not commonly adopted


## Hands on ##


```python
import string
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import re
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
```


```python
from sklearn.datasets import fetch_20newsgroups
```

**get the corpus**

*20 newsgroup*


```python
newsgroups = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
docs_raw = newsgroups.data
print(len(docs_raw))
```

```python

```


```python
stops_it = stopwords.words('italian')
stops_en = stopwords.words('english')

translator = str.maketrans(' ', ' ', string.punctuation) ## remove the punctuation
```


```python
def minimumSize(tokens,llen = 2):
    ## remove words smaller than llen chars
    tks = []
    for t in tokens:
        if(len(t) > llen):
            tks.append(t)
    return tks

def removeStops(tokens,stops = stops_it):
    # remove stop words
    remains = []
    for t in tokens:
        if(t not in stops):
            remains.append(t)
    return remains

def processText(text):
    ## tokenizer with stop words removal and minimum size 
    tks = word_tokenize(text)
    tks = [t.translate(translator) for t in tks] ## remove the punctuation
    tks = minimumSize(tks)
    tks = removeStops(tks,stops_en)
    return tks
```

### TFIDF vectorizer ###

It transforms each word in the D documents in a sparse matrix representing a normalized frequency of each word in each document. 


```python
n_features = 1000 
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,tokenizer=processText)
```

**n_features** it's the number of individual ters from the corpus to use (notice that rarely a language by humans uses more than few thousands of distinct words ). Having a large dataset it is safe to use large number for n_features, for short corpus n_features must be non large

**max_df** is the probability at which the more probable words must be removed (removes the most common words)

**min_df** removes the words appearing less than 2 times in the dataset.



```python
corpusT = docs_raw[0:500] ## let's use the first 500 documents

tfidf = tfidf_vectorizer.fit_transform(corpusT)
```


```python
tfidf
```




    <500x1000 sparse matrix of type '<class 'numpy.float64'>'
    	with 12953 stored elements in Compressed Sparse Row format>



**associate names (words) to each feature**


```python
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
```

**LDA**


```python
n_topics = 20
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10,
                                learning_method = 'batch')

```


```python
LatentDirichletAllocation(n_topics)
```




    LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                              evaluate_every=-1, learning_decay=0.7,
                              learning_method='batch', learning_offset=10.0,
                              max_doc_update_iter=100, max_iter=10,
                              mean_change_tol=0.001, n_components=20, n_jobs=None,
                              perp_tol=0.1, random_state=None,
                              topic_word_prior=None, total_samples=1000000.0,
                              verbose=0)



* **n_topics** is somehow arbitrary. 
* **max_iter** stops the iteration after maximum 10
* **learning method** is usually online but can be also batch (slower) when all data are processed at time


```python
lda.fit(tfidf) ## fit the model
```




    LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                              evaluate_every=-1, learning_decay=0.7,
                              learning_method='batch', learning_offset=10.0,
                              max_doc_update_iter=100, max_iter=10,
                              mean_change_tol=0.001, n_components=20, n_jobs=None,
                              perp_tol=0.1, random_state=None,
                              topic_word_prior=None, total_samples=1000000.0,
                              verbose=0)




```python
def mostImportantWordsPerTopic(feature_names,topic,n_top_words):
    mwords = []
    sort_topic = topic.argsort()
    mw = sort_topic[:-n_top_words - 1:-1] ## reversed list    
    for idx in mw:
        mwords.append(feature_names[idx])
    return mwords
        

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        
        most_important_words = mostImportantWordsPerTopic(feature_names,topic,n_top_words)

        message = "Topic #%d: " % topic_idx
        message += " ".join(most_important_words)
        print(message)
    print()
```

**Printing the topics**


```python
n_top_words = 10
print_top_words(lda, tfidf_feature_names, n_top_words)
```

    Topic #0: national today defined wrong reference son mail network appreciated information
    Topic #1: simms use machine bbs hardware sort love small congress anyway
    Topic #2: window reply cars write image technology helps windows fine long
    Topic #3: would know one people think could get good use really
    Topic #4: possible yes digital morality tiff audio turn live western entire
    Topic #5: year seen last great first memory problem house believe file
    Topic #6: years water speeds version plus current drugs needed starters faster
    Topic #7: team next small cameras crime large books looks battery rates
    Topic #8: israel launch space would say moon less notes water server
    Topic #9: list questions company got card avoid email also wondering thanks
    Topic #10: armenian less look turkish genocide problems special understood tower color
    Topic #11: program true truth ideas clipper pro read files send gear
    Topic #12: test max talking maybe eternal water playing cut heat assuming
    Topic #13: application jews modem problem regardless insurance reply scope left father
    Topic #14: prevent name abc spacecraft problem radio found land worked sale
    Topic #15: box somebody american neither thank land real force hell anyone
    Topic #16: god two find would someone comes cost information different like
    Topic #17: start windows place controller however research users feel life unless
    Topic #18: print thanks email motif problem reserve ibm shuttle one basically
    Topic #19: things apparently weeks serial received crypto purchased yesterday change whole
    


**NMF**


```python
nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
```

**parameters**

NMF is basically free of parameters :).
    
* alpha : regolarization parameter (used to smooth the frequencies and to improve the fit)
* l1_ratio : regolarization parameter (used to smooth the frequencies and to improve the fit)


```python
n_top_words = 10
print_top_words(nmf, tfidf_feature_names, n_top_words)
```

    Topic #0: would get one think people like know much could may
    Topic #1: thanks please email address list advance information available net anybody
    Topic #2: use simms memory machine mac could several need answer work
    Topic #3: year last yes old three years great game time mask
    Topic #4: problem found used light known check error however think running
    Topic #5: file printer print manager like another port instead name driver
    Topic #6: window box control want left get option application manager upper
    Topic #7: looking card working must email condition mail appreciated buy spend
    Topic #8: problems pain obvious gave anybody following ask sure also cars
    Topic #9: possible yes phone crypto interest invalid fire eternal understanding soviet
    Topic #10: things apparently worse like also little exactly seem basically reality
    Topic #11: post message product real research feel could server sorry error
    Topic #12: program windows files april run software microsoft image code version
    Topic #13: lost services man new think hand take nothing called considered
    Topic #14: anyone find know would information etc hello good like obvious
    Topic #15: land jews appears power jewish man worked right purpose part
    Topic #16: water steam heat hot used oil israel cup engine rather
    Topic #17: armenian turkish genocide armenians xsoviet turks russian muslim people kurds
    Topic #18: send asking reply want following new sale clock included server
    Topic #19: controller esdi ram help ide card scsi need bios appreciated
    


### Bonus visualization ###


```python
import pandas as pd
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
```


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
pyLDAvis.__version__ , pd.__version__
```




    ('2.1.1', '0.24.2')



**default visualization of topics and frequency in a multidimensional space**


```python
p = pyLDAvis.sklearn.prepare(lda, tfidf, tfidf_vectorizer)
pyLDAvis.save_html(p, 'lda.html')
```

<iframe src="/img/posts/Basic-Topic-Modelling/interactive_topic.html" height="800px" width="120%"></iframe>
 