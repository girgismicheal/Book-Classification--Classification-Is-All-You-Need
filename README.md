# Book-Classification--Classification-Is-All-You-Need

# Gutenberg Project Overview
**Project Gutenberg is a library of over 60,000 free eBooks**

![Gutenberg](https://drive.google.com/uc?export=view&id=1bOd8Hiv-sU8Skj1gYR-2cxLUEBIretyZ)


In this project, we selected some books from the Gutenburg library from different categories and then select random paragraphs from them and labeled these paragraphs by the book name for ground truth. After creating the dataset we used many transformation algorithms to embed the text to numbers for the modeling processes like (Fast-text,BERT, TF_IDF, BOW, Skip gram,Glove,LDA,Word2Vec, Doc2Vec)
<br><br>
After this, we tried many classification algorithms like(SVM, KNN, Decision Tree, GaussianNB, and BERT) and chose the champion one which achieved the highest accuracy.

**Recommended using GPU to compile the code much faster.
But it works well for CPU too.**
- GPU takes around 40 min, while CPU may take hours.

 

# Project Methodology

![Gutenberg](https://drive.google.com/uc?export=view&id=1-YX8_vqTOSjudFe5AiovLCDzA6SEOhLm)

# Project Main Steps


- [Data Exploration](#1)
- [Data Preprocessing](#2)
- [Word Embedding](#3)
  - [BOW](#4)
  - [TF-IDF](#5)
  - [Doc2Vec](#6)
  - [Bert Embedding](#7)
  - [Glove](#8)
  - [Fast text](#9)
  - [Word2Vec](#10)
  - [LDA (Linear Discriminant Analysis)](#11)
- [Word embedding dictionary](#12)
- [Training](#13)
- [BERT classifier](#14)
- [Validation](#15)
- [Error analysis](#16)
  - [Reduce the samples' word number](#17)
  - [Word Counts](#18)
- [Grid Search](#19)
- [Conclusion](#20)

# <a name="1">Data Exploration</a>
**By discovering the books’ content as shown below:**
> Moby Dick by Herman Melville 1851]\r\n\r\n\r\nETYMOLOGY.\r\n\r\n(Supplied by a Late Consumptive Usher to a Grammar School)\r\n\r\nThe pale Usher--threadbare in coat, heart, body, and brain; I see him\r\nnow.  He was ever dusting his old lexicons and grammars, with a queer\r\nhandkerchief, mockingly embellished with all the gay flags of all the\r\nknown nations of the world.  He loved to dust his old grammars; it\r\nsomehow mildly reminded him of his mortality.\r\n\r\n"While you take in hand to school others, and to teach them by what\r\nname a whale-fish is to be called in our tongue leaving out, through\r\nignorance, the letter H, which almost alone maketh the signification\r\nof the word, you deliver that which is not true." --HACKLUYT\r\n\r\n"WHALE. ... Sw. and Dan. HVAL.  This animal is named from roundness\r\nor rolling; for in Dan. HVALT is arched or vaulted." --WEBSTER\'S\r\nDICTIONARY\r\n\r\n"WHALE. ... It is more immediately from the Dut. and Ger. WALLEN;\r\nA.S. WALW-IAN, to roll, to wallow." --RICHARDSON\'S DICTIONARY\r\n\r

- Many problems have been found in books' content, so we should deal with them.

# <a name="2">Data Preprocessing</a>

**Clean the content of the books by:**
- Removing the word capitalization, unwanted characters, white spaces, and stop words.
- Replacing some patterns.
- Applying lemmatization and tokenization.

**The data after cleaning process**
> delight steelkilt strain oar stiff pull harpooneer get fast spear hand radney sprang bow always furious man seem boat bandage cry beach whale topmost back nothing loath bowsman haul blinding foam blent two whiteness together till sudden boat struck sunken ledge keel spill stand mate instant fell whale slippery back boat right dash aside swell radney toss sea flank whale struck spray instant dimly see veil wildly seek remove eye moby dick whale rush round sudden maelstrom seize swimmer jaw rear high plunge headlong go meantime first tap boat bottom lakeman slacken line drop astern whirlpool calmly look thought thought

**Dataset Building**

![image](/Image/Screenshot_1.png)

- Create a data frame containing 2 columns and 1000 rows representing the books' samples (Sample) and the book name (Label)

**Note:** Before starting to transform words. We split the data into training and testing, to prevent data leakage.

# <a name="3">Word Embedding</a>
It is one of the trivial steps to be followed for a better understanding of the context of what we are dealing with. After the initial text is cleaned and normalized, we need to transform it into features to be used for modeling.

We used some methods to assign weights to particular words, sentences, or documents within our data before modeling them. We go for numerical representation for individual words as it’s easy for the computer to process numbers.

  ## <a name="4">BOW</a>
A bag of words is a representation of text that describes the occurrence of words within a document, just keeps track of word counts and disregard the grammatical details and the word order. As we said that we split the data. So, we applied BOW to training and testing data. So, it transforms each sentence into an array of occurrences in this sentence.
```Python
from sklearn.feature_extraction.text import CountVectorizer

BOW = CountVectorizer()
BOW_train = BOW.fit_transform(X_train)
BOW_test = BOW.transform(X_test)
```
**Important Note:** We Applied the Linear discriminant analysis (LDA) on Bow to reduce its dimensions. as it's vector shape very large.
```Python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA()
lda_train = lda.fit_transform(BOW_train.toarray(), y_train)
lda_test = lda.transform(BOW_test.toarray())
lda_test.shape
```


## <a name="5">TF-IDF</a>
  TF-IDF (term frequency-inverse document frequency) is a statistical measure that evaluates how relevant a word is to a document in a collection of documents. This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents.
  <br><br>In addition, to understand the relation between each consecutive pair of words, tfidf with bigram has applied. Furthermore, we applied tfidf with trigram to find out whether there is a relation between each consecutive three words.
- In the project, used Uni-gram and Bi-gram  
```Python
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_ngram(n_gram,X_train=X_train,X_test=X_test):
    vectorizer = TfidfVectorizer(ngram_range=(1,n_gram))
    x_train_vec = vectorizer.fit_transform(X_train)
    x_test_vec = vectorizer.transform(X_test)
    return x_train_vec,x_test_vec
# Uni-Gram
X_trained1g_cv,X_test1g_cv = tfidf_ngram(1,X_train=X_train,X_test=X_test)
# Bi-Gram
X_trained2g_cv,X_test2g_cv = tfidf_ngram(2,X_train=X_train,X_test=X_test)
```

## <a name="6">Doc2Vec</a>
- Doc2Vec is a method for representing a document as a vector and is built on the word2vec approach.
- I have trained a model from scratch to embed each sentence or paragraph of the data frame as a vector of 50 elements.
```Python
#Import packages
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

def get_doc2vec_vector(df):
    # Tokenization of each document
    tokenized_doc = []
    for d in df['Sample of the book']:
        tokenized_doc.append(word_tokenize(d.lower()))

    # Convert tokenized document into gensim formated tagged data
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
    model = Doc2Vec(tagged_data, vector_size=50, window=2, min_count=1, workers=4, epochs = 100)

    doc2vec_vectors=[]
    for sentence in df['Sample of the book']:
        doc2vec_vectors.append(model.infer_vector(word_tokenize(sentence.lower())))
    return doc2vec_vectors

train_doc2vec_vectors=get_doc2vec_vector(df_train)
test_doc2vec_vectors=get_doc2vec_vector(df_test)
```

## <a name="7">Bert Embedding</a>
Bert can be used as a word embedding pretrained model and then use these embedded vectors to train another model like SVM or Naive bayes and DL models like RNN or LSTM 

- BERT (Bidirectional Encoder Representations from Transformers) is a highly complex and advanced language model that helps people automate language understanding.
- BERT is the encoder of transformers, and it consists of 12 layers in the base model, and 24 layers for the large model. So, we can take the output of these layers as an embedding vector from the pre-trained model.
- There are three approaches to the embedding vectors: concatenate the last four layers, the sum of the last four layers, or embed the full sentence by taking the mean of the embedding vectors of the tokenized words
- As the first two methods require computational power, we used the third one which takes the mean of columns of each word and each word is represented as a 768x1 vector. so, the whole sentence at the end is represented as a 768x1 vector

## Helper Function
  This help function build to pass the data through the models Glove, Fast-text, and Word2vec model and return the embedding vectors.
```Python
def get_vectors_pretrained(df, model):
    embedding_vectors = []
    for partition in df['Sample of the book']:
        sentence = []
        for word in partition.split(' '):
            try:
                sentence.append(model[word])
            except:
                pass
        sentence = np.array(sentence)
        sentence = sentence.mean(axis=0)
        embedding_vectors.append(sentence)
    embedding_vectors = np.array(embedding_vectors)
    return embedding_vectors
```

  ## <a name="8">Glove</a>
- Global vector for word representation is an unsupervised learning algorithm for word embedding.
- We trained a GloVe model on books’ data, that represents each word in a 300x1 Vector. We took the data frame after cleaning and get each paragraph and passed it to the corpus. After that, we trained the model on each word.
- We used also a pre-trained model “glove-wiki-gigaword-300”. Each word is represented by a 300x1 vector. Then, on each word of a sentence in the data frame, we replaced it with its vector representation.
```Python
import gensim.downloader as api
glove_model = api.load("glove-wiki-gigaword-300")  # load glove vectors
train_glove_embedding_vectors=get_vectors_pretrained(df_train,glove_model)
test_glove_embedding_vectors=get_vectors_pretrained(df_test,glove_model)
print(train_glove_embedding_vectors.shape)
```

  ## <a name="9">Fast text</a>
- FastText is a library for learning word embeddings and text classification. The model allows one to create unsupervised learning or supervised learning algorithms for obtaining vector representations for words.
- We loaded a pre-trained model from genism API ‘fasttext-wiki-news-subwords-300’.
```Python
import gensim.downloader as api
fast_text_model = api.load("fasttext-wiki-news-subwords-300")
train_fast_text_embedding_vectors=get_vectors_pretrained(df_train,glove_model)
test_fast_text_embedding_vectors=get_vectors_pretrained(df_test,glove_model)
print(train_fast_text_embedding_vectors.shape)
```

  ## <a name="10">Word2Vec</a>
- Word2vec is a method to represent each word as a vector.
- Used a pre-trained model “word2vec-google-news-300”.
```Python
import gensim.downloader as api
word2vec_model = api.load("word2vec-google-news-300") 
train_word2vec_embedding_vectors= get_vectors_pretrained(df_train,word2vec_model)
test_word2vec_embedding_vectors= get_vectors_pretrained(df_test,word2vec_model)
print(train_word2vec_embedding_vectors.shape)
```


# <a name="12">Word embedding dictionary</a>
> Saved the word Embedding as a pickle file 
for future use, the embedding takes a long time. So, if you want to start directly with the embedded data that has been computed previously.

The file dictionary:
```Python
text_embedding={
    'BOW':(BOW_train,BOW_test),
    'TF_IDF 1_gram':(X_trained1g_cv,X_test1g_cv),
    'TF_IDF 2_gram':(X_trained2g_cv,X_test2g_cv),
    'Doc2vec':(train_doc2vec_vectors,test_doc2vec_vectors),
    'LDA' :(lda_train,lda_test),

    'Glove':(train_glove_embedding_vectors,test_glove_embedding_vectors),
    'Word2vec':(train_word2vec_embedding_vectors,test_word2vec_embedding_vectors),
    'BERT Model':(BERT_embedding_vectors_train,BERT_embedding_vectors_test),
}
```
- File name is ""Embedding_Vectors_Classification.pkl"



# <a name="13">Training</a>
**Trained Models**
> - SVM is a supervised machine learning algorithm that separates classes using hyperplanes.
> - Gaussian NB is special type of Naïve Bayes algorithm that perform well on continuous data. 
> - KNN is a non-parametric supervised algorithm. Despite its simplicity it can be highly competitive in NLP applications. 
> - Decision Tree uses a tree-like model in Training phase to take a decision and studying its consequences.
- So, we have 32 models on all of our transformation methods.

**Choosing the Champion model and champion embedding method**
![image](/Image/Screenshot_4.png)

- From the results above, the best accuracy comes from the SVC classifier trained on the TF-IDF uni-gram embedding vector and the achieving training accuracy is 100% and testing accuracy is 99.5%.



# <a name="14">BERT classifier</a>
- BERT can be used as the main classifier by fine-tuning the model on our dataset.
- Used implemented BERT class from hugging face library which called transformers.

The Bert results are 100% training accuracy, and 88% testing, So, the SVM is still the champion model.

# <a name="15">Validation</a>
We applied 10-folds cross validation to estimate the skills of our machine learning models on different combinations of validation and training datasets.
## Cross validation on the champion model

# <a name="16">Error analysis</a>
 ## <a name="17">Reduce the samples' word number</a>
We reduced the number of words in each sentence to test if the accuracy of the champion model will decrease, increase or will still the same.

| Number_of_samples | Testing_Accuracy |
|-------------------|------------------|
| 70                | 0.95             |
| 50                | 0.9              |
| 40                | 0.885            |
| 30                | 0.86             |
| 20                | 0.765            |

We noticed that the accuracy decreased by decreasing number of words in each partition and this make sense because the model can’t classify which class when number of words(features) is small.
 ## <a name="18">Word Counts</a>
| word    | Example | Wrong | Correct   | Wrong is greater than |
|---------|---------|-------|-----------| --------------------- |
| catch   | 1       | 21    | 19        | 1                     |
| pea     | 1       | 2     | 4         | 0                     |
| roll    | 1       | 110   | 19        | 1                     |
| part    | 1       | 200   | 81        | 1                     |
| shop    | 3       | 5     | 41        | 0                     |
| ...     | ...     | ...   | ...       | ...                   |
| stop    | 1       | 50    | 105       | 0                     |
| stay    | 2       | 18    | 56        | 0                     |
| stand   | 1       | 218   | 81        | 1                     |
| still   | 1       | 312   | 90        | 1                     |
| sir     | 1       | 176   | 290       | 0                     |

- At first, we explored the weights of word examples in correct and wrong book classification to make sure that everything is working fine.


# <a name="19">Grid Search On the champion model</a>
```Python 
from sklearn.model_selection import GridSearchCV
param={
    'kernel' :['rbf','linear'],
    'C':[10, 1, .1, .01, .001],
}
clf=GridSearchCV (model_cv,param_grid=param)
clf.fit(X_trained1g_cv.toarray(),y_train)
```
- We found that the best hyperparameter is a linear kernel with a regularization coefficient equal to 1.


# <a name="20">Conclusion</a>
To wrap up, we made 32 model on different transformation methods and it was obvious that SVM perform better than other models, this is because the data is slightly few, and SVM performs better when data is small. When comparing transformation methods, it clear that TF-IDF uni-gram is trained better in most of the models, because as the length of n-grams increase, the frequency of finding this n-grams again decreases.
