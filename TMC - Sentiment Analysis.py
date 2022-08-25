# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:04:24 2022

@author: DELL
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google_play_scraper import app,Sort, reviews_all

rvws= reviews_all(
    'com.meatchop',
    lang='en', # defaults to 'en'
    country='in', # defaults to 'us'
    sort=Sort.MOST_RELEVANT,
    count=100,

)


df_tmc = pd.DataFrame(np.array(rvws),columns=['review'])
df_tmc= df_tmc.join(pd.DataFrame(df_tmc.pop('review').tolist()))

df_tmc.head()

df_tmc.info()
df_tmc['score'].value_counts()

df_tmc['score'].value_counts().plot(kind='pie',figsize=(8,8),autopct='%1.1f%%')

df_review=pd.DataFrame(df_tmc,columns=['content'])
df_review.head()

tmc_reviews=df_review.to_string(index=False)
import nltk
import re
tmc_reviews = re.sub("[^A-Za-z" "]+", " ", tmc_reviews).lower()
tmc_reviews = re.sub("[0-9" "]+"," ", tmc_reviews)

# words that contained in the reviews
tmc_reviews_words = tmc_reviews.split(" ")

tmc_reviews_words = tmc_reviews_words[1:]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(input='tmc_reviews_words',ngram_range=(1,1),smooth_idf=True)
X = vectorizer.fit_transform(tmc_reviews_words)

with open("E:\\DS TRAINING\\stop.txt", "r") as sw:
    stop_words = sw.read()
stop_words = stop_words.split("\n")
stop_words.extend(["tmc","TMC","CHROMPET","product"])

tmc_reviews_words = [w for w in tmc_reviews_words if not w in stop_words]
tmc_rev_string = " ".join(tmc_reviews_words)

from wordcloud import WordCloud
wordcloud_tmc = WordCloud(background_color='Pink',
                      width=1800,
                      height=1400
                     ).generate(tmc_rev_string)
plt.imshow(wordcloud_tmc)

with open("E:\\DS TRAINING\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

tmc_pos_in_pos = " ".join ([w for w in tmc_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(tmc_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)


#NEGATIVE WORDCLOUD
with open("E:\\DS TRAINING\\negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n") 
  
tmc_neg_in_neg = " ".join ([w for w in tmc_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(tmc_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)

#BIGRAM WORDCLOUD

nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS

WNL = nltk.WordNetLemmatizer()
text = tmc_rev_string.lower()

text = text.replace("'", "")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

stopwords_wc = set(STOPWORDS)
customised_words = ['manager','staff'] 

new_stopwords = stopwords_wc.union(customised_words)
text_content = [word for word in text_content if word not in new_stopwords]

text_content = [s for s in text_content if len(s) != 0]
text_content = [WNL.lemmatize(t) for t in text_content]

bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])

# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 100
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=new_stopwords)

wordCloud.generate_from_frequencies(words_dict)
plt.figure(4)
plt.title('Most frequently occurring bigrams of reviews connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()





reviews=df_review.values.tolist()
ip_reviews_string = " ".join(reviews)


with open("Review.txt","w",encoding='utf8') as output:
    output.write(str(df_review))

ip_rev_string=" ".join(df_review)
