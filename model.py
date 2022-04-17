from scipy import sparse
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

#******************** Intialization Phase ***********************
#Loading user-user based tf-idf model
with open('model/user-user-corr','rb') as fp:
	user_model = pickle.load(fp)

#Loading Logistic -regression model
with open('model/logit','rb') as fp:
	logit_model = pickle.load(fp)

#Loading data set
with open('data/review_data', 'rb') as fp:
	product_df = pickle.load(fp)

#Loading TF-IDF vectorizer for data transformation
with open('model/tf-idf-vectorizer', 'rb') as fp:
	tf_idf_model = pickle.load(fp)


#************************ Utility Methods ********************************

#It method will clean the review text
def scrub_words(text):
	"""Basic cleaning of texts."""
	#remove non-ascii and digits
	text=re.sub("(\\W|\\d)"," ",text)

	#remove whitespace
	text = text.strip()
	text = re.sub(' +', ' ',text)

	return text

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
	lemmatizer = nltk.stem.WordNetLemmatizer()
	#tokenize the sentence and find the POS tag for each token
	nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
	#tuple of (token, wordnet_tag)
	wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
	lemmatized_sentence = []
	for word, tag in wordnet_tagged:
		if tag is None:
			#if there is no available tag, append the token as is
			lemmatized_sentence.append(word)
		else:
			#else use the tag to lemmatize the token
			lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
	return " ".join(lemmatized_sentence)

#This method will Pre Process the review text
def Pre_Process(sentiments):
	sentiment_df = pd.DataFrame({'reviews_text':sentiments})
	#dropping nan values
	sentiment_df = sentiment_df[~sentiment_df.reviews_text.isna()]
	#converting into string
	sentiment_df['reviews_text'] = sentiment_df['reviews_text'].astype('str')

	# Remove punctuation 
	sentiment_df['reviews_text'] = sentiment_df['reviews_text'].str.replace('[^\w\s]','')

	# Remove Stopwords
	stop = stopwords.words('english')
	sentiment_df['reviews_text'] = sentiment_df['reviews_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
	sentiment_df['reviews_text'] = sentiment_df['reviews_text'].apply(lambda x: scrub_words(x))
	sentiment_df['reviews_text']= sentiment_df['reviews_text'].str.lower()
	sentiment_df['reviews_text'] = sentiment_df['reviews_text'].apply(lambda text: lemmatize_sentence(text))
	return sentiment_df

#This method will predict hte sentiment and will return top5 product.
def sentiment_analysis(products):
	#initializting data frame as product and sentiment %
	sbprs_df = pd.DataFrame(columns=['name', 'sentiment%'])
	#iterating for each product
	for product in products:
		#collecting all reviews from review data and pre processing it
		sentiment_df = Pre_Process(product_df[product_df.name == product]['reviews_text'])
		
		#transforming review's for sentiment analysis.tf-idf vectorizer model is used.
		X_train_transformed = tf_idf_model.transform(sentiment_df['reviews_text'])
		
		#Prediction using logistic regression model
		y_pred = logit_model.predict(X_train_transformed)
		#calculating psitive sentiment %
		pos_per = ( len([ele for ele in y_pred if ele == "Positive"]) / len(y_pred) ) *100
		#adding into dataframe as product and %
		sbprs_df = sbprs_df.append({'name': product, 'sentiment%':pos_per}, ignore_index=True)
	
	#recommending top 5 product based on the postive sentiment %
	return sbprs_df.sort_values('sentiment%', ascending=False).name[:5]

#user user collabarating model is used for product recommendation
def product_recommendation(user_input):
	top20 = user_model.loc[user_input].sort_values(ascending=False)[0:20]
	return top20.index