# This is basically the heart of my flask 
from flask import Flask, render_template, request, redirect, url_for
from scipy import sparse
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore")
# import xgboost

#Loading user-user based tf-idf model
with open('model/tf-idf-userbased','rb') as fp:
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

@app.route('/')
def home():
	return render_template('index.html')

def product_recommendation(user_input):
	top20 = user_model.loc[user_input].sort_values(ascending=False)[0:20]
	return top20.index

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

def sentiment_analysis(products):
	sbprs_df = pd.DataFrame(columns=['name', 'sentiment%'])
	for product in products:
		sentiment_df = Pre_Process(product_df[product_df.name == product]['reviews_text'])
		sentiment_df['reviews_text'] = sentiment_df['reviews_text'].apply(lambda text: lemmatize_sentence(text))
		#X_train_transformed = tf_idf_model.transform(product_df[product_df.name == product]['review_text'])
		X_train_transformed = tf_idf_model.transform(sentiment_df['reviews_text'])
		
		y_pred = logit_model.predict(X_train_transformed)
		pos_per = ( len([ele for ele in y_pred if ele == "Positive"]) / len(y_pred) ) *100
		sbprs_df = sbprs_df.append({'name': product, 'sentiment%':pos_per}, ignore_index=True)
	return sbprs_df

@app.route('/predict',methods=['POST'])
def predict():
	uid = request.form.get('uid')
	Input = uid
	prediction = product_recommendation(Input)
	sbprs = sentiment_analysis(prediction)
	top5 = sbprs.sort_values('sentiment%', ascending=False).name[:5]
	return render_template('product.html',username=str(uid), products=list(top5))

app = Flask(__name__)
if __name__ == "__main__":
    app.run()
