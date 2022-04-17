# This is basically the heart of my flask 
from flask import Flask, render_template, request, redirect, url_for
import warnings
import model
warnings.filterwarnings("ignore")
# import xgboost

app = Flask(__name__)

#************************* app code ***************************************
@app.route('/')
def home():
	#loading home page
	names = model.product_df.reviews_username[:50]
	unames = ' ,  '.join([str(elem) for elem in names])
	return render_template('index.html', users= unames)

@app.route('/predict',methods=['POST'])
def predict():
	#taking username as input
	uid = request.form.get('uid')
	Input = uid
	
	#based on username , top 20 product recommendation
	prediction = model.product_recommendation(Input)

	#sentiment analysis of the product.
	top5 = model.sentiment_analysis(prediction)

	names = model.product_df.reviews_username[:50]
	unames = ' ,  '.join([str(elem) for elem in names])
	#rendering output
	return render_template('product.html',users = unames,username=str(uid), products=list(top5))

if __name__ == "__main__":
    app.run()
