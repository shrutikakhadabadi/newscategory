import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
count_vectorizer = pickle.load(open('count_vectorizer.pickle', 'rb'))
nb_classifier = pickle.load(open('nb_classifier.pickle', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #new_data = [str(x) for x in request.form.values()]
    news = str(request.form['news'])
    new_data = [news]
    new_vector =count_vectorizer.transform(new_data)
    pred = nb_classifier.predict(new_vector)

    return render_template('index.html', prediction_text='Category of the news should be {}'.format(pred[0].upper()))



if __name__ == "__main__":
    app.run(debug=True)