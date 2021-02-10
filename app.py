from flask import Flask,render_template,request
import pickle
import numpy as np

cv = pickle.load(open("model/cv.pkl","rb"))
clf = pickle.load(open("model/clf.pkl","rb"))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['post'])
def predict():
    email = request.form.get('email')

    # predict for spam
    X_input = cv.transform([email]).toarray()
    y_pred = clf.predict(X_input)

    if y_pred[0] == 0:
        response = -1
    else:
        response = 1

    return render_template('index.html',response=response)


if __name__ == "__main__":
    app.run(debug=True)