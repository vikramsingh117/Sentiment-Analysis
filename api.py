from flask import Flask, request, jsonify, send_file, render_template
import re
import google.generativeai as genai
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)


@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Select the predictor to be loaded from Models folder
    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))
    try:
        genai.configure(api_key="AIzaSyCH4NpJTeQTW93mIbmZ5GWheVN7y5Mq94w");model = genai.GenerativeModel("gemini-1.5-flash");text_input = request.json["text"];response = model.generate_content("follow this strict format: 1.good or bad nextline  2. 'stock will go __ ' what will be the effect on stockprice market stock will go up or stock will go down and by how much assumeing it will impact some stock,thats it as response return whether the news is good or bad: also remove the numbers 1 2 3 just values"+text_input)
        print(response.text)
        # Single string prediction
        print(text_input)
        predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)

        return jsonify({"prediction": response.text})

    except Exception as e:
        return jsonify({"error": str(e)})


def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"


def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"


if __name__ == "__main__":
    app.run(port=5000, debug=True)
