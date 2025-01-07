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
    predictor = pickle.load(open(r"Models/model_rf.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))
    try:
        genai.configure(api_key="AIzaSyCt6I3_PyyhO8MBRqi7TsFWjxYocFELMME");
        model = genai.GenerativeModel("gemini-1.5-flash");text_input = request.json["text"];response = model.generate_content("Provide the response in this exact numbered format with no additional formatting or extra words: ""1. Good or Bad ""2. 'Stock will go __' up/down ""3. Topic-related stock name with a brief note on how it will perform: "+ text_input)

        # print(response.text)

        response_text = response.text.strip()

        # Split the response by numbered sections
        sections = response_text.split("\n")  # Assuming each item appears on a new line

        # Assign values based on expected numbering
        sentiment = sections[0].replace("1. ", "").strip() if len(sections) > 0 else None
        stock_direction = sections[1].replace("2. Stock will go ", "").strip() if len(sections) > 1 else None
        stock_name_performance = sections[2].replace("3. ", "").strip() if len(sections) > 2 else None

        # Print or log the extracted data (optional)
        print("model result")
        print(f"Sentiment: {sentiment}")
        print(f"Stock Direction: {stock_direction}")
        # print(f"Stock Name and Performance: {stock_name_performance}")

        return jsonify({
            "sentiment": sentiment,
            "stock_direction": "stock will go: "+stock_direction[3:],
            "stock_name_performance": stock_name_performance,
            "full_response": response_text
        })

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
