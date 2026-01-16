from flask import Flask, render_template, request
from model import predict_digit

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        index = int(request.form["index"])
        prediction = predict_digit(index)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
