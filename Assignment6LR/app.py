from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model/car_price_model.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "symboling": int(request.form["symboling"]),
        "fueltype": request.form["fueltype"],
        "aspiration": request.form["aspiration"],
        "doornumber": request.form["doornumber"],
        "carbody": request.form["carbody"],
        "drivewheel": request.form["drivewheel"],
        "enginelocation": request.form["enginelocation"],
        "wheelbase": float(request.form["wheelbase"]),
        "carlength": float(request.form["carlength"]),
        "carwidth": float(request.form["carwidth"]),
        "carheight": float(request.form["carheight"]),
        "curbweight": float(request.form["curbweight"]),
        "enginetype": request.form["enginetype"],
        "cylindernumber": request.form["cylindernumber"],
        "enginesize": float(request.form["enginesize"]),
        "fuelsystem": request.form["fuelsystem"],
        "boreratio": float(request.form["boreratio"]),
        "stroke": float(request.form["stroke"]),
        "compressionratio": float(request.form["compressionratio"]),
        "horsepower": float(request.form["horsepower"]),
        "peakrpm": float(request.form["peakrpm"]),
        "citympg": float(request.form["citympg"]),
        "highwaympg": float(request.form["highwaympg"]),
        "brand": request.form["brand"]
    }

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Car Price: ${round(prediction[0],2)}"
    )

if __name__ == "__main__":
    app.run(debug=True)