from flask import Flask, render_template, request
from prophet import Prophet
from prophet.serialize import model_from_json
import pandas as pd
import json

app = Flask(__name__)

# Load the model once at startup
with open("models/prophet_model.json", "r") as f:
    model = model_from_json(json.load(f))

@app.route("/", methods=["GET"])
def index():
    days = int(request.args.get("days", 90))

    future = model.make_future_dataframe(periods=days)
    future["snap"] = 0  
    forecast = model.predict(future)

    plot_data = forecast[["ds", "yhat"]].tail(days).to_dict(orient="records")

    return render_template("index.html", plot_data=plot_data, days=days)

if __name__ == "__main__":
    app.run(debug=True)