from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("model/avocado_model.pkl")

# TYPE LIST
type_classes = ['conventional', 'organic']

# REGION LIST SESUAI DATASET ASLI
region_classes = [
    'Albany', 'Atlanta', 'BaltimoreWashington', 'Boise', 'Boston',
    'BuffaloRochester', 'California', 'Charlotte', 'Chicago',
    'CincinnatiDayton', 'Columbus', 'DallasFtWorth', 'Denver',
    'Detroit', 'GrandRapids', 'HarrisburgScranton', 'HartfordSpringfield',
    'Houston', 'Indianapolis', 'Jacksonville', 'LasVegas', 'LosAngeles',
    'Louisville', 'MiamiFtLauderdale', 'Milwaukee', 'Minneapolis',
    'Nashville', 'NewOrleansMobile', 'NewYork', 'NorthernNewEngland',
    'Orlando', 'Philadelphia', 'PhoenixTucson', 'Pittsburgh',
    'Portland', 'RaleighGreensboro', 'RichmondNorfolk', 'Roanoke',
    'Sacramento', 'SanDiego', 'SanFrancisco', 'Seattle', 'Spokane',
    'StLouis', 'Syracuse', 'Tampa', 'TotalUS', 'West', 'WestTexNewMexico'
]

# Mapping label encoder manual
type_mapping = {type_classes[i]: i for i in range(len(type_classes))}
region_mapping = {region_classes[i]: i for i in range(len(region_classes))}

# Fitur model dalam urutan training
model_features = [
    "Total Volume",
    "4046",
    "4225",
    "4770",
    "Total Bags",
    "Small Bags",
    "Large Bags",
    "XLarge Bags",
    "type",
    "year",
    "region"
]



@app.route("/")
def index():
    return render_template("index.html", types=type_classes, regions=region_classes)

@app.route("/predict", methods=["POST"])
def predict():
    
    # Encode type & region
    type_text = request.form["type"]
    region_text = request.form["region"]

    type_encoded = type_mapping[type_text]
    region_encoded = region_mapping[region_text]

    # Numeric input
    input_data = {
        "Total Volume": float(request.form["Total Volume"]),
        "4046": float(request.form["4046"]),
        "4225": float(request.form["4225"]),
        "4770": float(request.form["4770"]),
        "Total Bags": float(request.form["Total Bags"]),
        "Small Bags": float(request.form["Small Bags"]),
        "Large Bags": float(request.form["Large Bags"]),
        "XLarge Bags": float(request.form["XLarge Bags"]),
        "year": float(request.form["year"]),
        "type": type_encoded,
        "region": region_encoded
    }

    df_input = pd.DataFrame([input_data], columns=model_features)

    prediction = model.predict(df_input)[0]
    prediction = round(prediction, 3)

    return render_template("result.html", prediction=prediction, data=input_data)

if __name__ == "__main__":
    app.run(debug=True)
