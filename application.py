from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        required_fields = [
            'model', 'seller_type', 'fuel_type', 'transmission_type',
            'vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats'
        ]
        if not all(request.form.get(field) for field in required_fields):
            return render_template('index.html', results='Please fill in all fields.')

        data = CustomData(
            model=request.form.get('model'),
            seller_type=request.form.get('seller_type'),
            fuel_type=request.form.get('fuel_type'),
            transmission_type=request.form.get('transmission_type'),
            vehicle_age=float(request.form.get('vehicle_age')),
            km_driven=float(request.form.get('km_driven')),
            mileage=float(request.form.get('mileage')),
            engine=float(request.form.get('engine')),
            max_power=float(request.form.get('max_power')),
            seats=float(request.form.get('seats')),
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        formatted_result = f"{results[0]:,.0f}"
        return render_template('index.html', results=formatted_result)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
