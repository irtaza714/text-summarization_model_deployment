from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            text=str(request.form.get('text'))
        )
        input_text = data.get_input_text()
        print("Before Prediction", input_text)

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(input_text)
        results = 'Summary: {}'.format (results)

        print("After Prediction", results)
        return render_template('home.html', results=results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
