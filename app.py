from flask import Flask, render_template, request
from utils import predict_disease

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('image')  # Use `get` to safely handle missing file
        if file:
            try:
                result = predict_disease(file)
                return render_template('result.html', result=result)
            except Exception as e:
                return f"Error in prediction: {e}"
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
