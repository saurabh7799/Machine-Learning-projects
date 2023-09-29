from flask import Flask, request, jsonify, render_template
import pickle
import json

# Load the trained model
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Load the column names
with open('columns.json', 'r') as f:
    columns = json.load(f)['data_columns']

# Create the Flask application
app = Flask(__name__)

# Define the route for predicting house prices
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    location = data['location']
    sqft = float(data['sqft'])
    bath = int(data['bath'])
    bhk = int(data['bhk'])

    loc_index = columns.index(location.lower())

    x = [0] * len(columns)
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    price = model.predict([x])[0]

    response = {
        'predicted_price': price
    }

    return jsonify(response)

# Define the route for serving the HTML page
@app.route('/')
def home():
    locations = columns[3:]  # Exclude first three elements
    return render_template('index.html', locations=locations)

# Start the Flask application
if __name__ == '__main__':
    app.run()
