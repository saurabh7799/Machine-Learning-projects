from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('Breast_Cancer_Detection.pickle', 'rb'))

@app.route('/')
def home():
    feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                     'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
                     'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                     'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                     'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                     'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
                     'symmetry_worst','fractal_dimension_worst']

    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                     'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
                     'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                     'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                     'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                     'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
                     'symmetry_worst','fractal_dimension_worst']
    
    # Get the feature values from the form
    feature_values = [request.form[feature] for feature in feature_names]
    
    # Check for empty input values
    if any(value == '' for value in feature_values):
        error_message = 'Please fill in all feature values for prediction.'
        return render_template('index.html', features=feature_names, error_message=error_message)
    
    # Convert the feature values to floats
    try:
        feature_values = [float(value) for value in feature_values]
    except ValueError:
        error_message = 'Please provide valid numeric values for all features.'
        return render_template('index.html', features=feature_names, error_message=error_message)
    
    # Create a DataFrame with the input features
    data = dict(zip(feature_names, feature_values))
    df = pd.DataFrame(data, index=[0])
    
    # Make the prediction
    prediction = model.predict(df)
    
    # Convert the prediction to a human-readable label
    if prediction[0] == 0:
        cancer_type = 'Benign'
    else:
        cancer_type = 'Malignant'
    
    # Render the prediction result template with the predicted cancer type
    return render_template('result.html', cancer_type=cancer_type)

if __name__ == '__main__':
    app.run(debug=True)
