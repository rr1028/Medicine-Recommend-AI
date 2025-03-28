import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, session
import pickle
import numpy as np
from data_processor import load_symptom_data, load_descriptions, load_precautions, load_medications, load_diets, load_workouts

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "healthapp-secret-key")

# Load data
symptoms_data = load_symptom_data()
descriptions = load_descriptions()
precautions = load_precautions()
medications = load_medications()
diets = load_diets()
workouts = load_workouts()

# Load or create the ML model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
    
    # Start background training for neural network model in a separate thread
    import threading
    def enhance_model_with_nn():
        print("Starting enhancement with neural network in background thread...")
        try:
            from model import train_model
            enhanced_model = train_model()
            # Update the global model when neural network training is complete
            global model
            model = enhanced_model
            print("Neural network model training completed successfully!")
        except Exception as e:
            print(f"Error in background neural network training: {str(e)}")
    
    # Start the enhancement thread
    training_thread = threading.Thread(target=enhance_model_with_nn)
    training_thread.daemon = True
    training_thread.start()
    
except FileNotFoundError:
    print("Model file not found. Creating a basic model...")
    from model import create_quick_model
    model = create_quick_model()
    print("Basic model created successfully")

@app.route('/')
def index():
    # Get list of all unique symptoms from the dataset (sorted alphabetically)
    all_symptoms = sorted(symptoms_data)
    return render_template('index.html', symptoms=all_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get selected symptoms from form
        selected_symptoms = request.form.getlist('symptoms')
        
        if not selected_symptoms:
            return render_template('index.html', symptoms=sorted(symptoms_data), 
                                  error="Please select at least one symptom")
        
        # Create input array for model
        input_data = np.zeros(len(symptoms_data))
        for symptom in selected_symptoms:
            if symptom in symptoms_data:
                input_data[symptoms_data.index(symptom)] = 1
        
        # Make prediction
        try:
            prediction = model.predict([input_data])[0]
            # Get disease information
            disease_info = {
                'disease': prediction,
                'description': descriptions.get(prediction, "No description available"),
                'precautions': precautions.get(prediction, ["No specific precautions available"]),
                'medications': medications.get(prediction, ["No specific medications available"]),
                'diet': diets.get(prediction, ["No specific diet recommendations available"]),
                'workout': workouts.get(prediction, ["No specific workout recommendations available"])
            }
            
            return render_template('result.html', info=disease_info, symptoms=selected_symptoms)
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return render_template('index.html', symptoms=sorted(symptoms_data), 
                                  error=f"Error making prediction: {str(e)}")
            
    return render_template('index.html', symptoms=sorted(symptoms_data))

@app.route('/api/symptoms')
def get_symptoms():
    return jsonify(sorted(symptoms_data))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
