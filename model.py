import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Create a custom hybrid model that combines all three classifiers
class HybridModel:
    def __init__(self, svm_model, dt_model, nn_model, scaler):
        self.svm_model = svm_model
        self.dt_model = dt_model
        self.nn_model = nn_model
        self.scaler = scaler
        # Get classes from one of the models
        self.classes_ = svm_model.classes_
        
    def predict(self, X):
        # Get predictions from each model
        svm_pred = self.svm_model.predict(X)
        dt_pred = self.dt_model.predict(X)
        X_scaled = self.scaler.transform(X)
        nn_pred = self.nn_model.predict(X_scaled)
        
        # Combine predictions (simple majority voting)
        pred_array = np.array([svm_pred, dt_pred, nn_pred])
        final_pred = []
        
        for i in range(len(X)):
            # Get all predictions for this sample
            sample_preds = [pred[i] for pred in pred_array]
            # Count occurrences of each prediction
            unique, counts = np.unique(sample_preds, return_counts=True)
            # Get the most common prediction
            majority_vote = unique[np.argmax(counts)]
            final_pred.append(majority_vote)
        
        return np.array(final_pred)
    
    def predict_proba(self, X):
        # This is needed for compatibility with the application
        # We'll use SVM probabilities for simplicity
        return self.svm_model.predict_proba(X)

def create_quick_model():
    """
    Create a quick model with SVC and Decision Tree only
    """
    print("Creating a quick model without neural network...")
    
    try:
        # Load the dataset
        training_data = pd.read_csv('attached_assets/Training.csv')
        
        # Get features (X) and target (y)
        # Assume the last column contains the disease labels
        X = training_data.iloc[:, :-1].values
        y = training_data.iloc[:, -1].values
        
        # Train SVM Classifier
        print("Training SVM classifier...")
        svm_model = SVC(kernel='linear', probability=True, random_state=42)
        svm_model.fit(X, y)
        
        # Train Decision Tree Classifier
        print("Training Decision Tree classifier...")
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X, y)
        
        # Create an ensemble model (voting classifier)
        ensemble_model = VotingClassifier(
            estimators=[
                ('svm', svm_model),
                ('dt', dt_model)
            ],
            voting='hard'
        )
        ensemble_model.fit(X, y)
        
        # Save the quick model
        with open('model.pkl', 'wb') as f:
            pickle.dump(ensemble_model, f)
        
        print("Quick model trained and saved as model.pkl")
        return ensemble_model
    
    except Exception as e:
        print(f"Error creating quick model: {str(e)}")
        model = SVC(probability=True, random_state=42)
        return model

def train_model():
    """
    Train multiple machine learning models (SVC, Decision Tree, Neural Network)
    using the dataset and save the ensemble model as a pickle file.
    Returns the trained model.
    """
    # First, make sure we have a quick model for immediate use
    quick_model = None
    try:
        with open('model.pkl', 'rb') as f:
            quick_model = pickle.load(f)
        print("Existing model found, will enhance it with neural network")
    except FileNotFoundError:
        quick_model = create_quick_model()
    
    print("Training full model with neural network...")
    
    try:
        # Load the dataset
        training_data = pd.read_csv('attached_assets/Training.csv')
        
        # Get features (X) and target (y)
        # Assume the last column contains the disease labels
        X = training_data.iloc[:, :-1].values
        y = training_data.iloc[:, -1].values
        
        # Scale the features for neural network
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train SVM Classifier
        print("Training SVM classifier...")
        svm_model = SVC(kernel='linear', probability=True, random_state=42)
        svm_model.fit(X, y)
        
        # Train Decision Tree Classifier
        print("Training Decision Tree classifier...")
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X, y)
        
        # Train Neural Network Classifier
        print("Training Neural Network classifier...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='constant',
            max_iter=500,
            random_state=42
        )
        nn_model.fit(X_scaled, y)
        
        # We are using the global HybridModel class defined at the top of the file
        
        # Create the hybrid model
        hybrid_model = HybridModel(svm_model, dt_model, nn_model, scaler)
        
        # Save the hybrid model
        with open('model_with_nn.pkl', 'wb') as f:
            pickle.dump(hybrid_model, f)
        
        # Replace the quick model
        with open('model.pkl', 'wb') as f:
            pickle.dump(hybrid_model, f)
        
        print("Full model with neural network trained and saved")
        return hybrid_model
    
    except Exception as e:
        print(f"Error training model: {str(e)}")
        print("Returning existing quick model")
        return quick_model

if __name__ == "__main__":
    create_quick_model()
