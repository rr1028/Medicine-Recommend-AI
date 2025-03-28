import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler

# Create a custom hybrid model that combines both classifiers
class HybridModel:
    def __init__(self, svm_model, nn_model, scaler):
        self.svm_model = svm_model
        self.nn_model = nn_model
        self.scaler = scaler
        self.classes_ = svm_model.classes_
        
    def predict(self, X):
        # Get predictions from each model
        svm_pred = self.svm_model.predict(X)
        X_scaled = self.scaler.transform(X)
        nn_pred = self.nn_model.predict(X_scaled)
        
        # Combine predictions (simple majority voting)
        pred_array = np.array([svm_pred, nn_pred])
        final_pred = []
        
        for i in range(len(X)):
            sample_preds = [pred[i] for pred in pred_array]
            unique, counts = np.unique(sample_preds, return_counts=True)
            majority_vote = unique[np.argmax(counts)]
            final_pred.append(majority_vote)
        
        return np.array(final_pred)
    
    def predict_proba(self, X):
        return self.svm_model.predict_proba(X)

def create_quick_model():
    print("Creating a quick model without neural network...")
    
    try:
        training_data = pd.read_csv('attached_assets/Training.csv')
        X = training_data.iloc[:, :-1].values
        y = training_data.iloc[:, -1].values
        
        print("Training SVM classifier...")
        svm_model = SVC(kernel='linear', probability=True, random_state=42)
        svm_model.fit(X, y)
        
        ensemble_model = VotingClassifier(
            estimators=[('svm', svm_model)],
            voting='soft'
        )
        ensemble_model.fit(X, y)
        
        with open('model.pkl', 'wb') as f:
            pickle.dump(ensemble_model, f)
        
        print("Quick model trained and saved as model.pkl")
        return ensemble_model
    
    except Exception as e:
        print(f"Error creating quick model: {str(e)}")
        return SVC(probability=True, random_state=42)

def train_model():
    quick_model = None
    try:
        with open('model.pkl', 'rb') as f:
            quick_model = pickle.load(f)
        print("Existing model found, will enhance it with neural network")
    except FileNotFoundError:
        quick_model = create_quick_model()
    
    print("Training full model with neural network...")
    
    try:
        training_data = pd.read_csv('attached_assets/Training.csv')
        X = training_data.iloc[:, :-1].values
        y = training_data.iloc[:, -1].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print("Training SVM classifier...")
        svm_model = SVC(kernel='linear', probability=True, random_state=42)
        svm_model.fit(X, y)
        
        print("Training Neural Network classifier...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='constant',
            max_iter=100,
            random_state=42
        )
        nn_model.fit(X_scaled, y)
        
        hybrid_model = HybridModel(svm_model, nn_model, scaler)
        
        with open('model_with_nn.pkl', 'wb') as f:
            pickle.dump(hybrid_model, f)
        
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
