import pandas as pd
import ast
import os

def load_symptom_data():
    """Load and process symptom data from the CSV file"""
    try:
        # Read the first row to get the column names
        df = pd.read_csv('attached_assets/Training.csv', nrows=0)
        # Get all column names except the last one (which is the disease)
        all_symptoms = list(df.columns[:-1])
        return all_symptoms
    except Exception as e:
        print(f"Error loading symptom data: {str(e)}")
        return ["itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", 
                "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", 
                "fatigue", "weight_loss", "restlessness", "lethargy", "cough", "high_fever"]

def load_descriptions():
    """Load disease descriptions from the CSV file"""
    try:
        descriptions_df = pd.read_csv('attached_assets/description.csv')
        descriptions = dict(zip(descriptions_df['Disease'], descriptions_df['Description']))
        return descriptions
    except Exception as e:
        print(f"Error loading descriptions: {str(e)}")
        return {}

def load_precautions():
    """Load precautions from the CSV file"""
    try:
        precautions_df = pd.read_csv('attached_assets/precautions_df.csv')
        
        # Create a dictionary with disease as key and a list of precautions as value
        precautions = {}
        for _, row in precautions_df.iterrows():
            disease = row['Disease']
            # Filter out empty precautions
            prec_list = [p for p in [row.get(f'Precaution_{i}', '') for i in range(1, 5)] if isinstance(p, str) and p]
            if disease and prec_list:
                precautions[disease] = prec_list
                
        return precautions
    except Exception as e:
        print(f"Error loading precautions: {str(e)}")
        return {}

def load_medications():
    """Load medications from the CSV file"""
    try:
        medications_df = pd.read_csv('attached_assets/medications.csv')
        
        # Create a dictionary with disease as key and a list of medications as value
        medications = {}
        for _, row in medications_df.iterrows():
            disease = row['Disease']
            med_str = row['Medication']
            
            if disease and isinstance(med_str, str):
                try:
                    # Evaluate the string representation of the list
                    med_list = ast.literal_eval(med_str)
                    medications[disease] = med_list
                except (SyntaxError, ValueError):
                    # If evaluation fails, use the string as is
                    medications[disease] = [med_str]
                    
        return medications
    except Exception as e:
        print(f"Error loading medications: {str(e)}")
        return {}

def load_diets():
    """Load diet recommendations from the CSV file"""
    try:
        diets_df = pd.read_csv('attached_assets/diets.csv')
        
        # Create a dictionary with disease as key and a list of diet recommendations as value
        diets = {}
        for _, row in diets_df.iterrows():
            disease = row['Disease']
            diet_str = row['Diet']
            
            if disease and isinstance(diet_str, str):
                try:
                    # Evaluate the string representation of the list
                    diet_list = ast.literal_eval(diet_str)
                    diets[disease] = diet_list
                except (SyntaxError, ValueError):
                    # If evaluation fails, use the string as is
                    diets[disease] = [diet_str]
                    
        return diets
    except Exception as e:
        print(f"Error loading diets: {str(e)}")
        return {}

def load_workouts():
    """Load workout recommendations from the CSV file"""
    try:
        workouts_df = pd.read_csv('attached_assets/workout_df.csv')
        
        # Create a dictionary with disease as key and a list of workout recommendations as value
        workouts = {}
        for _, row in workouts_df.iterrows():
            disease = row['disease']
            workout = row['workout']
            
            if disease and workout:
                if disease in workouts:
                    workouts[disease].append(workout)
                else:
                    workouts[disease] = [workout]
                    
        return workouts
    except Exception as e:
        print(f"Error loading workouts: {str(e)}")
        return {}

if __name__ == "__main__":
    # Test the functions
    symptoms = load_symptom_data()
    descriptions = load_descriptions()
    precautions = load_precautions()
    medications = load_medications()
    diets = load_diets()
    workouts = load_workouts()
    
    print(f"Loaded {len(symptoms)} symptoms")
    print(f"Loaded {len(descriptions)} descriptions")
    print(f"Loaded {len(precautions)} precautions")
    print(f"Loaded {len(medications)} medications")
    print(f"Loaded {len(diets)} diets")
    print(f"Loaded {len(workouts)} workouts")
