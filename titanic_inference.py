import pandas as pd
import joblib

# Load trained XGBoost model
MODEL_PATH = "model/AdvancedModeling/BestModel.pkl"
best_model = joblib.load(MODEL_PATH)

def predict_survival(input_data):
    """
    Predict Titanic passenger survival using the trained XGBoost model.
    """
    # Convert input to DataFrame
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    else:
        df_input = pd.DataFrame(input_data)

    # Get expected feature names
    try:
        expected_features = best_model.get_booster().feature_names
        # Reorder columns to match training data
        df_input = df_input.reindex(columns=expected_features, fill_value=0)
    except:
        # If we can't get feature names, proceed with current columns
        pass

    # Make predictions
    predictions = best_model.predict(df_input)
    probabilities = best_model.predict_proba(df_input)[:, 1]

    return pd.DataFrame({
        "Prediction": predictions,
        "Survival_Probability": probabilities
    })

# Example Usage
if __name__ == "__main__":
    passenger = {
        'pclass': 1,
        'fare': 80.5,
        'alone': 1,
        'sex_male': 0,
        'who_man': 0,
        'who_woman': 1,
        'embark_town_Queenstown': 0,
        'embark_town_Southampton': 1
    }

    result = predict_survival(passenger)
    print(result)