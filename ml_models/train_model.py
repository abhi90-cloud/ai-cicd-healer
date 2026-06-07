import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
import os
class FailurePredictor:
    def __init__(self):
        self.model = None
        self.model_path = "models/failure_predictor.pkl"
    def generate_training_data(self, n_samples=1000):
        print(f"Generating {n_samples} training samples...")
        np.random.seed(42)
        data = {
            'files_changed': np.random.randint(1, 200, n_samples),
            'lines_added': np.random.randint(10, 5000, n_samples),
            'lines_deleted': np.random.randint(0, 3000, n_samples),
            'test_count': np.random.randint(10, 500, n_samples),
            'dependency_updates': np.random.randint(0, 20, n_samples),
            'build_duration_avg': np.random.uniform(30, 600, n_samples),
            'previous_failures_7d': np.random.randint(0, 10, n_samples),
            'team_size': np.random.randint(1, 20, n_samples),
            'hour_of_day': np.random.randint(0, 24, n_samples),
            'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        }
        df = pd.DataFrame(data)
        # Generate labels (failures)
        failure_prob = (
            (df['files_changed'] > 50) * 0.3 +
            (df['dependency_updates'] > 5) * 0.25 +
            (df['previous_failures_7d'] > 3) * 0.3 +
            (df['build_duration_avg'] > 300) * 0.15
        )
        df['failed'] = (np.random.random(n_samples) < failure_prob).astype(int)
        print(f"Data generated: {df['failed'].sum()} failures, {n_samples - df['failed'].sum()} successes")
        return df
    def train(self, df=None):
        if df is None:
            df = self.generate_training_data()
        X = df.drop('failed', axis=1)
        y = df['failed']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Training XGBoost model...")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Success', 'Failure']))
        # Feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 5 Important Features:")
        print(importance.head())
        return accuracy
    def predict(self, features):
        if self.model is None:
            self.load_model()
        if self.model is None:
            return {"error": "Model not trained"}
        df = pd.DataFrame([features])
        proba = self.model.predict_proba(df)[0][1]
        prediction = proba > 0.5
        return {
            "failure_predicted": bool(prediction),
            "failure_probability": float(proba),
            "confidence": float(max(proba, 1-proba)),
            "risk_level": "high" if proba > 0.7 else ("medium" if proba > 0.3 else "low")
        }
    def save_model(self):
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
            return True
        print("No model found")
        return False
if __name__ == "__main__":
    print("=" * 50)
    print("AI Failure Predictor - Model Training")
    print("=" * 50)
    predictor = FailurePredictor()
    df = predictor.generate_training_data(2000)
    accuracy = predictor.train(df)
    predictor.save_model()
    # Test prediction
    test_features = {
        'files_changed': 75,
        'lines_added': 2000,
        'lines_deleted': 500,
        'test_count': 100,
        'dependency_updates': 8,
        'build_duration_avg': 400,
        'previous_failures_7d': 5,
        'team_size': 8,
        'hour_of_day': 14,
        'is_weekend': 0
    }
    print("\n" + "=" * 50)
    print("Test Prediction:")
    result = predictor.predict(test_features)
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("=" * 50)
