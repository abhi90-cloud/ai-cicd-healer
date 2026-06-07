"""
Advanced ML Pipeline - Ensemble Model with AutoML
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
class AdvancedFailurePredictor:
    """Advanced ML pipeline with multiple models and auto-optimization"""
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.model_path = "models/"
        self.metrics = {}
    def generate_realistic_data(self, n_samples=5000):
        """Generate realistic CI/CD pipeline data"""
        print(f"🧬 Generating {n_samples} realistic pipeline samples...")
        np.random.seed(42)
        # Realistic feature distributions
        data = {
            # Code complexity features
            'files_changed': np.random.lognormal(mean=2.5, sigma=1.0, size=n_samples).astype(int),
            'lines_added': np.random.lognormal(mean=5.0, sigma=1.5, size=n_samples).astype(int),
            'lines_deleted': np.random.lognormal(mean=4.0, sigma=1.5, size=n_samples).astype(int),
            'complexity_score': np.random.beta(2, 5, n_samples),
            # Test features
            'test_count': np.random.randint(10, 1000, n_samples),
            'test_coverage': np.random.beta(5, 2, n_samples) * 100,
            'flaky_test_count': np.random.poisson(lam=2, size=n_samples),
            # Dependency features
            'total_dependencies': np.random.randint(10, 200, n_samples),
            'outdated_dependencies': np.random.randint(0, 30, n_samples),
            'major_version_changes': np.random.poisson(lam=1, size=n_samples),
            'minor_version_changes': np.random.poisson(lam=3, size=n_samples),
            # Build features
            'avg_build_time': np.random.lognormal(mean=4.0, sigma=1.0, size=n_samples),
            'build_queue_time': np.random.exponential(scale=60, size=n_samples),
            'cache_hit_ratio': np.random.beta(8, 2, n_samples),
            # Historical features
            'previous_failures_7d': np.random.poisson(lam=3, size=n_samples),
            'previous_failures_30d': np.random.poisson(lam=8, size=n_samples),
            'success_streak': np.random.geometric(p=0.3, size=n_samples),
            'mean_time_to_recovery': np.random.lognormal(mean=3.0, sigma=1.0, size=n_samples),
            # Team features
            'team_size': np.random.randint(2, 15, n_samples),
            'commits_last_24h': np.random.poisson(lam=10, size=n_samples),
            'review_time_hours': np.random.lognormal(mean=2.0, sigma=1.0, size=n_samples),
            # Temporal features
            'hour_of_day': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'is_weekend': np.random.choice([0, 1], n_samples, p=[0.72, 0.28]),
            'is_business_hours': np.random.choice([0, 1], n_samples, p=[0.35, 0.65]),
        }
        df = pd.DataFrame(data)
        # Clip unrealistic values
        df['files_changed'] = df['files_changed'].clip(0, 500)
        df['lines_added'] = df['lines_added'].clip(0, 10000)
        df['test_coverage'] = df['test_coverage'].clip(0, 100)
        # Generate target with realistic failure patterns
        failure_prob = (
            # Code risk factors
            (df['files_changed'] > 50) * 0.15 +
            (df['complexity_score'] > 0.7) * 0.2 +
            (df['lines_added'] > 1000) * 0.1 +
            # Test risk factors
            (df['test_coverage'] < 60) * 0.25 +
            (df['flaky_test_count'] > 5) * 0.2 +
            # Dependency risk factors
            (df['major_version_changes'] > 2) * 0.2 +
            (df['outdated_dependencies'] > 15) * 0.15 +
            # Historical risk factors
            (df['previous_failures_7d'] > 5) * 0.25 +
            (df['success_streak'] < 3) * 0.15 +
            (df['mean_time_to_recovery'] > 100) * 0.1 +
            # Build risk factors
            (df['avg_build_time'] > 300) * 0.1 +
            (df['cache_hit_ratio'] < 0.3) * 0.1
        )
        # Normalize and add noise
        failure_prob = np.clip(failure_prob + np.random.normal(0, 0.05, n_samples), 0, 1)
        df['failed'] = (np.random.random(n_samples) < failure_prob).astype(int)
        failure_rate = df['failed'].mean() * 100
        print(f"✅ Generated {n_samples} samples | Failure rate: {failure_rate:.1f}%")
        return df
    def create_advanced_features(self, df):
        """Create advanced features from raw data"""
        print("🔧 Engineering advanced features...")
        # Interaction features
        df['change_density'] = df['lines_added'] / (df['files_changed'] + 1)
        df['test_per_file'] = df['test_count'] / (df['files_changed'] + 1)
        df['dependency_health'] = 1 - (df['outdated_dependencies'] / (df['total_dependencies'] + 1))
        df['failure_momentum'] = df['previous_failures_7d'] / (df['previous_failures_30d'] + 1)
        df['team_efficiency'] = df['commits_last_24h'] / (df['team_size'] + 1)
        # Ratio features
        df['add_delete_ratio'] = df['lines_added'] / (df['lines_deleted'] + 1)
        df['build_efficiency'] = df['cache_hit_ratio'] * df['avg_build_time']
        # Rolling features (simulated)
        df['failure_rate_7d'] = df['previous_failures_7d'] / 20
        df['failure_rate_30d'] = df['previous_failures_30d'] / 80
        # Replace infinities
        df = df.replace([np.inf, -np.inf], 0)
        return df
    def train_ensemble(self, df=None):
        """Train multiple models and create ensemble"""
        if df is None:
            df = self.generate_realistic_data()
        df = self.create_advanced_features(df)
        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['failed']]
        X = df[feature_cols]
        y = df['failed']
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print("\n" + "="*60)
        print("🤖 TRAINING ENSEMBLE MODELS")
        print("="*60)
        # Define models
        models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric='logloss',
                random_state=42
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=10,
                min_samples_split=5, min_samples_leaf=2,
                random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42
            ),
        }
        # Train each model
        results = {}
        for name, model in models.items():
            print(f"\n📊 Training {name}...")
            model.fit(X_train, y_train)
            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            accuracy = (y_pred == y_test).mean()
            auc = roc_auc_score(y_test, y_proba)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc
            }
            print(f"   Accuracy: {accuracy:.2%} | AUC: {auc:.3f}")
        # Create voting ensemble
        print("\n🔗 Creating Voting Ensemble...")
        estimators = [(name, results[name]['model']) for name in results]
        self.best_model = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probabilities
            weights=[results[name]['auc'] for name in results]
        )
        self.best_model.fit(X_train, y_train)
        # Final evaluation
        y_pred = self.best_model.predict(X_test)
        y_proba = self.best_model.predict_proba(X_test)[:, 1]
        final_accuracy = (y_pred == y_test).mean()
        final_auc = roc_auc_score(y_test, y_proba)
        self.metrics = {
            'accuracy': final_accuracy,
            'auc': final_auc,
            'individual_models': {name: results[name]['accuracy'] for name in results}
        }
        print("\n" + "="*60)
        print("🏆 FINAL ENSEMBLE RESULTS")
        print("="*60)
        print(f"Accuracy: {final_accuracy:.2%}")
        print(f"AUC-ROC:  {final_auc:.3f}")
        print("\nIndividual Model Performance:")
        for name, res in results.items():
            print(f"  {name:20s}: {res['accuracy']:.2%}")
        # Feature importance from XGBoost
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': results['xgboost']['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n📈 Top 10 Important Features:")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")
        return final_accuracy
    def predict_with_confidence(self, features_dict):
        """Make prediction with confidence intervals"""
        if self.best_model is None:
            self.load_best_model()
        # Convert to DataFrame
        df = pd.DataFrame([features_dict])
        # Get probabilities from all estimators
        probas = []
        for estimator in self.best_model.estimators_:
            proba = estimator.predict_proba(df)[0][1]
            probas.append(proba)
        mean_proba = np.mean(probas)
        std_proba = np.std(probas)
        # Risk level
        if mean_proba > 0.7:
            risk_level = "🔴 HIGH"
        elif mean_proba > 0.3:
            risk_level = "🟡 MEDIUM"
        else:
            risk_level = "🟢 LOW"
        return {
            'failure_predicted': mean_proba > 0.5,
            'probability': float(mean_proba),
            'confidence_interval': [float(mean_proba - 1.96*std_proba), float(mean_proba + 1.96*std_proba)],
            'risk_level': risk_level,
            'model_agreement': float(1 - std_proba),  # How much models agree
            'timestamp': datetime.now().isoformat()
        }
    def save_models(self):
        """Save all models and metadata"""
        os.makedirs(self.model_path, exist_ok=True)
        # Save ensemble
        joblib.dump(self.best_model, f"{self.model_path}/ensemble_model.pkl")
        # Save scaler and feature importance
        joblib.dump(self.scaler, f"{self.model_path}/scaler.pkl")
        self.feature_importance.to_csv(f"{self.model_path}/feature_importance.csv", index=False)
        # Save metrics
        with open(f"{self.model_path}/model_metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✅ Models saved to {self.model_path}/")
    def load_best_model(self):
        """Load saved models"""
        model_file = f"{self.model_path}/ensemble_model.pkl"
        if os.path.exists(model_file):
            self.best_model = joblib.load(model_file)
            print(f"✅ Model loaded from {model_file}")
            return True
        print("❌ No saved model found")
        return False
if __name__ == "__main__":
    print("="*60)
    print("🚀 ADVANCED ML PIPELINE TRAINING")
    print("="*60)
    predictor = AdvancedFailurePredictor()
    # Generate data and train
    df = predictor.generate_realistic_data(10000)
    accuracy = predictor.train_ensemble(df)
    # Save models
    predictor.save_models()
    # Test prediction
    print("\n" + "="*60)
    print("🧪 TESTING PREDICTION")
    print("="*60)
    test_features = {
        'files_changed': 120,
        'lines_added': 2500,
        'lines_deleted': 800,
        'complexity_score': 0.75,
        'test_count': 200,
        'test_coverage': 45.5,
        'flaky_test_count': 8,
        'total_dependencies': 150,
        'outdated_dependencies': 25,
        'major_version_changes': 3,
        'minor_version_changes': 8,
        'avg_build_time': 450,
        'build_queue_time': 120,
        'cache_hit_ratio': 0.25,
        'previous_failures_7d': 8,
        'previous_failures_30d': 20,
        'success_streak': 1,
        'mean_time_to_recovery': 180,
        'team_size': 6,
        'commits_last_24h': 25,
        'review_time_hours': 48,
        'hour_of_day': 16,
        'day_of_week': 3,
        'is_weekend': 0,
        'is_business_hours': 1,
        'change_density': 20.8,
        'test_per_file': 1.67,
        'dependency_health': 0.83,
        'failure_momentum': 0.4,
        'team_efficiency': 4.17,
        'add_delete_ratio': 3.125,
        'build_efficiency': 112.5,
        'failure_rate_7d': 0.4,
        'failure_rate_30d': 0.25
    }
    result = predictor.predict_with_confidence(test_features)
    print("\n📋 Prediction Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("\n" + "="*60)
    print("✅ Advanced ML Pipeline Complete!")
    print("="*60)
