# Refactored model_trainer.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight
import pickle
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class EnhancedClimateDisasterPredictor:
    """
    Enhanced Climate Disaster Risk Prediction Model
    With Robust Validation and Realistic Performance Expectations
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.class_names = []
        self.label_to_name = {}
        self.class_weights = None
        print("Enhanced Climate Disaster Prediction System Initialized")
    
    def load_processed_data(self, filename="processed_sar_data.npz"):
        """Load preprocessed SAR data with enhanced validation"""
        try:
            data = np.load(filename, allow_pickle=True)
            X = data['features']
            y = data['labels']
            disaster_types = data['disaster_types']
            self.feature_names = data['feature_names'].tolist()
            
            if len(X) == 0:
                raise ValueError("No features found in dataset")
            if len(np.unique(y)) < 2:
                raise ValueError("Insufficient class diversity - need at least 2 classes")
                
            valid_indices = np.all(np.isfinite(X), axis=1)
            X = X[valid_indices]
            y = y[valid_indices]
            disaster_types = [disaster_types[i] for i in range(len(disaster_types)) if valid_indices[i]]
            
            self.check_data_leakage(X, y)
            
            possible_class_names = ['Flood Risk', 'Urban Heat Risk', 'Fire Risk', 'Deforestation Risk']
            unique_labels = np.unique(y).astype(int)
            self.class_names = [possible_class_names[label] for label in unique_labels if label < len(possible_class_names)]
            self.label_to_name = {str(label): possible_class_names[label] for label in unique_labels if label < len(possible_class_names)}
            
            self.class_weights = compute_class_weight('balanced', classes=unique_labels, y=y)
            
            print(f"Data loaded successfully from {filename}")
            print(f"Samples: {len(X)}, Features: {X.shape[1]}")
            print(f"Classes: {self.class_names}")
            print(f"Class distribution: {dict(Counter(y))}")
            
            return X, y, disaster_types
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None
    
    def check_data_leakage(self, X, y):
        """Check for potential data leakage patterns"""
        print("\nPerforming data leakage analysis...")
        
        correlations = []
        for i in range(X.shape[1]):
            if len(np.unique(X[:, i])) > 1:
                corr = np.corrcoef(X[:, i], y)[0, 1]
                if not np.isnan(corr):
                    correlations.append((i, abs(corr)))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        high_corr_features = [idx for idx, corr in correlations if corr > 0.95]
        if high_corr_features:
            print(f"⚠️ WARNING: Features with suspiciously high correlations detected: {high_corr_features}")
            print("This may indicate data leakage or label encoding issues.")
        
        unique_samples = len(np.unique(X, axis=0))
        if unique_samples < len(X) * 0.8:
            print(f"⚠️ WARNING: Many duplicate samples detected ({unique_samples}/{len(X)} unique)")
        
        return len(high_corr_features) == 0
    
    def prepare_data(self, X, y):
        """Prepare data with robust validation"""
        print("\n" + "="*50)
        print("ENHANCED DATA PREPARATION")
        print("="*50)
        
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        class_counts = Counter(y)
        min_class_size = min(class_counts.values())
        if min_class_size < 2:
            print("⚠️ WARNING: Some classes have very few samples. Results may be unreliable.")
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y, shuffle=True
            )
        except ValueError as e:
            print(f"⚠️ Stratification failed: {e}. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, shuffle=True
            )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if np.any(np.std(X_train_scaled, axis=0) < 1e-6):
            print("⚠️ WARNING: Some features have very low variance after scaling.")
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Feature scaling completed")
        
        print("\nClass distribution:")
        for split_name, split_y in [("Training", y_train), ("Test", y_test), ("Overall", y)]:
            unique, counts = np.unique(split_y, return_counts=True)
            print(f"{split_name}:")
            for class_idx, count in zip(unique, counts):
                class_name = self.label_to_name.get(str(class_idx), f"Unknown Label {class_idx}")
                print(f"  {class_name}: {count} ({count/len(split_y)*100:.1f}%)")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def initialize_models(self):
        """Initialize models with regularization to prevent overfitting"""
        print("\nInitializing models with regularization...")
        
        unique_labels = list(map(int, self.label_to_name.keys()))
        class_weight_dict = {label: weight for label, weight in zip(unique_labels, self.class_weights)}
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=8, min_samples_split=10, min_samples_leaf=5,
                max_features='sqrt', random_state=42, class_weight=class_weight_dict, bootstrap=True
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8,
                min_samples_split=10, min_samples_leaf=5, random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0, penalty='l2', random_state=42, max_iter=1000,
                class_weight=class_weight_dict, solver='lbfgs'
            ),
            'Support Vector Machine': SVC(
                C=1.0, gamma='scale', kernel='rbf', random_state=42,
                probability=True, class_weight=class_weight_dict
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=6, min_samples_split=20, min_samples_leaf=10,
                random_state=42, class_weight=class_weight_dict
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(64, 32), alpha=0.01, learning_rate='adaptive',
                random_state=42, max_iter=500, early_stopping=True, validation_fraction=0.1
            )
        }
        print(f"Initialized {len(self.models)} models with regularization")
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate models with robust cross-validation"""
        print("\n" + "="*50)
        print("MODEL TRAINING AND EVALUATION")
        print("="*50)
        
        results = []
        trained_models = {}
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'CV_Score':<10}")
        print("-" * 80)
        
        for name, model in self.models.items():
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                y_pred_proba = None
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except:
                    pass
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                roc_auc = 0
                try:
                    if y_pred_proba is not None and len(np.unique(y_test)) > 2:
                        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                    elif y_pred_proba is not None and len(np.unique(y_test)) == 2:
                        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                except:
                    pass
                
                train_pred = model.predict(X_train)
                train_accuracy = accuracy_score(y_train, train_pred)
                overfitting_gap = train_accuracy - accuracy
                
                if overfitting_gap > 0.15:
                    print(f"⚠️ {name}: Potential overfitting detected (gap: {overfitting_gap:.3f})")
                if cv_std > 0.1:
                    print(f"⚠️ {name}: High CV variance ({cv_std:.3f}) - results may be unstable")
                
                results.append({
                    'Model': name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'ROC-AUC': roc_auc,
                    'CV_Mean': cv_mean,
                    'CV_Std': cv_std,
                    'Train_Accuracy': train_accuracy,
                    'Overfitting_Gap': overfitting_gap
                })
                
                trained_models[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'accuracy': accuracy,
                    'cv_score': cv_mean
                }
                
                print(f"{name:<20} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {cv_mean:<10.4f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
        
        return pd.DataFrame(results), trained_models, y_test
    
    def select_best_model(self, results_df, trained_models):
        """Select best model using multiple criteria"""
        print("\n" + "="*50)
        print("MODEL SELECTION")
        print("="*50)
        
        if len(results_df) == 0:
            print("No models were successfully trained!")
            return None, None
        
        results_df['Composite_Score'] = (
            0.4 * results_df['Accuracy'] +
            0.4 * results_df['CV_Mean'] +
            0.2 * (1 - results_df['Overfitting_Gap'].clip(0, 1))
        )
        
        best_idx = results_df['Composite_Score'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        self.best_model = trained_models[best_model_name]['model']
        
        print(f"Best Model: {best_model_name}")
        print(f"Test Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}")
        print(f"CV Score: {results_df.loc[best_idx, 'CV_Mean']:.4f} ± {results_df.loc[best_idx, 'CV_Std']:.4f}")
        print(f"Overfitting Gap: {results_df.loc[best_idx, 'Overfitting_Gap']:.4f}")
        
        top_3 = results_df.nlargest(3, 'Composite_Score')
        print(f"\nTop 3 Models by Composite Score:")
        for idx, (_, row) in enumerate(top_3.iterrows(), 1):
            print(f"{idx}. {row['Model']}: {row['Composite_Score']:.4f}")
        
        return best_model_name, self.best_model
    
    def detailed_evaluation(self, best_model_name, trained_models, X_test, y_test):
        """Perform detailed evaluation"""
        print(f"\n" + "="*50)
        print(f"DETAILED EVALUATION: {best_model_name}")
        print("="*50)
        
        y_pred = trained_models[best_model_name]['predictions']
        valid_labels = np.unique(y_test)
        valid_class_names = [self.label_to_name.get(str(label), f"Unknown Label {label}") for label in valid_labels]
        
        print("Classification Report:")
        try:
            report = classification_report(y_test, y_pred, target_names=valid_class_names, zero_division=0)
            print(report)
        except Exception as e:
            print(f"Error generating classification report: {e}")
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"{'Predicted':<12}", end="")
        for class_name in valid_class_names:
            print(f"{class_name[:8]:<10}", end="")
        print()
        
        for i, class_name in enumerate(valid_class_names):
            print(f"Actual {class_name[:8]:<5}", end="")
            for j in range(len(valid_class_names)):
                print(f"{cm[i,j] if i < len(cm) and j < len(cm[i]) else 0:<10}", end="")
            print()
        
        self.analyze_feature_importance(trained_models[best_model_name]['model'], X_test, y_test)
        
        return cm
    
    def analyze_feature_importance(self, model, X_test, y_test):
        """Analyze feature importance"""
        print(f"\nFeature Importance Analysis:")
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.feature_names[:len(importance)],
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print(f"Top 10 Most Important Features (Gini Importance):")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"{i+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")
        
        elif hasattr(model, 'coef_'):
            coef = np.abs(model.coef_[0]) if len(model.classes_) == 2 else np.abs(model.coef_).mean(axis=0)
            importance_df = pd.DataFrame({
                'feature': self.feature_names[:len(coef)],
                'importance': coef
            }).sort_values('importance', ascending=False)
            
            print(f"Top 10 Most Important Features (Coefficient Magnitude):")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"{i+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")
        
        else:
            try:
                print(f"Computing permutation importance...")
                result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
                importance_df = pd.DataFrame({
                    'feature': self.feature_names[:len(result.importances_mean)],
                    'importance': result.importances_mean,
                    'std': result.importances_std
                }).sort_values('importance', ascending=False)
                
                print(f"Top 10 Most Important Features (Permutation Importance):")
                for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                    print(f"{i+1:2d}. {row['feature']:<25}: {row['importance']:.4f} ± {row['std']:.4f}")
            except Exception as e:
                print(f"Could not compute permutation importance: {e}")
    
    def optimize_best_model(self, X_train, y_train, best_model_name):
        """Optimize hyperparameters"""
        print(f"\n" + "="*50)
        print(f"HYPERPARAMETER OPTIMIZATION: {best_model_name}")
        print("="*50)
        
        param_grids = {
            'Gradient Boosting': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 4, 5]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 150],
                'max_depth': [6, 8, 10],
                'min_samples_split': [5, 10, 15]
            },
            
            'Support Vector Machine': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l2']
            },
            'Neural Network': {
                'hidden_layer_sizes': [(32, 16), (64, 32), (100, 50)],
                'alpha': [0.001, 0.01, 0.1]
            }
        }
        
        param_grid = param_grids.get(best_model_name, {})
        if param_grid:
            try:
                cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                grid_search = GridSearchCV(
                    self.best_model, param_grid, cv=cv_strategy, scoring='accuracy', n_jobs=-1, return_train_score=True
                )
                
                print("Running hyperparameter optimization...")
                grid_search.fit(X_train, y_train)
                
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best CV score: {grid_search.best_score_:.4f}")
                
                best_train_score = grid_search.cv_results_['mean_train_score'][grid_search.best_index_]
                overfitting_gap = best_train_score - grid_search.best_score_
                if overfitting_gap > 0.1:
                    print(f"⚠️ Warning: Potential overfitting in optimized model (gap: {overfitting_gap:.3f})")
                
                self.best_model = grid_search.best_estimator_
            except Exception as e:
                print(f"Hyperparameter optimization failed: {e}")
                print("Using original model...")
        else:
            print("No hyperparameter grid defined for this model.")
        
        return self.best_model
    
    def save_model(self, model_name):
        """Save model, scaler, and metadata"""
        print(f"\n" + "="*50)
        print("MODEL PERSISTENCE")
        print("="*50)
        
        with open('climate_disaster_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        print("✅ Model saved as 'climate_disaster_model.pkl'")
        
        with open('climate_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("✅ Scaler saved as 'climate_scaler.pkl'")
        
        metadata = {
            'model_type': model_name,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'n_features': len(self.feature_names),
            'n_classes': len(self.class_names),
            'training_date': pd.Timestamp.now().isoformat(),
            'class_weights': self.class_weights.tolist() if self.class_weights is not None else None,
            'label_mapping': self.label_to_name
        }
        
        with open('climate_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("✅ Metadata saved as 'climate_model_metadata.json'")
    
    def predict_disaster_risk(self, features):
        """Predict disaster risk with confidence intervals"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        prediction = self.best_model.predict(features_scaled)[0]
        
        result = {
            'risk_level': self.label_to_name.get(str(prediction), f"Unknown Label {prediction}"),
            'risk_code': int(prediction),
            'confidence': None,
            'probabilities': None
        }
        
        if hasattr(self.best_model, 'predict_proba'):
            try:
                probabilities = self.best_model.predict_proba(features_scaled)[0]
                result['confidence'] = float(max(probabilities))
                prob_dict = {self.label_to_name.get(str(i), f"Unknown Label {i}"): float(prob) for i, prob in enumerate(probabilities)}
                result['probabilities'] = prob_dict
            except Exception as e:
                print(f"Warning: Could not compute probabilities: {e}")
        
        return result

def main():
    """Main training pipeline"""
    print("="*60)
    print("ENHANCED CLIMATE DISASTER PREDICTION MODEL TRAINING")
    print("="*60)
    
    predictor = EnhancedClimateDisasterPredictor()
    X, y, disaster_types = predictor.load_processed_data()
    
    if X is None or len(X) == 0:
        print("❌ Cannot proceed without valid data. Please run data_processor.py first.")
        return
    
    if len(X) < 50:
        print("⚠️ Warning: Very small dataset. Results may not be reliable.")
    if len(np.unique(y)) < 2:
        print("❌ Error: Need at least 2 classes for classification.")
        return
    
    X_train, X_test, y_train, y_test = predictor.prepare_data(X, y)
    predictor.initialize_models()
    results_df, trained_models, y_test = predictor.train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    if len(trained_models) == 0:
        print("❌ No models were successfully trained.")
        return
    
    best_model_name, best_model = predictor.select_best_model(results_df, trained_models)
    if best_model_name is None:
        print("❌ Could not select a best model.")
        return
    
    predictor.detailed_evaluation(best_model_name, trained_models, X_test, y_test)
    predictor.optimize_best_model(X_train, y_train, best_model_name)
    predictor.save_model(best_model_name)
    
    print(f"\n" + "="*50)
    print("FINAL MODEL VALIDATION")
    print("="*50)
    
    if len(X_test) > 0:
        test_indices = np.random.choice(len(X_test), min(3, len(X_test)), replace=False)
        for i, test_idx in enumerate(test_indices):
            test_sample = X_test[test_idx]
            actual_label = predictor.label_to_name.get(str(y_test[test_idx]), f"Unknown Label {y_test[test_idx]}")
            try:
                prediction_result = predictor.predict_disaster_risk(test_sample)
                print(f"\nTest Sample {i+1}:")
                print(f"  Actual: {actual_label}")
                print(f"  Predicted: {prediction_result['risk_level']}")
                if prediction_result['confidence']:
                    print(f"  Confidence: {prediction_result['confidence']:.3f}")
                if prediction_result['probabilities']:
                    print("  Class Probabilities:")
                    for class_name, prob in prediction_result['probabilities'].items():
                        print(f"    {class_name}: {prob:.3f}")
            except Exception as e:
                print(f"  ❌ Prediction failed: {e}")
    
    print(f"\n" + "="*60)
    print("ENHANCED MODEL TRAINING COMPLETED")
    print("="*60)
    
    best_result = results_df[results_df['Model'] == best_model_name].iloc[0]
    print(f"Final Model: {best_model_name}")
    print(f"Test Accuracy: {best_result['Accuracy']:.4f}")
    print(f"Cross-Validation Score: {best_result['CV_Mean']:.4f} ± {best_result['CV_Std']:.4f}")
    

if __name__ == "__main__":
    main()