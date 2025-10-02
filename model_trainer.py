import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
import pickle
import json
from collections import Counter
import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedClimateDisasterPredictor:
    """
    NASA Space Apps Challenge 2025 - 9 Disaster Types
    CORRECTED VERSION with Volcanic Eruption included
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        
        self.class_names = [
            'Flood',                    # 0
            'Urban Heat Risk',          # 1
            'Forest Fire',              # 2
            'Deforestation',            # 3
            'Drought',                  # 4
            'Tsunami',                  # 5
            'Landslide Monitoring',     # 6
            'Cyclone/Hurricane',        # 7
            'Volcanic Eruption'         # 8 
        ]
        
        self.label_to_name = {}
        self.class_weights = None
        
        logger.info("=" * 80)
        logger.info("NASA SPACE APPS CHALLENGE 2025")
        logger.info("CLIMATE DISASTER PREDICTION - MODEL TRAINER")
        logger.info("=" * 80)
        logger.info("9 Disaster Types:")
        for i, name in enumerate(self.class_names):
            logger.info(f"  [{i}] {name}")
        logger.info("=" * 80)
    
    def load_processed_data(self, filename="comprehensive_sar_data.npz"):
        """Load comprehensive SAR data"""
        try:
            data = np.load(filename, allow_pickle=True)
            X = data['features']
            y = data['labels']
            disaster_types = data['disaster_types']
            self.feature_names = data['feature_names'].tolist()
            
            if len(X) == 0:
                raise ValueError("No features found in dataset")
            
            
            valid_indices = np.all(np.isfinite(X), axis=1)
            invalid_count = len(X) - np.sum(valid_indices)
            if invalid_count > 0:
                logger.warning(f"Removed {invalid_count} samples with invalid values")
            
            X = X[valid_indices]
            y = y[valid_indices]
            disaster_types = [disaster_types[i] for i in range(len(disaster_types)) if valid_indices[i]]
            
            
            unique_labels = np.unique(y).astype(int)
            self.label_to_name = {str(label): self.class_names[label] 
                                 for label in unique_labels if label < len(self.class_names)}
            
            
            try:
                self.class_weights = compute_class_weight('balanced', classes=unique_labels, y=y)
            except:
                self.class_weights = np.ones(len(unique_labels))
            
            logger.info(f"\nData loaded from {filename}")
            logger.info(f"Total samples: {len(X)}")
            logger.info(f"Feature dimensions: {X.shape[1]}")
            logger.info(f"\nClass Distribution:")
            for label, count in sorted(Counter(y).items()):
                class_name = self.class_names[int(label)] if int(label) < len(self.class_names) else f"Class {label}"
                percentage = count/len(y)*100
                logger.info(f"  [{label}] {class_name}: {count} samples ({percentage:.1f}%)")
            
            return X, y, disaster_types
            
        except FileNotFoundError:
            logger.error(f"\nError: File '{filename}' not found")
            logger.error("Please run 'python data_processor.py' first")
            return None, None, None
        except Exception as e:
            logger.error(f"\nError loading data: {e}")
            return None, None, None
    
    def augment_minority_classes(self, X, y):
        """Data augmentation for minority classes"""
        unique_labels, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        target_count = int(max_count * 0.75)
        
        X_balanced = []
        y_balanced = []
        
        logger.info("\n" + "="*80)
        logger.info("DATA AUGMENTATION")
        logger.info("="*80)
        
        for label in unique_labels:
            X_label = X[y == label]
            y_label = y[y == label]
            
            if len(X_label) == 0:
                continue
            
            if len(X_label) < target_count:
                X_balanced.append(X_label)
                y_balanced.append(y_label)
                
                n_augment = target_count - len(X_label)
                X_augmented = []
                
                for _ in range(n_augment):
                    idx = np.random.randint(0, len(X_label))
                    sample = X_label[idx].copy()
                    noise_level = np.random.uniform(0.03, 0.07)
                    noise = np.random.normal(0, noise_level, sample.shape)
                    augmented = sample * (1 + noise)
                    augmented = np.clip(augmented, X_label.min(axis=0), X_label.max(axis=0))
                    X_augmented.append(augmented)
                
                X_balanced.append(np.array(X_augmented))
                y_balanced.append(np.full(n_augment, label))
                logger.info(f"{self.class_names[int(label)]}: {len(X_label)} -> {target_count} samples")
            else:
                X_balanced.append(X_label)
                y_balanced.append(y_label)
                logger.info(f"{self.class_names[int(label)]}: {len(X_label)} samples (unchanged)")
        
        X_balanced = np.vstack(X_balanced)
        y_balanced = np.hstack(y_balanced)
        logger.info(f"\nTotal balanced dataset: {len(X_balanced)} samples")
        logger.info("="*80)
        return X_balanced, y_balanced
    
    def prepare_data(self, X, y, augment=True):
        """Prepare data with optional augmentation"""
        logger.info("\n" + "="*80)
        logger.info("DATA PREPARATION")
        logger.info("="*80)
        
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if augment and len(X) < 1000:
            logger.info("Applying data augmentation...")
            X, y = self.augment_minority_classes(X, y)
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y, shuffle=True
            )
            logger.info("Stratified split successful")
        except ValueError:
            logger.warning("Stratification failed, using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, shuffle=True
            )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"\nTraining set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        logger.info("="*80)
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def initialize_models(self):
        """Initialize ensemble of models"""
        logger.info("\n" + "="*80)
        logger.info("INITIALIZING MODELS")
        logger.info("="*80)
        
        unique_labels = list(map(int, self.label_to_name.keys()))
        class_weight_dict = {label: weight for label, weight in zip(unique_labels, self.class_weights)}
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=300, max_depth=20, min_samples_split=4,
                min_samples_leaf=2, max_features='sqrt', random_state=42,
                class_weight=class_weight_dict, n_jobs=-1, bootstrap=True
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=250, max_depth=7, learning_rate=0.08,
                subsample=0.85, min_samples_split=4, random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                C=2.5, max_iter=2000, class_weight=class_weight_dict,
                random_state=42, solver='saga', n_jobs=-1, multi_class='multinomial'
            ),
            'SVM': SVC(
                C=1.5, kernel='rbf', gamma='scale', probability=True,
                class_weight=class_weight_dict, random_state=42, cache_size=500
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32), alpha=0.003,
                learning_rate='adaptive', learning_rate_init=0.001,
                random_state=42, max_iter=1000, early_stopping=True,
                validation_fraction=0.15, batch_size=32
            )
        }
        
        logger.info(f"Initialized {len(self.models)} models")
        logger.info("="*80)
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        logger.info("\n" + "="*80)
        logger.info("MODEL TRAINING")
        logger.info("="*80)
        
        results = []
        trained_models = {}
        
        n_classes = len(np.unique(y_train))
        n_splits = min(5, np.bincount(y_train).min())
        n_splits = max(3, n_splits)
        cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        logger.info(f"\n{'Model':<25} {'Accuracy':<12} {'F1-Score':<12} {'CV Mean':<12}")
        logger.info("-" * 61)
        
        for name, model in self.models.items():
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, 
                                          scoring='accuracy', n_jobs=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                results.append({
                    'Model': name, 'Accuracy': accuracy, 'Precision': precision,
                    'Recall': recall, 'F1-Score': f1, 'CV_Mean': cv_scores.mean(),
                    'CV_Std': cv_scores.std()
                })
                
                trained_models[name] = {
                    'model': model, 'predictions': y_pred,
                    'probabilities': model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None,
                    'accuracy': accuracy, 'f1_score': f1
                }
                
                logger.info(f"{name:<25} {accuracy:<12.4f} {f1:<12.4f} {cv_scores.mean():<12.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)[:100]}")
        
        logger.info("="*80)
        return pd.DataFrame(results), trained_models, y_test
    
    def select_best_model(self, results_df, trained_models):
        """Select best model with composite scoring"""
        logger.info("\n" + "="*80)
        logger.info("MODEL SELECTION")
        logger.info("="*80)
        
        if len(results_df) == 0:
            logger.error("No models trained successfully")
            return None, None
        
        results_df['Composite_Score'] = (
            0.30 * results_df['Accuracy'] + 0.35 * results_df['F1-Score'] +
            0.25 * results_df['CV_Mean'] + 0.10 * results_df['Recall']
        )
        results_df['Overfitting_Gap'] = results_df['Accuracy'] - results_df['CV_Mean']
        results_df['Adjusted_Score'] = results_df['Composite_Score'] - 0.8 * results_df['Overfitting_Gap'].clip(lower=0)
        
        best_idx = results_df['Adjusted_Score'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        self.best_model = trained_models[best_model_name]['model']
        
        logger.info(f"\nBest Model: {best_model_name}")
        logger.info(f"Test Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}")
        logger.info(f"F1-Score: {results_df.loc[best_idx, 'F1-Score']:.4f}")
        logger.info(f"CV Score: {results_df.loc[best_idx, 'CV_Mean']:.4f}")
        logger.info("="*80)
        return best_model_name, self.best_model
    
    def detailed_evaluation(self, best_model_name, trained_models, y_test):
        """Detailed evaluation"""
        logger.info(f"\n" + "="*80)
        logger.info(f"DETAILED EVALUATION: {best_model_name}")
        logger.info("="*80)
        
        y_pred = trained_models[best_model_name]['predictions']
        valid_labels = np.unique(y_test)
        valid_class_names = [self.class_names[int(label)] for label in valid_labels]
        
        logger.info("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=valid_class_names, 
                                   zero_division=0, digits=4))
        
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info("="*80)
        return cm
    
    def save_model(self, model_name):
        """Save model, scaler, and metadata"""
        logger.info(f"\n" + "="*80)
        logger.info("SAVING MODEL")
        logger.info("="*80)
        
        try:
            with open('climate_disaster_model.pkl', 'wb') as f:
                pickle.dump(self.best_model, f)
            logger.info("Model saved: climate_disaster_model.pkl")
            
            with open('climate_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info("Scaler saved: climate_scaler.pkl")
            
            metadata = {
                'model_type': model_name,
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'n_features': len(self.feature_names),
                'n_classes': len(self.class_names),
                'training_date': datetime.now().isoformat(),
                'label_mapping': self.label_to_name,
                'version': '3.0-NASA-Space-Apps-2025',
                'description': 'Multi-format SAR Climate Disaster Prediction (9 Types including Volcanic Eruption)'
            }
            
            with open('climate_model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info("Metadata saved: climate_model_metadata.json")
            logger.info("\nAll files saved successfully!")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")

def main():
    logger.info("\n" + "="*80)
    logger.info("NASA SPACE APPS CHALLENGE 2025 - MODEL TRAINING")
    logger.info("9 DISASTER TYPES INCLUDING VOLCANIC ERUPTION")
    logger.info("="*80)
    
    predictor = AdvancedClimateDisasterPredictor()
    X, y, disaster_types = predictor.load_processed_data()
    
    if X is None or len(X) == 0:
        logger.error("\nNo data found. Run 'python data_processor.py' first.")
        return
    
    X_train, X_test, y_train, y_test = predictor.prepare_data(X, y, augment=True)
    predictor.initialize_models()
    results_df, trained_models, y_test = predictor.train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )
    
    if len(trained_models) == 0:
        logger.error("\nNo models trained successfully")
        return
    
    best_model_name, best_model = predictor.select_best_model(results_df, trained_models)
    
    if best_model_name:
        predictor.detailed_evaluation(best_model_name, trained_models, y_test)
        predictor.save_model(best_model_name)
        
        logger.info(f"\n" + "="*80)
        logger.info("TRAINING COMPLETED")
        logger.info("="*80)
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"\nNext: Run 'python app.py' to start the application")
        logger.info("="*80)
    else:
        logger.error("\nFailed to select best model")

if __name__ == "__main__":
    main()
