# Refactored app.py
import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
import json
import logging
from datetime import datetime
from PIL import Image as PilImage
import cv2
from werkzeug.utils import secure_filename
import os

# Import enhanced SARDataProcessor from the updated data_processor.py
from data_processor import SARDataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB for larger SAR files
ALLOWED_EXTENSIONS = {'jpg', 'png', 'tif', 'tiff', 'filt'}  # .filt for GOES data

class EnhancedClimateDisasterApp:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.processor = SARDataProcessor()
        self.load_models()
    
    def load_models(self):
        """Load trained model and preprocessing components with enhanced error handling"""
        try:
            with open('climate_disaster_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Enhanced climate disaster model loaded successfully")
            
            with open('climate_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Feature scaler loaded successfully")
            
            with open('climate_model_metadata.json', 'r') as f:
                self.metadata = json.load(f)
            logger.info("Model metadata loaded successfully")
            
            logger.info(f"Model type: {self.metadata.get('model_type', 'Unknown')}")
            logger.info(f"Number of features: {self.metadata.get('n_features', 0)}")
            logger.info(f"Number of classes: {self.metadata.get('n_classes', 0)}")
            
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            logger.error("Please run the updated model_trainer.py first to train the models")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def allowed_file(self, filename):
        """Check if file extension is allowed"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def read_special_format(self, file_path):
        """Read special SAR formats like .filt files"""
        try:
            if file_path.suffix.lower() == '.filt':
                return self.processor.read_filt_file(file_path)
            else:
                img = self.load_image(file_path)
                return img
        except Exception as e:
            logger.error(f"Error reading special format file: {e}")
            return None
    
    def load_image(self, file_path):
        """Load image using PIL first, fallback to cv2"""
        try:
            pil_img = PilImage.open(str(file_path))
            img = np.array(pil_img.convert('L'))
            return img
        except Exception as e:
            logger.warning(f"PIL failed for {file_path}: {e}. Falling back to cv2.")
            try:
                img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                return img
            except Exception as e:
                logger.error(f"cv2 failed for {file_path}: {e}")
                return None
    
    def process_image(self, file, data_type='forest'):
        """Enhanced SAR image processing with support for multiple formats"""
        try:
            filename = secure_filename(file.filename)
            if not self.allowed_file(filename):
                raise ValueError("Invalid file type. Only jpg, png, tif, tiff, filt allowed.")
            
            temp_path = f"temp_{filename}"
            file.save(temp_path)
            
            try:
                if filename.lower().endswith('.filt'):
                    img = self.processor.read_filt_file(temp_path)
                else:
                    img = self.load_image(temp_path)
                
                if img is None:
                    raise ValueError("Could not load image. Please check file format.")
                
                img = cv2.resize(img, (256, 256))
                
                features = self.processor.extract_sar_features(img, data_type)
                
                return np.array(features)
                
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return None
    
    def predict_risk(self, features):
        """Make enhanced disaster risk prediction with confidence scoring"""
        if self.model is None or self.scaler is None:
            return None
        
        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            expected_features = self.metadata.get('n_features', 15)
            if features.shape[1] != expected_features:
                logger.warning(f"Feature count mismatch: expected {expected_features}, got {features.shape[1]}")
                if features.shape[1] < expected_features:
                    padding = np.zeros((features.shape[0], expected_features - features.shape[1]))
                    features = np.hstack([features, padding])
                else:
                    features = features[:, :expected_features]
            
            features_scaled = self.scaler.transform(features)
            
            prediction = self.model.predict(features_scaled)[0]
            
            class_names = self.metadata.get('class_names', ['Flood Risk', 'Urban Heat Risk', 'Fire Risk', 'Deforestation Risk'])
            label_mapping = self.metadata.get('label_mapping', {})
            
            # Improved mapping using model.classes_ if available
            class_to_name = {}
            if hasattr(self.model, 'classes_'):
                for idx, label in enumerate(self.model.classes_):
                    class_to_name[idx] = label_mapping.get(str(label), class_names[label] if label < len(class_names) else f"Class {label}")
                risk_level = class_to_name.get(np.where(self.model.classes_ == prediction)[0][0], f"Unknown Risk Level {prediction}")
            else:
                if str(int(prediction)) in label_mapping:
                    risk_level = label_mapping[str(int(prediction))]
                elif int(prediction) < len(class_names):
                    risk_level = class_names[int(prediction)]
                else:
                    risk_level = f"Unknown Risk Level {int(prediction)}"
            
            result = {
                'risk_level': risk_level,
                'risk_code': int(prediction),
                'timestamp': datetime.now().isoformat(),
                'model_version': self.metadata.get('model_type', 'Unknown'),
                'confidence': None,
                'probabilities': None
            }
            
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba(features_scaled)[0]
                    # Normalize probabilities to ensure they sum to 1
                    prob_sum = np.sum(probabilities)
                    if prob_sum > 0:
                        probabilities = probabilities / prob_sum
                    
                    result['confidence'] = float(np.max(probabilities))
                    
                    prob_dict = {}
                    if hasattr(self.model, 'classes_'):
                        for idx, prob in enumerate(probabilities):
                            label = self.model.classes_[idx]
                            class_name = label_mapping.get(str(label), class_names[int(label)] if int(label) < len(class_names) else f"Class {label}")
                            prob_dict[class_name] = float(prob)
                    else:
                        for i, prob in enumerate(probabilities):
                            class_name = label_mapping.get(str(i), class_names[i] if i < len(class_names) else f"Unknown Class {i}")
                            prob_dict[class_name] = float(prob)
                    
                    result['probabilities'] = prob_dict
                    
                except Exception as e:
                    logger.warning(f"Could not compute probabilities: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None

# Initialize the enhanced application
climate_app = EnhancedClimateDisasterApp()

@app.route('/')
def index():
    """Enhanced home page with comprehensive model information"""
    model_info = {}
    if climate_app.metadata:
        model_info = {
            'model_type': climate_app.metadata.get('model_type', 'Unknown'),
            'n_features': climate_app.metadata.get('n_features', 0),
            'n_classes': climate_app.metadata.get('n_classes', 0),
            'training_date': climate_app.metadata.get('training_date', 'Unknown'),
            'class_names': climate_app.metadata.get('class_names', []),
            'feature_names': climate_app.metadata.get('feature_names', [])
        }
    
    return render_template("index.html", model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint with comprehensive validation"""
    try:
        if climate_app.model is None:
            return render_template("index.html", 
                                 message="Error: Model not loaded. Ensure model files exist and are trained.",
                                 message_type="error",
                                 model_info=climate_app.metadata or {})
        
        image_file = request.files.get('image')
        features = None
        data_type = request.form.get('data_type', 'forest')
        
        if image_file and image_file.filename:
            logger.info(f"Processing uploaded image: {image_file.filename} for {data_type} analysis")
            features = climate_app.process_image(image_file, data_type)
            
            if features is None:
                return render_template("index.html", 
                                     message="Error: Could not process the uploaded image. Ensure it's a valid SAR image file (.jpg, .png, .tif, .filt).",
                                     message_type="error",
                                     model_info=climate_app.metadata or {})
        else:
            input_text = request.form.get('features', '').strip()
            
            if not input_text:
                return render_template("index.html", 
                                     message="Error: Provide SAR feature values or upload an image.",
                                     message_type="error",
                                     model_info=climate_app.metadata or {})
            
            try:
                feature_values = [float(x.strip()) for x in input_text.split(',')]
            except ValueError:
                return render_template("index.html", 
                                     message="Error: Provide valid numeric values separated by commas.",
                                     message_type="error",
                                     model_info=climate_app.metadata or {})
            
            expected_features = climate_app.metadata.get('n_features', 15)
            if len(feature_values) != expected_features:
                return render_template("index.html", 
                                     message=f"Error: Expected {expected_features} features, got {len(feature_values)}. Provide all required SAR features.",
                                     message_type="error",
                                     model_info=climate_app.metadata or {})
            
            features = np.array(feature_values)
        
        prediction_result = climate_app.predict_risk(features)
        
        if prediction_result is None:
            return render_template("index.html", 
                                 message="Error: Could not generate prediction. Check your input and try again.",
                                 message_type="error",
                                 model_info=climate_app.metadata or {})
        
        message = generate_enhanced_prediction_message(prediction_result, data_type)
        
        logger.info(f"Prediction made: {prediction_result['risk_level']} (confidence: {prediction_result.get('confidence', 'N/A')})")
        return render_template("index.html", 
                             message=message, 
                             message_type="success",
                             prediction=prediction_result,
                             model_info=climate_app.metadata or {})
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return render_template("index.html", 
                             message="An unexpected error occurred. Check your input and try again.",
                             message_type="error",
                             model_info=climate_app.metadata or {})

def generate_enhanced_prediction_message(prediction_result, data_type):
    """Generate enhanced prediction message with modern styling and comprehensive information"""
    risk_level = prediction_result['risk_level']
    confidence = prediction_result.get('confidence', 0)
    model_version = prediction_result.get('model_version', 'Unknown')
    
    risk_configs = {
        'Flood Risk': {
            'icon': '🌊',
            'gradient': 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
            'bg_class': 'bg-blue-500',
            'border_class': 'border-blue-500',
            'title': 'Flood Risk Assessment',
            'description': 'SAR analysis suggests potential flooding vulnerability.',
            'status_badge': 'CAUTION',
            'badge_class': 'badge-warning',
            'recommendations': [
                'Monitor precipitation forecasts and water levels',
                'Inspect flood barriers and drainage infrastructure',
                'Review emergency response and evacuation routes',
                'Coordinate with water management authorities',
                'Prepare emergency supplies and resources'
            ]
        },
        'Urban Heat Risk': {
            'icon': '🌡️',
            'gradient': 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
            'bg_class': 'bg-amber-500',
            'border_class': 'border-amber-500',
            'title': 'Urban Heat Island Risk',
            'description': 'Urban thermal analysis indicates elevated heat risk conditions.',
            'status_badge': 'ELEVATED',
            'badge_class': 'badge-warning',
            'recommendations': [
                'Monitor temperature forecasts and heat index',
                'Activate cooling centers and public facilities',
                'Issue heat warnings to vulnerable populations',
                'Implement urban heat mitigation measures',
                'Monitor air quality and heat-related health risks'
            ]
        },
        'Fire Risk': {
            'icon': '🔥',
            'gradient': 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
            'bg_class': 'bg-red-500',
            'border_class': 'border-red-500',
            'title': 'Wildfire Risk Detected',
            'description': 'SAR analysis shows indicators consistent with elevated wildfire conditions.',
            'status_badge': 'HIGH ALERT',
            'badge_class': 'badge-danger',
            'recommendations': [
                'Implement immediate fire weather monitoring',
                'Review and activate evacuation procedures',
                'Coordinate with local fire management authorities',
                'Consider temporary access restrictions to high-risk areas',
                'Monitor wind patterns and humidity levels'
            ]
        },
        'Deforestation Risk': {
            'icon': '🌲',
            'gradient': 'linear-gradient(135deg, #16a34a 0%, #15803d 100%)',
            'bg_class': 'bg-green-600',
            'border_class': 'border-green-600',
            'title': 'Deforestation Risk Detected',
            'description': 'Land cover analysis indicates potential deforestation or land change activity.',
            'status_badge': 'ALERT',
            'badge_class': 'badge-warning',
            'recommendations': [
                'Conduct ground verification surveys',
                'Monitor vegetation index changes over time',
                'Coordinate with environmental protection authorities',
                'Implement forest conservation measures',
                'Review land use policies in the area'
            ]
        }
    }
    
    risk_info = risk_configs.get(risk_level, risk_configs['Flood Risk'])  # Default to Flood if unknown
    
    message = f"""
    <div class="prediction-result-container" style="font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;">
        <!-- Main Risk Assessment Card -->
        <div class="card shadow-2xl border-0 mb-6 overflow-hidden" style="border-radius: 16px; backdrop-filter: blur(10px);">
            <div class="card-header text-white position-relative" style="background: {risk_info['gradient']}; padding: 2.5rem 2rem;">
                <div class="d-flex align-items-center justify-content-between">
                    <div class="d-flex align-items-center">
                        <span class="display-3 me-4" style="filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));">{risk_info['icon']}</span>
                        <div>
                            <h2 class="mb-2 fw-bold" style="font-size: 2rem; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">{risk_info['title']}</h2>
                            <span class="badge {risk_info['badge_class']} px-4 py-2" style="font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">{risk_info['status_badge']}</span>
                        </div>
                    </div>
                    <div class="text-end">
                        <div class="fs-6 opacity-75 mb-1">Model Confidence</div>
                        <div class="display-5 fw-bold" style="text-shadow: 0 2px 4px rgba(0,0,0,0.3);">{confidence:.0%}</div>
                        <div class="small opacity-75 mt-1">{model_version}</div>
                    </div>
                </div>
            </div>
            
            <div class="card-body p-0">
                <!-- Analysis Summary -->
                <div class="p-4 bg-gradient-to-r from-gray-50 to-white dark:from-gray-800 dark:to-gray-700">
                    <p class="lead mb-0 text-gray-700 dark:text-gray-200" style="font-size: 1.1rem; line-height: 1.6;">{risk_info['description']}</p>
                </div>
                
                <!-- Enhanced Confidence Meter -->
                <div class="px-4 py-3 border-b border-gray-100 dark:border-gray-700">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <span class="fw-semibold text-dark dark:text-white" style="font-size: 1rem;">Assessment Reliability</span>
                        <div class="d-flex align-items-center">
                            <span class="badge bg-primary px-3 py-2 me-2" style="font-size: 0.9rem;">{confidence:.1%}</span>
                            <small class="text-muted">Confidence Score</small>
                        </div>
                    </div>
                    <div class="progress mb-2" style="height: 12px; border-radius: 10px; background: #e5e7eb;">
                        <div class="progress-bar progress-bar-striped progress-bar-animated" 
                             style="width: {confidence*100:.0f}%; background: {risk_info['gradient']}; border-radius: 10px; transition: width 1.2s ease-out;"></div>
                    </div>
                    <div class="d-flex justify-content-between text-xs text-muted">
                        <span>Low</span>
                        <span>Moderate</span>
                        <span>High</span>
                        <span>Very High</span>
                    </div>
                </div>
                
                <!-- Action Items Section -->
                <div class="p-4">
                    <h5 class="fw-bold mb-4 d-flex align-items-center text-dark dark:text-white">
                        <i class="bi bi-list-check me-3 text-2xl" style="color: {risk_info['gradient'].split()[2]};"></i>
                        <span>📋 Recommended Actions</span>
                    </h5>
                    <div class="row g-3">
    """
    
    for i, rec in enumerate(risk_info['recommendations']):
        message += f"""
                        <div class="col-md-6 col-lg-4">
                            <div class="d-flex align-items-start p-4 bg-gray-50 dark:bg-gray-800 rounded-xl border-l-4 hover:shadow-lg transition-all duration-300" 
                                 style="border-left-color: {risk_info['gradient'].split()[2]}; min-height: 80px;">
                                <span class="badge bg-secondary rounded-circle me-3 d-flex align-items-center justify-content-center flex-shrink-0" 
                                      style="width: 28px; height: 28px; font-size: 0.8rem; font-weight: 600;">{i+1}</span>
                                <span class="text-dark dark:text-white fw-medium" style="font-size: 0.95rem; line-height: 1.4;">{rec}</span>
                            </div>
                        </div>
        """
    
    message += """
                    </div>
                </div>
            </div>
        </div>
    """
    
    if prediction_result.get('probabilities'):
        message += """
        <div class="card shadow-lg border-0 mb-4" style="border-radius: 16px;">
            <div class="card-header text-white" style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); border-radius: 16px 16px 0 0; padding: 1.5rem;">
                <h5 class="mb-0 fw-bold d-flex align-items-center">
                    <span class="me-3">📊</span> Detailed Risk Assessment Breakdown
                </h5>
            </div>
            <div class="card-body p-4" style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);">
                <div class="row g-4">
        """
        
        sorted_probs = sorted(prediction_result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        
        for risk_type, probability in sorted_probs:
            percentage = probability * 100
            if percentage >= 70:
                bar_color = '#dc2626'
                intensity = 'Very High'
                bg_color = '#fef2f2'
            elif percentage >= 50:
                bar_color = '#ea580c'
                intensity = 'High'
                bg_color = '#fff7ed'
            elif percentage >= 30:
                bar_color = '#d97706'
                intensity = 'Moderate'
                bg_color = '#fffbeb'
            elif percentage >= 15:
                bar_color = '#059669'
                intensity = 'Low'
                bg_color = '#f0fdf4'
            else:
                bar_color = '#10b981'
                intensity = 'Very Low'
                bg_color = '#f0fdf4'
            
            message += f"""
                    <div class="col-md-6 col-lg-3 mb-3">
                        <div class="p-4 rounded-xl shadow-sm border hover:shadow-md transition-all duration-300" 
                             style="background: {bg_color}; border-color: {bar_color}20;">
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <span class="fw-bold text-gray-800" style="font-size: 0.9rem;">{risk_type}</span>
                                <span class="badge text-white fw-semibold" 
                                      style="background-color: {bar_color}; padding: 0.8rem 0.8rem; font-size: 0.8rem;">{percentage:.1f}%</span>
                            </div>
                            <div class="progress mb-2" style="height: 8px; border-radius: 10px; background-color: #e5e7eb;">
                                <div class="progress-bar" 
                                     style="width: {percentage}%; background-color: {bar_color}; border-radius: 10px; 
                                            transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);"></div>
                            </div>
                            <div class="text-center">
                                <small class="text-gray-600 fw-medium">{intensity} Probability</small>
                            </div>
                        </div>
                    </div>
            """
        
        message += """
                </div>
            </div>
        </div>
        """
    
    message += f"""
        <div class="card border-0 shadow-lg mb-4" style="border-radius: 16px; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);">
            <div class="card-body p-4">
                <div class="d-flex align-items-start">
                    <div class="me-4">
                        <span style="font-size: 2.5rem;">⚠️</span>
                    </div>
                    <div class="flex-grow-1">
                        <h5 class="fw-bold text-gray-800 mb-3">Important Usage Guidelines</h5>
                        <div class="alert alert-warning bg-gradient-to-r border-l-4 border-yellow-500 rounded-lg mb-0" 
                             style="background: linear-gradient(90deg, #fef3c7 0%, #fde68a 100%); border-left: 4px solid #f59e0b;">
                            <p class="mb-2 fw-semibold text-gray-800">This system provides research-grade risk assessment for early warning purposes.</p>
                            <p class="mb-2 text-gray-700">Predictions are based on SAR satellite data analysis using {model_version} model and should supplement, not replace, official weather services and emergency management guidance.</p>
                            <p class="mb-0 fw-bold text-gray-800">Always prioritize official evacuation orders, weather warnings, and emergency management directives.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Enhanced Analysis Metadata -->
        <div class="text-center mt-4 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg border">
            <div class="row align-items-center text-sm text-gray-600 dark:text-gray-400">
                <div class="col-md-4">
                    <i class="bi bi-clock me-2"></i>
                    <span>Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</span>
                </div>
                <div class="col-md-4">
                    <i class="bi bi-satellite me-2"></i>
                    <span>SAR Data Processing System v3.0</span>
                </div>
                <div class="col-md-4">
                    <i class="bi bi-cpu me-2"></i>
                    <span>Model: {model_version}</span>
                </div>
            </div>
        </div>
    </div>
    
    <style>
        .prediction-result-container {{
            animation: fadeInUp 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(40px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .card {{
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .card:hover {{
            transform: translateY(-4px);
        }}
        
        .progress-bar {{
            transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .badge {{
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        
        .lead {{
            line-height: 1.7;
        }}
        
        @media (max-width: 768px) {{
            .prediction-result-container {{
                padding: 0.5rem;
            }}
            .card-header {{
                padding: 1.5rem 1rem !important;
            }}
            .display-3 {{
                font-size: 2.5rem !important;
            }}
        }}
    </style>
    """
    
    return message

if __name__ == "__main__":
    if climate_app.model is not None:
        print("✅ Enhanced Climate Disaster Prediction System Ready")
        print("🛰️ SAR-based risk assessment model loaded")
        print("🌍 Multi-format SAR data support enabled")
        print("🔬 Starting enhanced server at http://127.0.0.1:5000")
    else:
        print("⚠️ Model not loaded. Please run the updated model_trainer.py first.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)