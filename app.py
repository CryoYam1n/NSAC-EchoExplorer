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
import h5py
import struct
import warnings

try:
    import geopandas as gpd
    from shapely.geometry import box
    import rasterio
    from rasterio.features import rasterize
    GEOPANDAS_AVAILABLE = RASTERIO_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = RASTERIO_AVAILABLE = False
    print("Warning: Geopandas or Rasterio not available")

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    print("Warning: GDAL not available")

try:
    import netCDF4 as nc
    NETCDF_AVAILABLE = True
except ImportError:
    NETCDF_AVAILABLE = False
    print("Warning: NetCDF4 not available")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'png', 'tif', 'tiff', 'filt', 'h5', 'hdf5', 'bsq', 'shp', 'nc', 'nc4', 'img'}

class AdvancedClimateDisasterApp:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = []
        self.image_size = (256, 256)
        self.max_pixels = 10_000_000
        self.load_models()

    def load_models(self):
        """Load trained models with enhanced error handling"""
        try:
            with open('climate_disaster_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully")

            with open('climate_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Scaler loaded successfully")

            with open('climate_model_metadata.json', 'r') as f:
                self.metadata = json.load(f)
                self.feature_names = self.metadata.get('feature_names', [])
            logger.info(f"Metadata loaded: {self.metadata.get('model_type', 'Unknown')}")
            logger.info(f"Model supports: {self.metadata.get('n_features', 'Unknown')} features")
            logger.info(f"Classes: {self.metadata.get('n_classes', 'Unknown')} disaster types")

        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            logger.error("Please run: python model_trainer.py")
            self.metadata = {
                'n_features': 15,
                'n_classes': 9,  
                'class_names': [
                    'Flood', 'Urban Heat Risk', 'Forest Fire', 'Deforestation',
                    'Drought', 'Tsunami', 'Landslide Monitoring', 'Cyclone/Hurricane',
                    'Volcanic Eruption'  
                ],
                'model_type': 'Not Trained - Please Run model_trainer.py',
                'training_date': 'N/A',
                'version': 'Not Available',
                'label_mapping': {
                    '0': 'Flood',
                    '1': 'Urban Heat Risk',
                    '2': 'Forest Fire',
                    '3': 'Deforestation',
                    '4': 'Drought',
                    '5': 'Tsunami',
                    '6': 'Landslide Monitoring',
                    '7': 'Cyclone/Hurricane',
                    '8': 'Volcanic Eruption'  # ADDED
                }
            }
            self.feature_names = [
                'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 'median_intensity',
                'glcm_homogeneity', 'glcm_energy', 'glcm_entropy', 'glcm_contrast', 'glcm_correlation',
                'gradient_mean', 'gradient_std', 'fft_mean', 'fft_std', 'domain_specific_feature'
            ]
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

    def allowed_file(self, filename):
        """Check if file extension is allowed"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def read_h5_file(self, file_path):
        """Read HDF5/H5 files"""
        try:
            with h5py.File(file_path, 'r') as f:
                def extract_data(obj, depth=0):
                    if depth > 5:
                        return None
                    if isinstance(obj, h5py.Dataset):
                        try:
                            data = obj[()]
                            if isinstance(data, np.ndarray) and data.size > 0:
                                return data
                        except:
                            pass
                    elif isinstance(obj, h5py.Group):
                        for key in obj.keys():
                            result = extract_data(obj[key], depth + 1)
                            if result is not None:
                                return result
                    return None
                data = extract_data(f)
                return self._normalize_array(data) if data is not None else None
        except Exception as e:
            logger.error(f"Error reading H5 file {file_path}: {e}")
        return None

    def read_nc4_file(self, file_path):
        """Read NetCDF4 files"""
        if not NETCDF_AVAILABLE:
            logger.warning("NetCDF4 not available, falling back to H5 reader")
            return self.read_h5_file(file_path)
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                priority_vars = ['lwe_thickness', 'soil_moisture', 'ssha',
                               'sea_surface_height', 'precipitation', 'data', 'value']
                for var_name in priority_vars:
                    if var_name in dataset.variables:
                        data = dataset.variables[var_name][:]
                        return self._normalize_array(data)
                if len(dataset.variables) > 0:
                    first_var = list(dataset.variables.keys())[0]
                    data = dataset.variables[first_var][:]
                    return self._normalize_array(data)
        except Exception as e:
            logger.error(f"Error reading NC4 file {file_path}: {e}")
            return self.read_h5_file(file_path)
        return None

    def read_bsq_file(self, file_path):
        """Read BSQ/IMG files"""
        if GDAL_AVAILABLE:
            try:
                dataset = gdal.Open(str(file_path))
                if dataset:
                    band = dataset.GetRasterBand(1)
                    return self._normalize_array(band.ReadAsArray())
            except Exception as e:
                logger.warning(f"GDAL failed for BSQ {file_path}: {e}")
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
                if len(data) % 4 == 0:
                    values = struct.unpack(f'>{len(data)//4}f', data)
                    size = int(np.sqrt(len(values)))
                    if size * size == len(values):
                        return self._normalize_array(np.array(values).reshape(size, size))
        except Exception as e:
            logger.error(f"Error reading BSQ file {file_path}: {e}")
        return None

    def read_filt_file(self, file_path):
        """Read NASA GOES .filt format files"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
                for dtype, fmt in [(np.float32, 'f'), (np.float64, 'd'), (np.uint16, 'H')]:
                    element_size = np.dtype(dtype).itemsize
                    if len(data) % element_size == 0:
                        try:
                            values = struct.unpack(f'>{len(data)//element_size}{fmt}', data)
                            size = int(np.sqrt(len(values)))
                            if size * size == len(values):
                                return self._normalize_array(np.array(values).reshape(size, size))
                        except:
                            continue
                img_array = np.frombuffer(data, dtype=np.uint8)
                size = int(np.sqrt(len(img_array)))
                if size * size <= len(img_array):
                    return img_array[:size*size].reshape(size, size)
        except Exception as e:
            logger.error(f"Error reading FILT file {file_path}: {e}")
        return None

    def read_shapefile(self, file_path):
        """Read shapefiles and rasterize to image"""
        if not GEOPANDAS_AVAILABLE or not RASTERIO_AVAILABLE:
            logger.warning("Geopandas or Rasterio not available for shapefile processing")
            return None
        try:
            gdf = gpd.read_file(file_path)
            if gdf.empty:
                logger.warning(f"Empty shapefile: {file_path}")
                return None
            bounds = gdf.total_bounds
            width, height = self.image_size
            transform = rasterio.transform.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
            shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
            if not shapes:
                return None
            img_array = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)
            return img_array
        except Exception as e:
            logger.error(f"Error reading shapefile {file_path}: {e}")
        return None

    def read_geotiff(self, file_path):
        """Read GeoTIFF files using rasterio, GDAL, PIL, or cv2"""
        try:
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            if not os.access(file_path, os.R_OK):
                logger.error(f"Permission denied for {file_path}")
                return None
            if RASTERIO_AVAILABLE:
                try:
                    with rasterio.open(file_path) as src:
                        data = src.read(1)
                        return self._normalize_array(data)
                except Exception as e:
                    logger.warning(f"Rasterio failed for {file_path}: {e}")
            if GDAL_AVAILABLE:
                try:
                    dataset = gdal.Open(str(file_path))
                    if dataset:
                        band = dataset.GetRasterBand(1)
                        data = band.ReadAsArray()
                        return self._normalize_array(data)
                except Exception as e:
                    logger.warning(f"GDAL failed for {file_path}: {e}")
            try:
                with PilImage.open(file_path) as img:
                    if img.size[0] * img.size[1] > self.max_pixels:
                        ratio = np.sqrt(self.max_pixels / (img.size[0] * img.size[1]))
                        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                        img = img.resize(new_size, PilImage.LANCZOS)
                    if img.mode != 'L':
                        img = img.convert('L')
                    return np.array(img)
            except Exception as e:
                logger.warning(f"PIL failed for {file_path}: {e}")
            try:
                img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return img
            except Exception as e:
                logger.error(f"OpenCV failed for {file_path}: {e}")
            logger.error(f"Failed to read TIFF file {file_path} (size: {file_size:.2f} MB)")
            return None
        except Exception as e:
            logger.error(f"Error reading TIFF file {file_path}: {e}")
        return None

    def load_image(self, file_path):
        """Load standard images using PIL/OpenCV"""
        try:
            pil_img = PilImage.open(str(file_path))
            if pil_img.mode == 'RGBA':
                pil_img = pil_img.convert('RGB')
            if pil_img.mode != 'L':
                pil_img = pil_img.convert('L')
            img = np.array(pil_img)
            return img
        except Exception:
            try:
                img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return img
            except Exception as e:
                logger.error(f"Error loading image {file_path}: {e}")
        return None

    def _normalize_array(self, data):
        """Normalize array to 0-255 uint8"""
        if data is None or data.size == 0:
            return None
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if data.ndim > 2:
            data = data[0] if data.shape[0] == 1 else data.mean(axis=0)
        if data.max() == data.min():
            return np.zeros(self.image_size, dtype=np.uint8)
        normalized = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
        if normalized.shape != self.image_size:
            try:
                normalized = cv2.resize(normalized, self.image_size, interpolation=cv2.INTER_AREA)
            except:
                return np.zeros(self.image_size, dtype=np.uint8)
        return normalized

    def process_uploaded_file(self, file, data_type='auto'):
        """Process uploaded file"""
        try:
            filename = secure_filename(file.filename)
            if not self.allowed_file(filename):
                raise ValueError(f"Invalid file type: {filename}")
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)
            try:
                file_size = os.path.getsize(temp_path) / (1024 * 1024)
                if file_size > 500:
                    raise ValueError(f"File too large: {file_size:.2f} MB exceeds 500 MB limit")
                ext = filename.rsplit('.', 1)[1].lower()
                img = None
                if ext in ['h5', 'hdf5']:
                    img = self.read_h5_file(temp_path)
                elif ext in ['nc', 'nc4']:
                    img = self.read_nc4_file(temp_path)
                elif ext in ['bsq', 'img']:
                    img = self.read_bsq_file(temp_path)
                elif ext == 'filt':
                    img = self.read_filt_file(temp_path)
                elif ext in ['tif', 'tiff']:
                    img = self.read_geotiff(temp_path)
                elif ext == 'shp':
                    img = self.read_shapefile(temp_path)
                elif ext in ['jpg', 'png']:
                    img = self.load_image(temp_path)
                if img is None:
                    raise ValueError("Could not process the uploaded file")
                if img.shape != self.image_size:
                    img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
                context_map = {
                    'flood': 'flood',
                    'drought': 'flood',
                    'urban_heat_risk': 'urban',
                    'forest_fire': 'fire',
                    'deforestation': 'fire',
                    'tsunami': 'weather',
                    'landslide_monitoring': 'weather',
                    'cyclone_hurricane': 'weather',
                    'volcanic_eruption': 'volcanic',  # ADDED
                    'auto': 'weather'
                }
                context = context_map.get(data_type.lower().replace(' ', '_'), 'weather')
                features = self.extract_sar_features(img, context, data_type)
                return np.array(features)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            logger.error(f"File processing error: {str(e)}")
            return None

    def extract_sar_features(self, image, context, data_type):
        """Extract exactly 15 SAR features - CORRECTED for 9 disaster types"""
        features = []
        if image is None or image.size == 0:
            return [0.0] * 15
        features.extend([
            float(np.mean(image)),
            float(np.std(image)),
            float(np.min(image)),
            float(np.max(image)),
            float(np.median(image))
        ])
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
        local_patterns = cv2.filter2D(image.astype(np.float32), -1, kernel)
        contrast = float(np.var(image))
        homogeneity = float(1.0 / (1.0 + contrast))
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-10)
        energy = float(np.sum(hist ** 2))
        entropy = float(-np.sum(hist * np.log(hist + 1e-10)))
        correlation = np.corrcoef(image.flatten(), local_patterns.flatten())[0, 1]
        correlation = 0.0 if np.isnan(correlation) else float(abs(correlation))
        features.extend([homogeneity, energy, entropy, contrast, correlation])
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features.extend([
            float(np.mean(gradient_magnitude)),
            float(np.std(gradient_magnitude))
        ])
        fft = np.fft.fft2(image)
        fft_magnitude = np.abs(fft)
        features.extend([
            float(np.mean(fft_magnitude)),
            float(np.std(fft_magnitude))
        ])
        normalized = image.astype(np.float32) / 255.0
        data_type_lower = data_type.lower().replace(' ', '_')
        
        if context == 'fire' or data_type_lower in ['forest_fire', 'deforestation']:
            dark_areas = np.mean(normalized < 0.3)
            medium_areas = np.mean((normalized >= 0.3) & (normalized < 0.7))
            domain_feature = 0.6 * dark_areas + 0.3 * medium_areas + 0.1 * (1 - np.mean(normalized))
        elif context == 'flood' or data_type_lower in ['flood', 'drought']:
            very_dark = np.sum(normalized < 0.15)
            moderately_dark = np.sum((normalized >= 0.15) & (normalized < 0.25))
            domain_feature = min((0.8 * very_dark + 0.3 * moderately_dark) / image.size, 1.0)
        elif context == 'urban' or data_type_lower == 'urban_heat_risk':
            very_bright = np.sum(normalized > 0.7)
            moderately_bright = np.sum((normalized >= 0.5) & (normalized <= 0.7))
            domain_feature = min((0.9 * very_bright + 0.4 * moderately_bright) / image.size, 1.0)
        elif data_type_lower == 'tsunami':
            very_dark = np.sum(normalized < 0.2)
            domain_feature = min(0.9 * very_dark / image.size, 1.0)
        elif data_type_lower == 'landslide_monitoring':
            domain_feature = float(np.mean(gradient_magnitude > np.percentile(gradient_magnitude, 75)))
        elif data_type_lower == 'cyclone_hurricane':
            domain_feature = float(np.var(normalized))
        elif context == 'volcanic' or data_type_lower == 'volcanic_eruption':  # ADDED
            very_bright = np.sum(normalized > 0.8)
            texture_var = float(np.std(local_patterns))
            domain_feature = min((0.7 * very_bright / image.size) + (0.3 * texture_var / 255.0), 1.0)
        else:
            domain_feature = float(np.mean(normalized))
        
        features.append(float(domain_feature))
        if len(features) != 15:
            logger.error(f"Feature count mismatch: {len(features)} instead of 15")
            return [0.0] * 15
        return features

    def predict_risk(self, features):
        """Make prediction with confidence scores"""
        if self.model is None or self.scaler is None:
            return None
        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)
            expected_features = self.metadata.get('n_features', 15)
            if features.shape[1] != expected_features:
                logger.warning(f"Feature dimension mismatch: {features.shape[1]} vs {expected_features}")
                if features.shape[1] < expected_features:
                    padding = np.zeros((features.shape[0], expected_features - features.shape[1]))
                    features = np.hstack([features, padding])
                else:
                    features = features[:, :expected_features]
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            class_names = self.metadata.get('class_names', [])
            label_mapping = self.metadata.get('label_mapping', {})
            risk_level = label_mapping.get(str(int(prediction)), 
                                         class_names[int(prediction)] if int(prediction) < len(class_names) else f"Risk Level {int(prediction)}")
            result = {
                'risk_level': risk_level,
                'risk_code': int(prediction),
                'timestamp': datetime.now().isoformat(),
                'model_version': self.metadata.get('version', 'Unknown'),
                'model_training_date': self.metadata.get('training_date', 'Unknown'),
                'confidence': None,
                'probabilities': None,
                'feature_count': features.shape[1]
            }
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba(features_scaled)[0]
                    result['confidence'] = float(np.max(probabilities))
                    prob_dict = {label_mapping.get(str(i), class_names[i] if i < len(class_names) else f"Class {i}"): float(prob) 
                                 for i, prob in enumerate(probabilities)}
                    result['probabilities'] = prob_dict
                    result['top_risks'] = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                except Exception as e:
                    logger.warning(f"Could not compute probabilities: {e}")
            return result
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None

@app.route('/')
def index():
    """Home page"""
    model_info = {
        'model_type': climate_app.metadata.get('model_type', 'Unknown'),
        'n_features': climate_app.metadata.get('n_features', 15),
        'n_classes': climate_app.metadata.get('n_classes', 9),  # CHANGED from 8 to 9
        'training_date': climate_app.metadata.get('training_date', 'Unknown'),
        'class_names': climate_app.metadata.get('class_names', []),
        'version': climate_app.metadata.get('version', '3.0-NASA-Space-Apps-2025'),
        'feature_names': climate_app.feature_names,
        # CORRECTED: 9 disaster types
        'disaster_types': ['Flood', 'Urban Heat Risk', 'Forest Fire', 'Deforestation', 
                          'Drought', 'Tsunami', 'Landslide Monitoring', 'Cyclone/Hurricane', 
                          'Volcanic Eruption']
    }
    return render_template("index.html", model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        if climate_app.model is None:
            return render_template("index.html",
                                 message="""Model not loaded.<br>Please train the model first by running:<br><code>python model_trainer.py</code><br><br>Or use the manual feature input below.""",
                                 message_type="warning",
                                 model_info=climate_app.metadata or {})
        image_file = request.files.get('image')
        features = None
        data_type = request.form.get('data_type', 'auto')
        if image_file and image_file.filename:
            logger.info(f"Processing uploaded file: {image_file.filename}")
            features = climate_app.process_uploaded_file(image_file, data_type)
            if features is None:
                return render_template("index.html",
                                     message="Could not process the uploaded file. Please check:<br>- File format (TIFF, HDF5, BSQ, FILT, SHP, NC4, JPG, PNG)<br>- File integrity<br>- File permissions<br>- File size (< 500 MB)",
                                     message_type="error",
                                     model_info=climate_app.metadata or {})
        else:
            input_text = request.form.get('features', '').strip()
            if not input_text:
                return render_template("index.html",
                                     message="Please provide either a file upload or manual feature values.",
                                     message_type="error",
                                     model_info=climate_app.metadata or {})
            try:
                feature_values = [float(x.strip()) for x in input_text.split(',')]
            except ValueError:
                return render_template("index.html",
                                     message="Invalid feature format. Please provide 15 comma-separated numbers.",
                                     message_type="error",
                                     model_info=climate_app.metadata or {})
            expected_features = climate_app.metadata.get('n_features', 15)
            if len(feature_values) != expected_features:
                return render_template("index.html",
                                     message=f"Expected {expected_features} features, but received {len(feature_values)}.<br>Please provide exactly {expected_features} comma-separated values.",
                                     message_type="error",
                                     model_info=climate_app.metadata or {})
            features = np.array(feature_values)
        prediction_result = climate_app.predict_risk(features)
        if prediction_result is None:
            return render_template("index.html",
                                 message="Prediction failed. Please check your input and try again.",
                                 message_type="error",
                                 model_info=climate_app.metadata or {})
        message = generate_enhanced_prediction_message(prediction_result, data_type)
        logger.info(f"Prediction: {prediction_result['risk_level']} (Confidence: {prediction_result.get('confidence', 0):.2f})")
        return render_template("index.html",
                             message=message,
                             message_type="success",
                             prediction=prediction_result,
                             model_info=climate_app.metadata or {})
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return render_template("index.html",
                             message=f"An unexpected error occurred: {str(e)}<br>Please check the server logs for details.",
                             message_type="error",
                             model_info=climate_app.metadata or {})

def generate_enhanced_prediction_message(prediction_result, data_type):
    """Generate comprehensive prediction display - CORRECTED for 9 disasters"""
    risk_level = prediction_result['risk_level']
    confidence = prediction_result.get('confidence', 0)
    
    risk_configs = {
        'Flood': {
            'icon': '🌊',
            'color': "#0f6afd",
            'severity': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low',
            'description': 'Potential flooding detected based on water body analysis and soil moisture',
            'recommendations': [
                'Monitor water levels and precipitation forecasts',
                'Check drainage systems and flood barriers',
                'Review evacuation routes and emergency plans',
                'Coordinate with local flood management authorities'
            ],
            'immediate_actions': [
                'Alert emergency services if confidence > 80%',
                'Monitor weather updates regularly',
                'Prepare emergency supplies'
            ]
        },
        'Urban Heat Risk': {
            'icon': '🌡️',
            'color': '#f59e0b',
            'severity': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low',
            'description': 'Elevated urban temperatures indicating heat island effect',
            'recommendations': [
                'Monitor temperature forecasts and heat indices',
                'Activate cooling centers for vulnerable populations',
                'Issue public heat warnings and advisories',
                'Check on elderly and at-risk individuals'
            ],
            'immediate_actions': [
                'Stay hydrated and avoid prolonged sun exposure',
                'Use air conditioning or fans',
                'Check on vulnerable community members'
            ]
        },
        'Forest Fire': {
            'icon': '🔥',
            'color': '#ef4444',
            'severity': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low',
            'description': 'Elevated fire risk detected based on vegetation analysis and thermal patterns',
            'recommendations': [
                'Monitor fire weather conditions and wind patterns',
                'Review and update evacuation procedures',
                'Coordinate with fire services and emergency responders',
                'Restrict access to high-risk forested areas'
            ],
            'immediate_actions': [
                'Report any signs of fire immediately',
                'Prepare evacuation plan',
                'Monitor local fire department alerts'
            ]
        },
        'Deforestation': {
            'icon': '🌲',
            'color': '#16a34a',
            'severity': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low',
            'description': 'Potential deforestation activity detected through vegetation cover analysis',
            'recommendations': [
                'Conduct ground verification surveys',
                'Monitor vegetation changes over time',
                'Coordinate with environmental protection agencies',
                'Review and enforce land use policies'
            ],
            'immediate_actions': [
                'Report suspicious land clearing activities',
                'Document vegetation changes with photos',
                'Contact local environmental authorities'
            ]
        },
        'Drought': {
            'icon': '🏜️',
            'color': '#d97706',
            'severity': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low',
            'description': 'Drought conditions indicated by soil moisture analysis and vegetation stress',
            'recommendations': [
                'Monitor soil moisture levels and precipitation deficits',
                'Implement water conservation measures',
                'Assess agricultural impact and crop status',
                'Coordinate regional water resource management'
            ],
            'immediate_actions': [
                'Follow local water restriction guidelines',
                'Report water supply issues',
                'Monitor crop and vegetation health'
            ]
        },
        'Tsunami': {
            'icon': '🌊',
            'color': "#016bd4",
            'severity': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low',
            'description': 'Tsunami risk detected based on coastal water patterns and seismic activity indicators',
            'recommendations': [
                'Monitor tsunami warning systems and seismic activity',
                'Review coastal evacuation routes and plans',
                'Coordinate with national tsunami warning centers',
                'Conduct public awareness campaigns'
            ],
            'immediate_actions': [
                'Follow tsunami warning alerts',
                'Move to higher ground if advised',
                'Stay informed via official channels'
            ]
        },
        'Landslide Monitoring': {
            'icon': '🏔️',
            'color': '#8b4513',
            'severity': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low',
            'description': 'Landslide risk detected based on terrain instability and precipitation patterns',
            'recommendations': [
                'Monitor rainfall and soil stability data',
                'Inspect slopes and retaining structures',
                'Update landslide evacuation plans',
                'Coordinate with geological survey teams'
            ],
            'immediate_actions': [
                'Avoid unstable slopes during heavy rain',
                'Report unusual ground movement',
                'Follow local authority guidance'
            ]
        },
        'Cyclone/Hurricane': {
            'icon': '🌪️',
            'color': '#8b5cf6',
            'severity': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low',
            'description': 'Cyclone or hurricane risk detected based on atmospheric and oceanic patterns',
            'recommendations': [
                'Monitor storm tracks and weather forecasts',
                'Reinforce infrastructure against high winds',
                'Prepare emergency shelters and supplies',
                'Coordinate with meteorological agencies'
            ],
            'immediate_actions': [
                'Follow storm warnings and advisories',
                'Secure property and evacuate if ordered',
                'Stay updated via weather services'
            ]
        },
        # ADDED: Volcanic Eruption configuration
        'Volcanic Eruption': {
            'icon': '🌋',
            'color': '#dc2626',
            'severity': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low',
            'description': 'Volcanic activity detected through thermal anomalies and surface deformation',
            'recommendations': [
                'Monitor volcanic activity and seismic data',
                'Establish exclusion zones around active vents',
                'Prepare for ashfall and lava flow scenarios',
                'Coordinate with volcanological observatories'
            ],
            'immediate_actions': [
                'Follow evacuation orders immediately',
                'Prepare emergency go-bags and supplies',
                'Monitor volcanic alert levels'
            ]
        }
    }
    config = risk_configs.get(risk_level, risk_configs['Flood'])
    message = f"""
    <div class="prediction-container" style="font-family: system-ui, -apple-system, sans-serif;">
        <div class="card border-0 shadow-lg mb-4" style="border-radius: 12px; overflow: hidden;">
            <div class="card-header text-white" style="background: linear-gradient(135deg, {config['color']} 0%, {config['color']}dd 100%); padding: 2rem;">
                <div class="d-flex align-items-center justify-content-between">
                    <div>
                        <h2 class="mb-2 fw-bold">{config['icon']} {risk_level}</h2>
                        <p class="mb-1 opacity-90">{config['description']}</p>
                        <div class="d-flex gap-3 mt-2">
                            <div class="badge bg-light text-dark px-3 py-2">
                                <i class="fas fa-shield-alt me-1"></i>Severity: {config['severity']}
                            </div>
                            <div class="badge bg-light text-dark px-3 py-2">
                                <i class="fas fa-chart-line me-1"></i>Confidence: {confidence:.0%}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="card-body p-4">
                <div class="row">
                    <div class="col-md-6">
                        <h5 class="fw-bold mb-3">Risk Assessment</h5>
                        <div class="progress mb-2" style="height: 20px;">
                            <div class="progress-bar" style="width: {confidence*100}%; background-color: {config['color']};">
                                {confidence:.1%}
                            </div>
                        </div>
                        <small class="text-muted">Model Confidence Score</small>
                    </div>
                    <div class="col-md-6">
                        <h5 class="fw-bold mb-3">Immediate Actions</h5>
                        <ul class="list-unstyled">
    """
    for action in config['immediate_actions']:
        message += f'<li class="mb-1">✅ {action}</li>'
    message += """
                        </ul>
                    </div>
                </div>
                <div class="mt-4">
                    <h5 class="fw-bold mb-3">Recommended Response Plan</h5>
                    <div class="row g-3">
    """
    for i, rec in enumerate(config['recommendations'], 1):
        message += f"""
                        <div class="col-md-6">
                            <div class="p-3 border rounded h-100">
                                <div class="d-flex align-items-start">
                                    <span class="badge bg-primary me-2">{i}</span>
                                    <span>{rec}</span>
                                </div>
                            </div>
                        </div>
        """
    message += """
                    </div>
                </div>
    """
    if prediction_result.get('probabilities'):
        message += """
                <div class="mt-4">
                    <h5 class="fw-bold mb-3">Detailed Risk Analysis</h5>
                    <div class="table-responsive">
                        <table class="table table-sm table-bordered">
                            <thead class="table-light">
                                <tr>
                                    <th>Risk Type</th>
                                    <th>Probability</th>
                                    <th>Confidence</th>
                                </tr>
                            </thead>
                            <tbody>
        """
        for risk_type, prob in sorted(prediction_result['probabilities'].items(), 
                                     key=lambda x: x[1], reverse=True):
            percentage = prob * 100
            bar_color = config['color'] if risk_type == risk_level else '#6c757d'
            message += f"""
                                <tr>
                                    <td>{risk_type}</td>
                                    <td>
                                        <div class="d-flex align-items-center">
                                            <div class="progress flex-grow-1 me-2" style="height: 8px;">
                                                <div class="progress-bar" style="width: {percentage}%; background-color: {bar_color};"></div>
                                            </div>
                                            <span>{percentage:.1f}%</span>
                                        </div>
                                    </td>
                                    <td>{'Primary' if risk_type == risk_level else 'Secondary'}</td>
                                </tr>
            """
        message += """
                            </tbody>
                        </table>
                    </div>
                </div>
        """
    message += f"""
                <div class="alert alert-warning mt-4">
                    <strong>⚠️ NASA Research Disclaimer:</strong> This assessment uses machine learning on multi-format SAR data 
                    (TIFF, HDF5, BSQ, FILT, SHP, NC4, JPG, PNG). Always follow official emergency management guidance and 
                    verify with ground truth data. Model version: {prediction_result.get('model_version', 'Unknown')}
                </div>
            </div>
        </div>
    </div>
    """
    return message

climate_app = AdvancedClimateDisasterApp()

if __name__ == '__main__':
    app.run(debug=True)
