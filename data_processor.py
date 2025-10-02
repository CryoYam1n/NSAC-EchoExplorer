# data_processor.py - NASA Space Apps Challenge 2025 (COMPLETE - 9 DISASTERS)
import numpy as np
import cv2
import os
import json
import h5py
import struct
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from PIL import Image as PilImage
import warnings
import logging
from collections import Counter

# Optional imports with fallbacks
try:
    import geopandas as gpd
    from shapely.geometry import box
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: Geopandas not available - Shapefile support limited")

try:
    import rasterio
    from rasterio.features import rasterize
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: Rasterio not available - GeoTIFF support limited")

try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    print("Warning: GDAL not available - BSQ/IMG support limited")

try:
    import netCDF4 as nc
    NETCDF_AVAILABLE = True
except ImportError:
    NETCDF_AVAILABLE = False
    print("Warning: NetCDF4 not available - NC4 support limited")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedSARDataProcessor:
    """
    NASA Space Apps Challenge 2025 - 9 Climate Disaster Types
    COMPLETE VERSION with Volcanic Eruption included
    """
    
    def __init__(self, base_path=None):
        if base_path is None:
            possible_paths = [
                Path(r"E:\Echo Explorer\NASA SAR Data"),
                Path(r"E:\Nasa Space Apps Challenge- 2025\Echo Explorer\NASA SAR Data")
            ]
            self.base_path = None
            for path in possible_paths:
                if path.exists():
                    self.base_path = path
                    break
            if self.base_path is None:
                self.base_path = possible_paths[0]
        else:
            self.base_path = Path(base_path)
        
        # CRITICAL: 9 disaster types with consistent labeling
        self.disaster_types = {
            'Flood': 0,
            'Urban Heat Risk': 1,
            'Forest Fire': 2,
            'Deforestation': 3,
            'Drought': 4,
            'Tsunami': 5,
            'Landslide Monitoring': 6,
            'Cyclone/Hurricane': 7,
            'Volcanic Eruption': 8  # ADDED - was missing
        }
        
        # === FLOOD DATASETS (Label 0) ===
        self.flood_paths = {
            'waterbodies_flood_1': self.base_path / "WaterBodies Dataset(flood)" / "data",
            'waterbodies_2': self.base_path / "Water Bodies Dataset",
            'sentinel_flood_cyclone': self.base_path / "Flood &  Cyclone (SENTINEL-1B_SINGLE_POL_METADATA_GRD_HIGH_RES)",
            'smap_soil_moisture': self.base_path / "SMAPSentinel-1 L2 RadiometerRadar 30-Second Scene 3 km EASE-Grid Soil Moisture V003",
        }
        
        # === URBAN HEAT ISLAND (Label 1) ===
        self.urban_heat_paths = {
            'uhi_yearly': self.base_path / "urban heat island-sdei-yceo-sfc-uhi",
            'uhi_train': self.base_path / "urban heat island data",
            'uhi_global_2013': self.base_path / "sdei-global-uhi-2013",
            'uhi_urban_cluster': self.base_path / "sdei-yceo-sfc-uhi-v4-urban-cluster-means-shp"
        }
        
        # === FOREST FIRE (Label 2) ===
        self.fire_paths = {
            'goes_fire': self.base_path / "forest fire(LBA-ECO LC-35 GOES Imager)" / "data",
            'global_fire_2016': self.base_path / "Global_fire_atlas_V1_ignitions_2016",
            'modis_fire': self.base_path / "LC39_MODIS_Fire_SA_1186" / "data",
            'cms_global_fire': self.base_path / "CMS_Global_Fire_Atlas_1642" / "data"
        }
        
        # === DEFORESTATION (Label 3) ===
        self.deforestation_paths = {
            'modis_deforest': self.base_path / "LC39_MODIS_Fire_SA_1186" / "data",
            'cms_deforest': self.base_path / "CMS_Global_Fire_Atlas_1642" / "data"
        }
        
        # === DROUGHT (Label 4) ===
        self.drought_paths = {
            'flood_draught_comp': self.base_path / "Floods & Draught" / "comp",
            'flood_draught_fews': self.base_path / "Floods & Draught" / "FEWS_precip_711",
            'flood_draught_imerg': self.base_path / "Floods & Draught" / "IMERG_Precip_Canada_Alaska_2097",
            'flood_draught_grace': self.base_path / "Floods & Draught" / "GRACEDADM_CLSM025GL_7D"
        }
        
        # === TSUNAMI (Label 5) ===
        self.tsunami_paths = {
            'jason3_tsunami': self.base_path / "Tsunami-Jason-3 GPS based orbit and SSHA OGDR"
        }
        
        # === LANDSLIDE MONITORING (Label 6) ===
        self.landslide_paths = {
            'hls_landsat': self.base_path / "HLS Landsat Operational Land Imager Surface"
        }
        
        # === CYCLONE/HURRICANE (Label 7) ===
        self.cyclone_paths = {
            'cyclone_tisa_1': self.base_path / "CycloneHurricane-TISAavg_SampleRead_SYN1deg_R5-922",
            'cyclone_tisa_2': self.base_path / "TISAavg_SampleRead_SYN1deg_R5-922 (1)"
        }
        
        # === VOLCANIC ERUPTION (Label 8) - ADDED ===
        self.volcanic_paths = {
            'aster_volcanic': self.base_path / "Volcanic Eruption-ASTER Global Emissivity Dataset"
        }
        
        # === TEMPERATURE & HUMIDITY (for context enhancement) ===
        self.climate_paths = {
            'temp_humidity': self.base_path / "Maryland_Temperature_Humidity_1319" / "data"
        }
        
        self.image_size = (256, 256)
        self.max_pixels = 10_000_000
        self.max_files_per_type = 1000
        self.batch_size = 50
        
        logger.info("=" * 80)
        logger.info("NASA SPACE APPS CHALLENGE 2025 - SAR DATA PROCESSOR")
        logger.info("=" * 80)
        logger.info(f"Base Path: {self.base_path}")
        logger.info("9 Disaster Types:")
        for name, label in sorted(self.disaster_types.items(), key=lambda x: x[1]):
            logger.info(f"  [{label}] {name}")
        logger.info("=" * 80)
    
    def verify_all_datasets(self):
        """Comprehensive dataset verification"""
        logger.info("\n" + "="*80)
        logger.info("DATASET VERIFICATION")
        logger.info("="*80)
        
        total_files = 0
        dataset_summary = {}
        
        all_datasets = {
            'FLOOD (Label 0)': self.flood_paths,
            'URBAN HEAT ISLAND (Label 1)': self.urban_heat_paths,
            'FOREST FIRE (Label 2)': self.fire_paths,
            'DEFORESTATION (Label 3)': self.deforestation_paths,
            'DROUGHT (Label 4)': self.drought_paths,
            'TSUNAMI (Label 5)': self.tsunami_paths,
            'LANDSLIDE MONITORING (Label 6)': self.landslide_paths,
            'CYCLONE/HURRICANE (Label 7)': self.cyclone_paths,
            'VOLCANIC ERUPTION (Label 8)': self.volcanic_paths,
            'CLIMATE/WEATHER (context)': self.climate_paths
        }
        
        for category, paths in all_datasets.items():
            logger.info(f"\n{category}:")
            for name, path in paths.items():
                count = self._count_files(path, name)
                total_files += count
                dataset_summary[name] = count
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TOTAL FILES FOUND: {total_files}")
        logger.info(f"{'='*80}")
        
        if total_files == 0:
            logger.error("\nWARNING: No data files found!")
            logger.error(f"Please check that your data is at: {self.base_path}")
        
        return total_files > 0, dataset_summary
    
    def _count_files(self, path, dataset_name):
        """Count files in a dataset path"""
        if not path.exists():
            logger.warning(f"  {dataset_name}: NOT FOUND at {path}")
            return 0
        
        extensions = ['.jpg', '.png', '.tif', '.tiff', '.h5', '.hdf5', '.filt', 
                     '.csv', '.bsq', '.shp', '.nc4', '.nc', '.img', '.if', '.dbf',
                     '.prj', '.shx', '.sbx', '.CPG', '.xml', '.ovr', '.aux']
        
        files = []
        try:
            for ext in extensions:
                files.extend(list(path.glob(f"*{ext}")))
                if len(files) < 100:
                    files.extend(list(path.glob(f"*/*{ext}")))
        except PermissionError:
            logger.warning(f"  {dataset_name}: Permission denied")
            return 0
        except Exception as e:
            logger.warning(f"  {dataset_name}: Error - {str(e)[:50]}")
            return 0
        
        # Remove duplicates and auxiliary files
        main_files = [f for f in files if f.suffix.lower() not in ['.xml', '.ovr', '.aux']]
        files = list(set(main_files))[:self.max_files_per_type]
        
        logger.info(f"  {dataset_name}: {len(files)} files")
        return len(files)
    
    # ==================== FILE READERS ====================
    
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
                if data is not None:
                    return self._normalize_array(data)
        except Exception as e:
            logger.debug(f"H5 read error {file_path.name}: {str(e)[:100]}")
        return None
    
    def read_nc4_file(self, file_path):
        """Read NetCDF4 files"""
        if not NETCDF_AVAILABLE:
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
        except:
            return self.read_h5_file(file_path)
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
            logger.debug(f"FILT read error {file_path.name}: {str(e)[:100]}")
        return None
    
    def read_bsq_file(self, file_path):
        """Read BSQ/IMG files"""
        if GDAL_AVAILABLE:
            try:
                dataset = gdal.Open(str(file_path))
                if dataset:
                    band = dataset.GetRasterBand(1)
                    return self._normalize_array(band.ReadAsArray())
            except:
                pass
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
                if len(data) % 4 == 0:
                    values = struct.unpack(f'>{len(data)//4}f', data)
                    size = int(np.sqrt(len(values)))
                    if size * size == len(values):
                        return self._normalize_array(np.array(values).reshape(size, size))
        except:
            pass
        return None
    
    def read_shapefile(self, file_path):
        """Read shapefiles and rasterize"""
        if not GEOPANDAS_AVAILABLE or not RASTERIO_AVAILABLE:
            return None
        
        try:
            gdf = gpd.read_file(file_path)
            if gdf.empty:
                return None
            
            bounds = gdf.total_bounds
            width, height = self.image_size
            transform = rasterio.transform.from_bounds(
                bounds[0], bounds[1], bounds[2], bounds[3], width, height
            )
            
            shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
            if not shapes:
                return None
                
            img_array = rasterize(
                shapes, out_shape=(height, width), 
                transform=transform, fill=0, dtype=np.uint8
            )
            return img_array
        except Exception as e:
            logger.debug(f"Shapefile error {file_path.name}: {str(e)[:100]}")
        return None
    
    def read_tiff(self, file_path):
        """Read TIFF/GeoTIFF files"""
        if RASTERIO_AVAILABLE:
            try:
                with rasterio.open(file_path) as src:
                    data = src.read(1)
                    return self._normalize_array(data)
            except:
                pass
        
        if GDAL_AVAILABLE:
            try:
                dataset = gdal.Open(str(file_path))
                if dataset:
                    data = dataset.GetRasterBand(1).ReadAsArray()
                    return self._normalize_array(data)
            except:
                pass
        
        try:
            with PilImage.open(file_path) as img:
                if img.size[0] * img.size[1] > self.max_pixels:
                    ratio = np.sqrt(self.max_pixels / (img.size[0] * img.size[1]))
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, PilImage.LANCZOS)
                if img.mode != 'L':
                    img = img.convert('L')
                return np.array(img)
        except:
            pass
        
        try:
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
        except:
            pass
        
        return None
    
    def load_image(self, file_path):
        """Load standard images"""
        try:
            img = PilImage.open(str(file_path))
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            if img.mode != 'L':
                img = img.convert('L')
            return np.array(img)
        except:
            try:
                return cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            except:
                return None
    
    def read_csv_climate(self, file_path):
        """Read CSV climate data"""
        try:
            df = pd.read_csv(file_path)
            
            temp_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['temp', 'temperature'])]
            humid_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['humid', 'humidity', 'rh'])]
            
            if temp_cols and len(df) > 0:
                temp_data = df[temp_cols[0]].values
                if len(temp_data) >= 256:
                    size = int(np.sqrt(len(temp_data)))
                    img_data = temp_data[:size*size].reshape(size, size)
                    return self._normalize_array(img_data)
                else:
                    tiled = np.tile(temp_data, (256 // len(temp_data) + 1))
                    img_data = tiled[:256*256].reshape(256, 256)
                    return self._normalize_array(img_data)
        except Exception as e:
            logger.debug(f"CSV error {file_path.name}: {str(e)[:100]}")
        return None
    
    def _normalize_array(self, data):
        """Normalize array to 0-255 uint8"""
        if data is None or data.size == 0:
            return None
        
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        if data.ndim > 2:
            if data.shape[0] <= 3:
                data = data[0]
            else:
                data = data.mean(axis=0)
        
        if data.max() == data.min():
            return np.zeros(self.image_size, dtype=np.uint8)
        
        normalized = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
        
        if normalized.shape != self.image_size:
            try:
                normalized = cv2.resize(normalized, self.image_size, interpolation=cv2.INTER_AREA)
            except:
                return np.zeros(self.image_size, dtype=np.uint8)
        
        return normalized
    
    # ==================== DATA LOADING ====================
    
    def load_dataset_by_type(self, paths_dict, disaster_label, disaster_type_name, data_context):
        """Generic dataset loader"""
        logger.info(f"\nLoading {disaster_type_name} Data (Label {disaster_label})...")
        X, y, types = [], [], []
        
        for name, path in paths_dict.items():
            if not path.exists():
                continue
            
            files = []
            for ext in ['.tif', '.tiff', '.h5', '.hdf5', '.nc4', '.nc', 
                       '.bsq', '.img', '.filt', '.shp', '.jpg', '.png', '.csv']:
                try:
                    files.extend(list(path.glob(f"*{ext}")))
                    if len(files) < 50:
                        files.extend(list(path.glob(f"*/*{ext}")))
                except:
                    pass
            
            files = list(set(files))[:self.max_files_per_type]
            
            if not files:
                continue
            
            for file in tqdm(files, desc=f"  {name}"):
                img = None
                ext = file.suffix.lower()
                
                try:
                    if ext in ['.h5', '.hdf5']:
                        img = self.read_h5_file(file)
                    elif ext in ['.nc4', '.nc']:
                        img = self.read_nc4_file(file)
                    elif ext in ['.bsq', '.img']:
                        img = self.read_bsq_file(file)
                    elif ext == '.filt':
                        img = self.read_filt_file(file)
                    elif ext in ['.tif', '.tiff']:
                        img = self.read_tiff(file)
                    elif ext == '.shp':
                        img = self.read_shapefile(file)
                    elif ext == '.csv':
                        img = self.read_csv_climate(file)
                    elif ext in ['.jpg', '.png']:
                        img = self.load_image(file)
                    
                    if img is not None and img.size > 0:
                        if img.shape != self.image_size:
                            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
                        
                        features = self.extract_sar_features(img, data_context, disaster_type_name)
                        X.append(features)
                        y.append(disaster_label)
                        types.append(disaster_type_name)
                except Exception as e:
                    logger.debug(f"Error processing {file.name}: {str(e)[:100]}")
                    continue
        
        logger.info(f"  Loaded {len(X)} samples for {disaster_type_name}")
        return X, y, types
    
    def extract_sar_features(self, image, data_type, disaster_name):
        """Extract exactly 15 SAR features - CONSISTENT across all files"""
        features = []
        
        if image is None or image.size == 0:
            return [0.0] * 15
        
        # 1-5: Statistical features
        features.extend([
            float(np.mean(image)),
            float(np.std(image)),
            float(np.min(image)),
            float(np.max(image)),
            float(np.median(image))
        ])
        
        # 6-10: Texture features (GLCM approximation)
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
        
        # 11-12: Gradient features
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            float(np.mean(gradient_magnitude)),
            float(np.std(gradient_magnitude))
        ])
        
        # 13-14: Frequency features
        fft = np.fft.fft2(image)
        fft_magnitude = np.abs(fft)
        features.extend([
            float(np.mean(fft_magnitude)),
            float(np.std(fft_magnitude))
        ])
        
        # 15: Domain-specific feature - IMPROVED for 9 disaster types
        normalized = image.astype(np.float32) / 255.0
        
        if data_type == 'fire' or disaster_name == 'Forest Fire':
            dark_areas = np.mean(normalized < 0.3)
            medium_areas = np.mean((normalized >= 0.3) & (normalized < 0.7))
            domain_feature = 0.6 * dark_areas + 0.3 * medium_areas + 0.1 * (1 - np.mean(normalized))
        
        elif data_type == 'flood' or disaster_name == 'Flood':
            very_dark = np.sum(normalized < 0.15)
            moderately_dark = np.sum((normalized >= 0.15) & (normalized < 0.25))
            domain_feature = min((0.8 * very_dark + 0.3 * moderately_dark) / image.size, 1.0)
        
        elif data_type == 'urban' or disaster_name == 'Urban Heat Risk':
            very_bright = np.sum(normalized > 0.7)
            moderately_bright = np.sum((normalized >= 0.5) & (normalized <= 0.7))
            domain_feature = min((0.9 * very_bright + 0.4 * moderately_bright) / image.size, 1.0)
        
        elif disaster_name == 'Drought':
            # Drought: low moisture = higher brightness in specific bands
            bright_areas = np.mean(normalized > 0.6)
            domain_feature = float(bright_areas * 0.8 + np.std(normalized) * 0.2)
        
        elif disaster_name == 'Tsunami':
            # Tsunami: Focus on water bodies (dark areas)
            very_dark = np.sum(normalized < 0.2)
            domain_feature = min(0.9 * very_dark / image.size, 1.0)
        
        elif disaster_name == 'Landslide Monitoring':
            # Landslide: High gradient areas (terrain instability)
            domain_feature = float(np.mean(gradient_magnitude > np.percentile(gradient_magnitude, 75)))
        
        elif disaster_name == 'Cyclone/Hurricane':
            # Cyclone: High variance patterns (storm systems)
            domain_feature = float(np.var(normalized) * 2.0)
        
        elif disaster_name == 'Volcanic Eruption':
            # Volcanic: Thermal anomalies (bright spots) + texture variation
            very_bright = np.sum(normalized > 0.8)
            texture_var = float(np.std(local_patterns))
            domain_feature = min((0.7 * very_bright / image.size) + (0.3 * texture_var / 255.0), 1.0)
        
        elif disaster_name == 'Deforestation':
            # Deforestation: Change in vegetation (moderate brightness changes)
            moderate_bright = np.mean((normalized >= 0.4) & (normalized <= 0.6))
            domain_feature = float(moderate_bright)
        
        else:
            domain_feature = float(np.mean(normalized))
        
        features.append(float(domain_feature))
        
        if len(features) != 15:
            logger.error(f"Feature count mismatch: {len(features)} instead of 15")
            return [0.0] * 15
        
        return features
    
    def process_all_data(self):
        """Process all 9 disaster types"""
        logger.info("\n" + "="*80)
        logger.info("PROCESSING ALL 9 DISASTER TYPES")
        logger.info("="*80)
        
        X_all, y_all, types_all = [], [], []
        
        # CORRECTED: Load all 9 disaster types with proper labels
        datasets_config = [
            (self.flood_paths, 0, 'Flood', 'flood'),
            (self.urban_heat_paths, 1, 'Urban Heat Risk', 'urban'),
            (self.fire_paths, 2, 'Forest Fire', 'fire'),
            (self.deforestation_paths, 3, 'Deforestation', 'fire'),
            (self.drought_paths, 4, 'Drought', 'flood'),
            (self.tsunami_paths, 5, 'Tsunami', 'weather'),
            (self.landslide_paths, 6, 'Landslide Monitoring', 'weather'),
            (self.cyclone_paths, 7, 'Cyclone/Hurricane', 'weather'),
            (self.volcanic_paths, 8, 'Volcanic Eruption', 'volcanic')  # ADDED
        ]
        
        for paths, label, disaster_name, context in datasets_config:
            X, y, types = self.load_dataset_by_type(paths, label, disaster_name, context)
            X_all.extend(X)
            y_all.extend(y)
            types_all.extend(types)
        
        if len(X_all) == 0:
            logger.error("No data loaded! Please check dataset paths.")
            return False
        
        # Convert to numpy arrays
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        
        # Feature names
        feature_names = [
            'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 'median_intensity',
            'glcm_homogeneity', 'glcm_energy', 'glcm_entropy', 'glcm_contrast', 'glcm_correlation',
            'gradient_mean', 'gradient_std', 'fft_mean', 'fft_std', 'domain_specific_feature'
        ]
        
        # Save processed data
        np.savez('comprehensive_sar_data.npz',
                features=X_all,
                labels=y_all,
                disaster_types=types_all,
                feature_names=feature_names)
        
        logger.info("\n" + "="*80)
        logger.info("DATA PROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"Total samples: {len(X_all)}")
        logger.info(f"Features per sample: {X_all.shape[1]}")
        logger.info(f"\nClass distribution:")
        class_dist = Counter(y_all)
        class_names = list(self.disaster_types.keys())
        for label in sorted(class_dist.keys()):
            logger.info(f"  [{label}] {class_names[label]}: {class_dist[label]} samples")
        logger.info(f"\nSaved to: comprehensive_sar_data.npz")
        logger.info("="*80)
        
        return True

def main():
    """Main processing pipeline"""
    logger.info("="*80)
    logger.info("NASA SPACE APPS CHALLENGE 2025 - DATA PROCESSING")
    logger.info("9 DISASTER TYPES: Flood, Urban Heat, Fire, Deforestation,")
    logger.info("                  Drought, Tsunami, Landslide, Cyclone, Volcanic")
    logger.info("="*80)
    
    processor = AdvancedSARDataProcessor()
    
    # Verify datasets
    logger.info("\nStep 1: Verifying datasets...")
    valid, summary = processor.verify_all_datasets()
    
    if not valid:
        logger.error("\nNo data found. Please check data directories.")
        logger.error(f"Expected base path: {processor.base_path}")
        logger.error("\nEnsure your data is in one of these locations:")
        logger.error("  - E:\\Echo Explorer\\NASA SAR Data\\")
        logger.error("  - E:\\Nasa Space Apps Challenge- 2025\\Echo Explorer\\NASA SAR Data\\")
        return
    
    # Process all data
    logger.info("\nStep 2: Processing datasets...")
    success = processor.process_all_data()
    
    if success:
        logger.info("\n" + "="*80)
        logger.info("SUCCESS!")
        logger.info("="*80)
        logger.info("Next step: Run 'python model_trainer.py' to train the model")
        logger.info("="*80)
    else:
        logger.error("\nData processing failed")

if __name__ == "__main__":
    main()
