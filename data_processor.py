import numpy as np
import cv2
import os
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import warnings
import struct
from PIL import Image as PilImage
warnings.filterwarnings('ignore')

class SARDataProcessor:
    """
    Professional SAR Data Processing Pipeline
    For Climate Disaster Risk Prediction with Real NASA Dataset Integration
    """
    
    def __init__(self, base_path="E:/Nasa Space Apps Challenge- 2025/Echo Explorer/NASA SAR Data/"):
        self.base_path = Path(base_path)
        
        
        self.forest_fire_path = self.base_path / "forest fire(LBA-ECO LC-35 GOES Imager)/data"
        self.flood_path1 = self.base_path / "WaterBodies Dataset(flood)/data"
        self.flood_path2 = self.base_path / "Water Bodies Dataset/Images"
        self.urban_heat_path = self.base_path / "urban heat island data"
        self.urban_data_path = self.base_path / "urban_data"
        
        
        self.image_size = (256, 256)
        self.features = []
        self.labels = []
        
        print("Enhanced SAR Data Processor Initialized")
        print(f"Base Path: {self.base_path}")
        
    def verify_data_paths(self):
        """Verify all data paths exist and count files"""
        paths = {
            'Forest Fire Data': self.forest_fire_path,
            'Flood Dataset 1': self.flood_path1,
            'Flood Dataset 2 (Images)': self.flood_path2,
            'Urban Heat Island': self.urban_heat_path,
            'Urban Classification Data': self.urban_data_path
        }
        
        total_files = 0
        for name, path in paths.items():
            if path.exists():
                print(f"✅ {name} found at: {path}")
                
                if name == 'Forest Fire Data':
                    files = list(path.glob("*.filt")) + list(path.glob("*.samer.*"))
                elif name == 'Urban Heat Island':
                    files = list(path.glob("*.tif")) + list(path.glob("*.tiff"))
                elif 'Flood' in name:
                    files = list(path.glob("*.jpg")) + list(path.glob("*.png")) + list(path.glob("*.tif"))
                else:
                    files = list(path.rglob("*.png")) + list(path.rglob("*.jpg")) + list(path.rglob("*.tif"))
                
                print(f"  Files found: {len(files)}")
                total_files += len(files)
            else:
                print(f"❌ {name} NOT found at: {path}")
        
        print(f"\nTotal dataset files available: {total_files}")
        return total_files > 0
    
    def read_filt_file(self, file_path):
        """Read NASA GOES .filt format files"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
                
                if len(data) % 4 == 0:
                    values = struct.unpack(f'>{len(data)//4}f', data)
                    size = int(np.sqrt(len(values)))
                    if size * size == len(values):
                        img_array = np.array(values).reshape(size, size)
                        img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                        return img_array
                
                img_array = np.frombuffer(data, dtype=np.uint8)
                size = int(np.sqrt(len(img_array)))
                if size * size <= len(img_array):
                    img_array = img_array[:size*size].reshape(size, size)
                    return img_array
                    
        except Exception as e:
            print(f"Error reading .filt file {file_path}: {e}")
            return None
    
    def load_image(self, file_path):
        """Load image using PIL first, fallback to cv2 for robustness"""
        try:
            pil_img = PilImage.open(str(file_path))
            img = np.array(pil_img.convert('L'))
            return img
        except Exception as e:
            print(f"PIL failed for {file_path}: {e}. Falling back to cv2.")
            try:
                img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                return img
            except Exception as e:
                print(f"cv2 failed for {file_path}: {e}")
                return None
    
    def load_forest_fire_data(self, limit=None):
        """Load forest fire data from NASA GOES dataset"""
        print("\nLoading NASA GOES Forest Fire Data...")
        
        forest_features = []
        forest_labels = []
        
        if self.forest_fire_path.exists():
            filt_files = list(self.forest_fire_path.glob("*.filt"))
            samer_files = list(self.forest_fire_path.glob("*.samer.*"))
            all_files = filt_files + samer_files
            
            if limit:
                all_files = all_files[:limit]
            
            print(f"Found {len(all_files)} forest fire files")
            
            for file_path in tqdm(all_files, desc="Processing forest fire data"):
                try:
                    if file_path.suffix == '.filt':
                        img = self.read_filt_file(file_path)
                    else:
                        img = self.load_image(file_path)
                    
                    if img is not None and img.size > 0:
                        img = cv2.resize(img, self.image_size)
                        features = self.extract_sar_features(img, 'forest')
                        forest_features.append(features)
                        forest_labels.append(2)  
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
        return np.array(forest_features), np.array(forest_labels)
    
    def load_flood_data(self, limit=None):
        """Load flood/water body data from both datasets"""
        print("\nLoading Flood/Water Body Data...")
        
        flood_features = []
        flood_labels = []
        
        flood_paths = [self.flood_path1, self.flood_path2]
        
        for flood_path in flood_paths:
            if flood_path.exists():
                image_files = (list(flood_path.glob("*.jpg")) + 
                               list(flood_path.glob("*.png")) + 
                               list(flood_path.glob("*.tif")) + 
                               list(flood_path.glob("*.tiff")))
                
                if limit and len(image_files) > limit // 2:
                    image_files = image_files[:limit // 2]
                
                print(f"Processing {len(image_files)} files from {flood_path}")
                
                for img_file in tqdm(image_files, desc=f"Processing {flood_path.name}"):
                    try:
                        img = self.load_image(img_file)
                        if img is None:
                            continue
                            
                        img = cv2.resize(img, self.image_size)
                        features = self.extract_sar_features(img, 'wetland')
                        flood_features.append(features)
                        flood_labels.append(0)  
                        
                    except Exception as e:
                        print(f"Error processing {img_file}: {e}")
                        
        return np.array(flood_features), np.array(flood_labels)
    
    def load_urban_heat_data(self, limit=None):
        """Load urban heat island and land cover data"""
        print("\nLoading Urban Heat Island and Land Cover Data...")
        
        urban_features = []
        urban_labels = []
        
        
        if self.urban_heat_path.exists():
            tif_files = list(self.urban_heat_path.glob("*.tif")) + list(self.urban_heat_path.glob("*.tiff"))
            
            if limit:
                tif_files = tif_files[:limit // 2]
            
            print(f"Processing {len(tif_files)} urban heat files")
            
            for tif_file in tqdm(tif_files, desc="Processing urban heat data"):
                try:
                    img = self.load_image(tif_file)
                    if img is None:
                        continue
                    
                    img = cv2.resize(img, self.image_size)
                    features = self.extract_sar_features(img, 'urban')
                    urban_features.append(features)
                    urban_labels.append(1)  
                    
                except Exception as e:
                    print(f"Error processing {tif_file}: {e}")
        
        
        if self.urban_data_path.exists():
            urban_categories = ['urban', 'agri', 'barrenland', 'grassland']
            
            for category in urban_categories:
                category_path = self.urban_data_path / category
                if category_path.exists():
                    for subdir in ['s1', 's2']:
                        subdir_path = category_path / subdir
                        if subdir_path.exists():
                            png_files = list(subdir_path.glob("*.png"))
                            
                            max_files = (limit // (len(urban_categories) * 2)) if limit else len(png_files)
                            png_files = png_files[:max_files]
                            
                            for png_file in tqdm(png_files, desc=f"Processing {category}/{subdir}"):
                                try:
                                    img = self.load_image(png_file)
                                    if img is None:
                                        continue
                                    
                                    img = cv2.resize(img, self.image_size)
                                    features = self.extract_sar_features(img, 'urban')
                                    urban_features.append(features)
                                    
                                   
                                    if category == 'urban':
                                        urban_labels.append(1)  
                                    else:
                                        urban_labels.append(3)  
                                
                                except Exception as e:
                                    print(f"Error processing {png_file}: {e}")
        
        return np.array(urban_features), np.array(urban_labels)
    
    def extract_sar_features(self, image, data_type):
        """Enhanced SAR feature extraction"""
        features = []
        
        features.extend([
            np.mean(image),
            np.std(image),
            np.min(image),
            np.max(image),
            np.median(image)
        ])
        
        glcm_features = self.calculate_glcm_features(image)
        features.extend(glcm_features)
        
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude)
        ])
        
        fft = np.fft.fft2(image)
        fft_magnitude = np.abs(fft)
        features.extend([
            np.mean(fft_magnitude),
            np.std(fft_magnitude)
        ])
        
        if data_type == 'forest':
            forest_index = self.calculate_forest_index(image)
            features.append(forest_index)
        elif data_type == 'wetland':
            water_coverage = self.estimate_water_coverage(image)
            features.append(water_coverage)
        elif data_type == 'urban':
            urban_density = self.estimate_urban_density(image)
            features.append(urban_density)
            
        return features
    
    def calculate_glcm_features(self, image):
        """Enhanced texture features calculation"""
        mean_intensity = np.mean(image)
        
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        local_patterns = cv2.filter2D(image.astype(np.float32), -1, kernel)
        
        contrast = np.var(image)
        homogeneity = 1 / (1 + contrast)
        
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        energy = np.sum(hist ** 2)
        
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        correlation = np.corrcoef(image.flatten(), local_patterns.flatten())[0,1]
        if np.isnan(correlation):
            correlation = 0
            
        return [homogeneity, energy, entropy, contrast, abs(correlation)]
    
    def calculate_forest_index(self, image):
        """More realistic forest/vegetation index"""
        normalized_image = image / 255.0
        dark_areas = np.mean(normalized_image < 0.3)
        medium_areas = np.mean((normalized_image >= 0.3) & (normalized_image < 0.7))
        vegetation_index = 0.6 * dark_areas + 0.3 * medium_areas + 0.1 * (1 - np.mean(normalized_image))
        return min(max(vegetation_index, 0), 1)
    
    def estimate_water_coverage(self, image):
        """Enhanced water coverage estimation for features (not labels)"""
        normalized_image = image / 255.0
        water_threshold1 = 0.15
        water_threshold2 = 0.25
        very_dark = np.sum(normalized_image < water_threshold1)
        moderately_dark = np.sum((normalized_image >= water_threshold1) & (normalized_image < water_threshold2))
        total_pixels = image.size
        water_coverage = (0.8 * very_dark + 0.3 * moderately_dark) / total_pixels
        return min(water_coverage, 1.0)
    
    def estimate_urban_density(self, image):
        """Enhanced urban density estimation for features (not labels)"""
        normalized_image = image / 255.0
        bright_threshold1 = 0.7
        bright_threshold2 = 0.5
        very_bright = np.sum(normalized_image > bright_threshold1)
        moderately_bright = np.sum((normalized_image >= bright_threshold2) & (normalized_image <= bright_threshold1))
        total_pixels = image.size
        urban_density = (0.9 * very_bright + 0.4 * moderately_bright) / total_pixels
        return min(urban_density, 1.0)
    
    def create_integrated_dataset(self, limit_per_type=200):
        """Create integrated dataset using dataset metadata for labels"""
        print("\n" + "="*60)
        print("CREATING REALISTIC INTEGRATED CLIMATE DISASTER DATASET")
        print("="*60)
        
        forest_X, forest_y = self.load_forest_fire_data(limit=limit_per_type)
        flood_X, flood_y = self.load_flood_data(limit=limit_per_type)
        urban_X, urban_y = self.load_urban_heat_data(limit=limit_per_type)
        
        all_features = []
        all_labels = []
        disaster_types = []
        
        if len(forest_X) > 0:
            all_features.extend(forest_X)
            all_labels.extend(forest_y)
            disaster_types.extend(['fire'] * len(forest_X))
            
        if len(flood_X) > 0:
            all_features.extend(flood_X)
            all_labels.extend(flood_y)
            disaster_types.extend(['flood'] * len(flood_X))
            
        if len(urban_X) > 0:
            all_features.extend(urban_X)
            all_labels.extend(urban_y)
            disaster_types.extend(['heat' if label == 1 else 'deforestation' for label in urban_y])
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"\nRealistic Integrated Dataset Created:")
        print(f"Total samples: {len(X)}")
        print(f"Feature dimensions: {X.shape[1] if len(X) > 0 else 0}")
        print(f"Label distribution:")
        if len(y) > 0:
            unique, counts = np.unique(y, return_counts=True)
            class_names = ['Flood Risk', 'Urban Heat Risk', 'Fire Risk', 'Deforestation Risk']
            for label, count in zip(unique, counts):
                risk_type = class_names[int(label)]
                print(f"  {risk_type}: {count} ({count/len(y)*100:.1f}%)")
        
        return X, y, disaster_types
    
    def save_processed_data(self, X, y, disaster_types, filename="processed_sar_data.npz"):
        """Save processed data for model training"""
        np.savez_compressed(
            filename,
            features=X,
            labels=y,
            disaster_types=disaster_types,
            feature_names=self.get_feature_names()
        )
        print(f"\nProcessed data saved to: {filename}")
        
        metadata = {
            'total_samples': len(X),
            'feature_count': X.shape[1] if len(X) > 0 else 0,
            'disaster_types': list(set(disaster_types)),
            'class_distribution': {
                int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))
            } if len(y) > 0 else {},
            'processing_date': str(np.datetime64('now')),
            'data_sources': [
                'NASA GOES Forest Fire Data',
                'Water Bodies Flood Dataset', 
                'Urban Heat Island Data',
                'Urban Classification Dataset'
            ]
        }
        
        with open(filename.replace('.npz', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def get_feature_names(self):
        """Get feature names for the extracted features"""
        base_features = [
            'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 'median_intensity',
            'glcm_homogeneity', 'glcm_energy', 'glcm_entropy', 'glcm_contrast', 'glcm_correlation',
            'gradient_mean', 'gradient_std',
            'fft_mean', 'fft_std',
            'domain_specific_feature'
        ]
        return base_features

def main():
    """Enhanced main processing pipeline"""
    print("="*60)
    print("ENHANCED SAR DATA PROCESSING FOR CLIMATE DISASTER PREDICTION")
    print("="*60)
    
    processor = SARDataProcessor()
    
    if not processor.verify_data_paths():
        print("\n⚠️ No data found. Please check your data paths and try again.")
        return
    
    X, y, disaster_types = processor.create_integrated_dataset(limit_per_type=100)
    
    if len(X) > 0:
        metadata = processor.save_processed_data(X, y, disaster_types)
        print(f"\n" + "="*60)
        print("ENHANCED DATA PROCESSING COMPLETED")
        print("="*60)
        print(f"Dataset ready for model training!")
        print(f"Samples: {metadata['total_samples']}")
        print(f"Features: {metadata['feature_count']}")
        print(f"Classes: {list(metadata['class_distribution'].keys())}")
        print(f"Data Sources: {len(metadata['data_sources'])}")
    else:
        print("\n⚠️ No data could be processed. Please check data paths and file formats.")

if __name__ == "__main__":
    main()
