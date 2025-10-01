# NASA Space Apps Challenge 2025: Echo Explorer - Climate Disaster Risk Platform

> A comprehensive climate disaster prediction system leveraging NASA SAR (Synthetic Aperture Radar) data and machine learning to provide early warning capabilities for floods, wildfires, urban heat events, and deforestation, combined with an early warning risk platform and the CosmoRadar Earth Observing Mission, powered by NASA.
<div align="center">
  <!-- Python Badge -->
  <a href="https://www.python.org/downloads/" target="_blank">
    <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </a>

  <!-- Flask Badge -->
  <a href="https://flask.palletsprojects.com/" target="_blank">
    <img src="https://img.shields.io/badge/Flask-2.0+-green?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
  </a>



  <!-- NASA Space Apps Challenge Badge -->
  <a href="https://spaceappschallenge.org/" target="_blank">
    <img src="https://img.shields.io/badge/NASA%20Space%20Apps-2025-orange?style=for-the-badge&logo=nasa&logoColor=white" alt="NASA Space Apps Challenge">
  </a>
</div>

<div align="center">
  <!-- Echo Explorer Badge -->
  <a href="https://echoexplorer.netlify.app/" target="_blank">
    <img src="https://img.shields.io/badge/Echo%20Explorer-🌎-blueviolet?style=for-the-badge" alt="Echo Explorer">
  </a>
</div>

## 🌍 Project Overview

Echo Explorer is an advanced web application developed for the NASA Space Apps Challenge 2025, designed to transform complex satellite data into actionable climate disaster predictions. The system processes multiple SAR data formats and employs state-of-the-art machine learning algorithms to assess environmental risks with high accuracy.

### Key Capabilities

- **Multi-modal SAR Analysis**: Processes NASA GOES .filt files and standard satellite imagery
- **Real-time Risk Assessment**: Provides confidence-scored predictions for four disaster types
- **Interactive Visualization**: Modern web interface with professional data presentation
- **Comprehensive Feature Extraction**: 15+ specialized SAR features for enhanced prediction accuracy


---

## 📊NASA Dataset Categories

### 1. 🌊 Flood & Drought Analysis

#### Precipitation Data
- **Dataset:** ABoVE: Bias-Corrected IMERG Monthly Precipitation for Alaska and Canada (2000-2020)
- **Download Link:** [FEWS_precip_711.zip](https://data.ornldaac.earthdata.nasa.gov/protected/bundle/FEWS_precip_711.zip)
- **Local Path:** `Echo Explorer/NASA SAR Data/Floods & Draught/FEWS_precip_711`
- **Format:** `.img` files
- **Description:** Bias-corrected IMERG precipitation data (daily & monthly)
- **Use Case:** Flood and drought risk prediction

#### Water Bodies Monitoring
- **Dataset:** IMERG Precipitation Canada Alaska Water Bodies
- **Download Link:** [IMERG_Precip_Canada_Alaska_2097.zip](https://data.ornldaac.earthdata.nasa.gov/protected/bundle/IMERG_Precip_Canada_Alaska_2097.zip)
- **Local Paths:** 
  - `Echo Explorer/NASA SAR Data/WaterBodies Dataset(flood)/data`
  - `Echo Explorer/NASA SAR Data/Water Bodies Dataset`
- **Format:** `.if` files, `.jpg` images
- **Description:** Water body detection and monitoring dataset
- **Use Case:** Flood risk assessment and water resource management

#### Soil Moisture Data
- **Dataset:** SMAP/Sentinel-1 L2 Radiometer/Radar 30-Second Scene 3 km EASE-Grid Soil Moisture V003
- **Local Path:** `Echo Explorer/NASA SAR Data/SMAPSentinel-1 L2 RadiometerRadar 30-Second Scene 3 km EASE-Grid Soil Moisture V003`
- **Format:** `.h5` files (e.g., `SMAP_L2_SM_SP_1AIWDV_20250924T142712_20250923T135925_118W39N_R19240_001.h5`)
- **Description:** Combined SMAP and Sentinel-1 soil moisture measurements
- **Use Case:** Drought monitoring and flood risk assessment

#### GRACE Land Data Assimilation
- **Dataset:** GRACEDADM CLSM025GL 7D
- **Local Path:** `Echo Explorer/NASA SAR Data/Floods & Draught/GRACEDADM_CLSM025GL_7D`
- **Format:** `.nc4` files (e.g., `GRACEDADM_CLSM025GL_7D.A20231225.030.nc4`)
- **Additional:** `subset_GRACEDADM_CLSM025GL_7D_3.0_20251001_061142_.txt` (contains ~1100 downloadable file links)
- **Format:** `.bsq` files (e.g., `africa_gba2000-01.bsq`)
- **Description:** Land surface data assimilation system providing essential land-related information
- **Use Case:** Provides critical land information for flood and drought risk modeling when combined with climate and weather data

#### Flood & Cyclone SAR Imagery
- **Dataset:** SENTINEL-1B Single Polarization GRD High Resolution
- **Local Path:** `Echo Explorer/NASA SAR Data/Flood & Cyclone (SENTINEL-1B_SINGLE_POL_METADATA_GRD_HIGH_RES)`
- **Format:** `.tiff` files (5 files)
- **Description:** Sentinel-1B SAR imagery for flood and cyclone detection
- **Use Case:** Real-time flood extent mapping and cyclone impact assessment

#### Comparative Analysis Data
- **Local Path:** `Echo Explorer/NASA SAR Data/Floods & Draught/comp`
- **Format:** 4 CSV files
- **Description:** Comparative analysis datasets for flood and drought studies

---

### 2. 🔥 Forest Fire & Deforestation

#### South America Fire Data
- **Dataset:** LBA-ECO LC-35 GOES Imager Active Fire Detection Data, South America (2000-2005)
- **Source:** https://data.nasa.gov/
- **Local Path:** `Echo Explorer/NASA SAR Data/forest fire(LBA-ECO LC-35 GOES Imager)/data`
- **Format:** `.filt` files (e.g., `f20000011245.samer.v60.g8.filt`)
- **Description:** GOES satellite active fire detection data
- **Use Case:** Historical fire occurrence and frequency analysis

#### MODIS Fire Data - South America
- **Dataset:** LBA-ECO LC-39 MODIS Active Fire and Frequency Data (2000-2007)
- **Download Link:** [LC39_MODIS_Fire_SA_1186.zip](https://data.ornldaac.earthdata.nasa.gov/protected/bundle/LC39_MODIS_Fire_SA_1186.zip)
- **Local Path:** `Echo Explorer/NASA SAR Data/LC39_MODIS_Fire_SA_1186/data`
- **Format:** `.dbf`, `.prj`, `.sbn`, `.sbx`, `.shp`, `.shx`, `.tif`, `.xml` files
- **Examples:** 
  - `sa_2000_2001_terra_subset.shp`
  - `sa0001_terra_neighborhood_variety.tif`
- **Description:** MODIS-based active fire detection and annual fire frequency estimates
- **Use Case:** Forest fire occurrence and frequency prediction

#### Global Fire Atlas
- **Dataset:** Global Fire Atlas with Characteristics of Individual Fires (2003-2016)
- **Download Link:** [CMS_Global_Fire_Atlas_1642.zip](https://data.ornldaac.earthdata.nasa.gov/protected/bundle/CMS_Global_Fire_Atlas_1642.zip)
- **Local Paths:**
  - `Echo Explorer/NASA SAR Data/Global_fire_atlas_V1_ignitions_2016`
  - `Echo Explorer/NASA SAR Data/CMS_Global_Fire_Atlas_1642/data`
- **Format:** `.dbf`, `.prj`, `.shp`, `.shx`, `.tif` files
- **Description:** Global dataset tracking ignition timing, fire size, duration, and expansion
- **Use Case:** Global forest fire spread and dynamics analysis

#### Burned Forest Site Data
- **Dataset:** ABoVE: Synthesis of Burned and Unburned Forest Site Data, Alaska and Canada (1983–2016)
- **Download Link:** [ABoVE_Plot_Data_Burned_Sites_1744.zip](https://data.ornldaac.earthdata.nasa.gov/protected/bundle/ABoVE_Plot_Data_Burned_Sites_1744.zip)
- **Description:** Long-term forest fire impact analysis
- **Use Case:** Forest fire risk analysis and vegetation recovery studies

---

### 3. 🌡️ Urban Heat Island (UHI)

#### Global UHI Dataset
- **Dataset:** Global Urban Heat Island (UHI) Data Set, 2013
- **Access Link:** [Global Urban Heat Island (UHI) Data Set](https://search.earthdata.nasa.gov/search/granules?p=C3550192492-ESDIS)
- **Local Paths:**
  - `Echo Explorer/NASA SAR Data/urban heat island`
  - `Echo Explorer/NASA SAR Data/urban heat island data`
- **Format:** `.tif`, `.aux.xml`, `.ovr` files
- **Examples:**
  - `Summer_UHI_yearly_pixel_2003.tif`
  - `Summer_UHI_yearly_pixel_2006.tif`
  - `TrainArea_001.tif`
- **Description:** Multi-year summer urban heat island intensity data
- **Use Case:** Urban heat stress analysis and mitigation planning

#### UHI Shapefile Datasets
- **Local Paths:**
  - `Echo Explorer/NASA SAR Data/sdei-global-uhi-2013`
  - `Echo Explorer/NASA SAR Data/sdei-yceo-sfc-uhi-v4-urban-cluster-means-shp`
- **Format:** `.CPG`, `.dbf`, `.prj`, `.shp`, `.shx`, `.sbx`, `.xml` files
- **Description:** Global and urban cluster UHI spatial datasets
- **Use Case:** Urban planning and heat vulnerability mapping

---

### 4. 🌀 Cyclone / Hurricane

#### CERES Atmospheric Data
- **Dataset:** CERES and GEO-Enhanced TOA, Within-Atmosphere and Surface Fluxes, Clouds and Aerosols Daily Terra-Aqua Edition4A
- **Download Link:** [TISAavg_SampleRead_SYN1deg_R5-922.zip](https://asdc.larc.nasa.gov/documents/ceres/read_software/TISAavg_SampleRead_SYN1deg_R5-922.zip)
- **Local Paths:**
  - `Echo Explorer/NASA SAR Data/CycloneHurricane-TISAavg_SampleRead_SYN1deg_R5-922`
  - `Echo Explorer/NASA SAR Data/TISAavg_SampleRead_SYN1deg_R5-922 (1)` (3-hour temporal resolution)
- **Format:** Binary files with metadata
- **Examples:**
  - `CER_SYN1deg-3Hour_Sample_R5V1`
  - `CER_SYN1deg-3Hour_Sample_R5V1.dump`
  - `CER_SYN1deg-3Hour_Sample_R5V1.met`
  - `SYN1deg_HDFread.h`
- **Description:** Top-of-atmosphere, cloud cover, aerosols, and surface flux data
- **Use Case:** Cyclone/hurricane formation and atmospheric condition analysis

---

### 5. 🌊 Tsunami

#### Jason-3 Sea Surface Height
- **Dataset:** Jason-3 GPS Orbit and Sea Surface Height Anomalies (OGDR)
- **Access Link:** [Jason-3 Dataset](https://search.earthdata.nasa.gov/search/granules?p=C2205122298-POCLOUD)
- **Local Path:** `Echo Explorer/NASA SAR Data/Tsunami-Jason-3 GPS based orbit and SSHA OGDR`
- **Format:** `.nc` files (e.g., `JA3_GPSOPR_2PgS609_209_20250921_175932_20250921_195515.nc`)
- **Description:** GPS-based satellite altimetry for sea surface height anomaly detection
- **Use Case:** Sea-level rise, storm surge, and tsunami impact analysis

---

### 6. 🏔️ Landslide Monitoring

#### Landsat Surface Reflectance
- **Dataset:** HLS Landsat Operational Land Imager Surface Reflectance (30m Global Daily v2.0)
- **Access Link:** [Landsat OLI Dataset](https://search.earthdata.nasa.gov/search/granules?p=C2021957657-LPCLOUD)
- **Local Path:** `Echo Explorer/NASA SAR Data/HLS Landsat Operational Land Imager Surface`
- **Format:** `.tif` files
- **Description:** 30m resolution surface reflectance data
- **Use Case:** Land cover change, deforestation, floodplain mapping, and landslide monitoring

---

### 7. 🌋 Volcanic Eruption Risk

#### ASTER Global Emissivity
- **Dataset:** ASTER Global Emissivity Dataset (Monthly, 0.05 deg, HDF5 V041)
- **Access Link:** [ASTER Dataset](https://search.earthdata.nasa.gov/search/granules?p=C2763268461-LPCLOUD)
- **Description:** Thermal infrared monitoring with emissivity and surface temperature mapping
- **Use Case:** Volcanic eruption detection, lava flow, and thermal anomaly analysis

---

### 8. 🌡️ Climate & Weather Reference Data

#### Temperature & Humidity
- **Dataset:** Maryland Temperature Humidity Dataset
- **Local Path:** `Echo Explorer/NASA SAR Data/Maryland_Temperature_Humidity_1319/data`
- **Format:** `.csv` files (e.g., `RelativeHumidity_20130905-20130918_preCal_office.csv`)
- **Description:** Ground-based temperature and humidity measurements
- **Use Case:** Climate model validation and local weather pattern analysis

#### Water Vapor Data
- **Dataset:** SAFARI 2000 MODIS MOD05_L2 Water Vapor Data (Binary Format)
- **Download Link:** [modis_MOD05_watervapor_812.zip](https://data.ornldaac.earthdata.nasa.gov/protected/bundle/modis_MOD05_watervapor_812.zip)
- **Source:** https://data.nasa.gov/
- **Description:** MODIS atmospheric water vapor measurements
- **Use Case:** Rainfall estimation and drought analysis

---



---

## 🔗 Data Sources

- **NASA Earthdata:** https://data.nasa.gov/
- **ORNL DAAC:** https://data.ornldaac.earthdata.nasa.gov/
- **NASA LARC ASDC:** https://asdc.larc.nasa.gov/
- **NASA PO.DAAC:** https://search.earthdata.nasa.gov/


---

## 🏠 Hero & Index Section  

The **Hero & Index** section introduces *Echo Explorer’s* vision:  
- Combining **radar insights** with **Earth intelligence**.  
- A user-friendly landing interface designed for **clarity, accessibility, and global impact**.  
- Establishes the project’s mission to bridge **space technology** with **climate action**.  


<img width="1766" height="844" alt="image" src="https://github.com/user-attachments/assets/c6ef5e6e-1dcd-4756-82c1-fea707c7b72a" />


## 🛰️ CosmoRadar System  

The **CosmoRadar System** is a radar-based Earth observation module designed to transform complex satellite data into **clear, actionable insights**. It shows how advanced radar technology can be applied to:  

- **Detect environmental changes** such as deforestation, flooding, or land shifts.  
- **Provide early warnings** for climate events and natural disasters.  
- **Support informed decision-making** through accurate and detailed geospatial intelligence.  

This system bridges the gap between **space technology and real-world impact**, helping communities, policymakers, and organizations respond more effectively to environmental challenges.  


<img width="1849" height="913" alt="Screenshot 2025-09-01 171831" src="https://github.com/user-attachments/assets/298fb426-4fc7-4b85-8b53-5c85b1deae07" />
<img width="1847" height="905" alt="Screenshot 2025-09-01 172041" src="https://github.com/user-attachments/assets/cf6e20f7-e4c7-4593-94f0-46f60e82946b" />


## 📡 SAR Data Analysis & Climate Risk Prediction  

This module focuses on **SAR-driven climate intelligence**, powered by:  
- **Data Science workflows** for processing and analyzing large-scale radar datasets.  
- **Machine Learning & Deep Learning pipelines** for climate risk modeling.  
- **Predictive frameworks** capable of anticipating environmental and societal risks.  

<img width="1142" height="2520" alt="image" src="https://github.com/user-attachments/assets/0c688e61-5ce5-43b3-ba82-28cad0aea3fb" />
<img width="1126" height="832" alt="Screenshot 2025-09-19 151506" src="https://github.com/user-attachments/assets/ed334335-feb4-4451-ac7f-1e50c5e90075" />




## 🌐 Climate Disaster Risk Platform & Global Security Intelligence  

This module extends beyond climate monitoring to address **global resilience and security**. By integrating SAR analytics with AI-driven intelligence, the platform provides:  

- **Disaster Risk Forecasting** – anticipating the frequency, intensity, and impact of climate-related hazards.  
- **Vulnerability Mapping** – identifying regions, infrastructures, and populations most at risk.  
- **Predictive Security Intelligence** – delivering insights on climate-induced risks that may escalate into humanitarian crises or geopolitical instability.  

Together, these capabilities position the platform as a **decision-support system** for governments, humanitarian agencies, and international organizations working toward long-term resilience and global security.  

<img width="1155" height="922" alt="Screenshot 2025-09-16 034902" src="https://github.com/user-attachments/assets/5492c0d4-79cd-4958-af8d-04c622dd660d" />








## 🚀 Features

### Core Functionality

- **Climate Disaster Prediction**: ML-powered analysis for floods, fires, urban heat, and deforestation
- **Multi-format Support**: Handles .filt, .jpg, .png, .tif, and .tiff files
- **Interactive Dashboard**: Real-time system status and model performance metrics
- **Professional Reporting**: Detailed risk assessments with actionable recommendations

### Technical Highlights

- **Advanced ML Pipeline**: Ensemble methods with cross-validation and hyperparameter optimization
- **Professional UI/UX**: Responsive design with dark/light theme support
- **Robust Error Handling**: Comprehensive validation and fallback mechanisms
- **Performance Monitoring**: Built-in metrics tracking and model evaluation

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask App      │    │  ML Pipeline    │
│  (HTML/CSS/JS)  │◄──►│   (app.py)       │◄──►│ (model_trainer) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Data Processor  │
                       │ (data_processor) │
                       └──────────────────┘
                                │
                                ▼
                       ┌────────────────────┐
                       │   NASA SAR Data    │
                       │ (.filt, .tif, etc) │
                       └────────────────────┘
```

## 📋 Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB recommended for large datasets)
- 2GB free disk space
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🛠️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/echo-explorer.git
cd echo-explorer
```

### 2. Create Virtual Environment

```bash
python -m venv echo_env

# Windows
echo_env\Scripts\activate

# macOS/Linux
source echo_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Directory Structure Setup

Ensure your project follows this structure:

```
echo-explorer/
├── app.py                 # Main Flask application
├── data_processor.py      # SAR data processing pipeline
├── model_trainer.py       # ML model training and evaluation
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main web interface
├── static/
│   ├── css/
│   ├── js/
│   └── assets/           # Images and media files
└── data/                 # NASA SAR datasets (create this folder)
    ├── forest_fire/
    ├── flood_data/
    ├── urban_heat/
    └── urban_classification/
```

### 5. Data Preparation

Place your NASA SAR data in the appropriate directories:

```bash
mkdir -p data/forest_fire data/flood_data data/urban_heat data/urban_classification
```

Copy your NASA datasets to these folders according to disaster type.

### 6. Model Training

```bash
# Process the SAR datasets
python data_processor.py

# Train the ML models
python model_trainer.py
```

### 7. Launch Application

```bash
python app.py
```

Navigate to `http://localhost:5000` in your web browser.

## 📊 Model Performance

### Current Metrics (Validation Set)

| Model | Accuracy | Precision | Recall | F1-Score | CV Score |
|-------|----------|-----------|--------|----------|----------|
| **Support Vector Machine** | 96.55% | 96.60% | 96.55% | 96.55% | 95.75% |
| Gradient Boosting | 96.55% | 96.92% | 96.55% | 96.54% | 96.15% |
| Random Forest | 96.55% | 96.65% | 96.55% | 96.53% | 95.38% |
| Logistic Regression | 94.25% | 94.54% | 94.25% | 94.23% | 95.37% |

*Note: Performance metrics based on processed NASA SAR datasets with 5-fold cross-validation.*

## 🔧 Usage Guide

### Web Interface

1. **Access the Dashboard**: Open `http://localhost:5000`
2. **Select Analysis Type**: Choose from Forest Fire, Flood, or Urban Heat analysis
3. **Data Input Options**:
   - Upload SAR images (recommended)
   - Enter manual feature values
4. **View Results**: Receive detailed risk assessment with confidence scores

### API Endpoints

```python
# Prediction endpoint
POST /predict
Content-Type: multipart/form-data

# Parameters:
# - image: SAR image file (optional)
# - features: comma-separated feature values (optional)
# - data_type: 'forest', 'wetland', or 'urban'
```

### Feature Input Format

For manual feature input, provide 15 comma-separated numerical values:

```
156.5,45.2,89.1,255.0,178.3,0.85,0.92,2.1,8.5,12.4,8.9,1250.5,890.2,0.65,0.72
```

## 🧪 Testing & Validation

### Run Unit Tests

```bash
python -m pytest tests/ -v
```

### Validate Model Performance

```bash
python model_trainer.py --validate
```

### Data Quality Checks

```bash
python data_processor.py --verify
```

## 🔍 Supported File Formats

### SAR Data Formats

- **.filt**: NASA GOES Imager format (specialized binary)
- **.tif/.tiff**: GeoTIFF satellite imagery
- **.jpg/.png**: Standard image formats (converted from SAR)

### Feature Specifications

The system extracts 15 specialized features from SAR data:

1. **Statistical Features**: Mean, std, min, max, median intensity
2. **Texture Features**: GLCM homogeneity, energy, entropy, contrast, correlation
3. **Gradient Features**: Gradient magnitude statistics
4. **Frequency Features**: FFT-based spectral analysis
5. **Domain-Specific**: Forest index, water coverage, or urban density

## ⚙️ Configuration

### Environment Variables

```bash
# Optional: Set data path
export SAR_DATA_PATH=/path/to/your/sar/data

# Optional: Set model cache directory
export MODEL_CACHE_DIR=/path/to/model/cache
```

### Model Configuration

Edit `model_trainer.py` to adjust:

- Cross-validation folds
- Hyperparameter search ranges
- Feature selection criteria
- Performance thresholds

## 🚨 Troubleshooting

### Common Issues

1. **Model files not found**
   ```bash
   # Solution: Run model training
   python model_trainer.py
   ```

2. **Image processing errors**
   ```bash
   # Check file format and size
   # Supported: .jpg, .png, .tif, .tiff, .filt
   # Max size: 16MB
   ```

3. **Memory issues during training**
   ```bash
   # Reduce dataset size in data_processor.py
   # Adjust limit_per_type parameter
   ```

### Performance Optimization

- **Memory**: Increase system RAM for larger datasets
- **Processing**: Use SSD storage for faster I/O operations
- **Training**: Consider GPU acceleration for neural networks

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test thoroughly
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open Pull Request

### Code Standards

- Follow PEP 8 Python style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📈 Roadmap

### Planned Features

- [ ] Real-time satellite data integration
- [ ] Advanced ensemble methods
- [ ] API rate limiting and authentication
- [ ] Automated model retraining pipeline


##  Acknowledgments

- **NASA Space Apps Challenge 2025** for the opportunity and datasets
- **NASA Earth Science Division** for SAR data access
- **Open Source Community** for excellent Python libraries

## 📞 Contact & Support

### Project Maintainer
**Ashabul Yamin Tuhin**  
📧 Email: ashabulyamintuhin@gmail.com  
🔗 GitHub: [CryoYam1n](https://github.com/your-infernoYam1n)




---

<div align="center">

**🌍 Empowering climate science through space technology 🛰️**

Developed for for NASA Space Apps Challenge 2025

[![NASA](https://img.shields.io/badge/NASA-Space%20Apps-orange.svg)](https://spaceappschallenge.org/)
[![Earth](https://img.shields.io/badge/Planet-Earth-blue.svg)](https://earthobservatory.nasa.gov/)

</div>
