# Let's create the main classifier file first
classifier_code = '''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class NDVILandCoverClassifier:
    """
    NDVI Land Cover Classification using Logistic Regression
    Specifically designed for competition with noisy time-series data
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.classes = ['Water', 'Impervious', 'Farm', 'Forest', 'Grass', 'Orchard']
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        try:
            data = pd.read_csv(filepath)
            print(f"Loaded data: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            return data
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            return None
            
    def get_ndvi_columns(self, data):
        """Extract NDVI time-series columns"""
        ndvi_cols = [col for col in data.columns if col.endswith('_N')]
        ndvi_cols.sort()  # Sort chronologically
        print(f"Found {len(ndvi_cols)} NDVI columns")
        return ndvi_cols
        
    def clean_ndvi_series(self, series):
        """Clean and denoise NDVI time series"""
        # Handle missing values with linear interpolation
        series_clean = series.interpolate(method='linear', limit_direction='both')
        
        # Fill remaining NaN with median
        series_clean = series_clean.fillna(series_clean.median())
        
        # Apply Savitzky-Golay filter for noise reduction
        if len(series_clean) >= 5:  # Need minimum points for filter
            try:
                series_smooth = savgol_filter(series_clean, window_length=5, polyorder=2)
                return pd.Series(series_smooth, index=series.index)
            except:
                return series_clean
        
        return series_clean
        
    def extract_features(self, ndvi_data):
        """Extract comprehensive features from NDVI time series"""
        features = []
        feature_names = []
        
        for idx, row in ndvi_data.iterrows():
            row_features = []
            
            # Clean the time series
            clean_series = self.clean_ndvi_series(row)
            
            # Basic statistical features
            row_features.extend([
                clean_series.mean(),
                clean_series.std(),
                clean_series.min(),
                clean_series.max(),
                clean_series.max() - clean_series.min(),  # range
                clean_series.median(),
                clean_series.quantile(0.25),
                clean_series.quantile(0.75),
                clean_series.skew(),
                clean_series.kurtosis()
            ])
            
            # Temporal features
            # Early season (first 1/3)
            early_season = clean_series[:len(clean_series)//3]
            mid_season = clean_series[len(clean_series)//3:2*len(clean_series)//3]
            late_season = clean_series[2*len(clean_series)//3:]
            
            row_features.extend([
                early_season.mean(),
                mid_season.mean(),
                late_season.mean(),
                mid_season.mean() - early_season.mean(),  # growth rate
                late_season.mean() - mid_season.mean(),   # decline rate
            ])
            
            # Trend features
            x = np.arange(len(clean_series))
            slope = np.polyfit(x, clean_series, 1)[0]
            row_features.append(slope)
            
            # Change point features
            diff_series = clean_series.diff().dropna()
            row_features.extend([
                diff_series.abs().mean(),  # average change
                diff_series.abs().max(),   # maximum change
                (diff_series > 0).sum(),   # number of increases
                (diff_series < 0).sum(),   # number of decreases
            ])
            
            features.append(row_features)
            
        # Define feature names for first iteration
        if not self.feature_names:
            self.feature_names = [
                'mean', 'std', 'min', 'max', 'range', 'median', 'q25', 'q75', 'skew', 'kurtosis',
                'early_mean', 'mid_mean', 'late_mean', 'growth_rate', 'decline_rate',
                'trend_slope', 'avg_change', 'max_change', 'n_increases', 'n_decreases'
            ]
            
        return np.array(features)
        
    def clean_and_preprocess(self, data):
        """Complete preprocessing pipeline"""
        print("Starting data preprocessing...")
        
        # Get NDVI columns
        ndvi_cols = self.get_ndvi_columns(data)
        
        if not ndvi_cols:
            raise ValueError("No NDVI columns found in data")
            
        # Extract NDVI data
        ndvi_data = data[ndvi_cols]
        
        # Extract features
        features = self.extract_features(ndvi_data)
        
        # Create feature dataframe
        feature_df = pd.DataFrame(features, columns=self.feature_names)
        
        # Add ID column
        feature_df['ID'] = data['ID'].values
        
        # Add class column if it exists (training data)
        if 'class' in data.columns:
            feature_df['class'] = data['class'].values
            
        print(f"Extracted {len(self.feature_names)} features")
        print("Preprocessing complete!")
        
        return feature_df
        
    def prepare_features_target(self, processed_data):
        """Prepare features and target for model training"""
        # Get feature columns (exclude ID and class)
        feature_cols = [col for col in processed_data.columns if col not in ['ID', 'class']]
        X = processed_data[feature_cols]
        
        # Prepare target if available
        if 'class' in processed_data.columns:
            y = processed_data['class']
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            return X, y_encoded
        else:
            return X, None
            
    def create_pipeline(self):
        """Create sklearn pipeline with preprocessing and model"""
        # Create preprocessing steps
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Create complete pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                random_state=self.random_state,
                max_iter=1000,
                C=1.0
            ))
        ])
        
        return pipeline
        
    def train_model(self, X, y):
        """Train the logistic regression model"""
        print("Training model...")
        
        # Create pipeline
        self.pipeline = self.create_pipeline()
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Train model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.pipeline.predict(X_train)
        val_pred = self.pipeline.predict(X_val)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, X, y, cv=5, scoring='accuracy')
        
        results = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return results
        
    def predict(self, X):
        """Make predictions on new data"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train_model first.")
            
        predictions = self.pipeline.predict(X)
        
        # Convert back to original labels
        pred_labels = self.label_encoder.inverse_transform(predictions)
        
        return pred_labels
        
    def create_submission(self, test_data, output_file='submission.csv'):
        """Create submission file from test data"""
        print("Creating submission file...")
        
        # Preprocess test data
        processed_test = self.clean_and_preprocess(test_data)
        X_test, _ = self.prepare_features_target(processed_test)
        
        # Make predictions
        predictions = self.predict(X_test)
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'ID': processed_test['ID'],
            'class': predictions
        })
        
        # Save submission
        submission.to_csv(output_file, index=False)
        print(f"Submission saved to {output_file}")
        print(f"Predictions distribution:")
        print(submission['class'].value_counts())
        
        return submission
'''

# Save the classifier file
with open('ndvi_land_cover_classifier.py', 'w') as f:
    f.write(classifier_code)

print("Created ndvi_land_cover_classifier.py")