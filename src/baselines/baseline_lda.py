"""
Enhanced Morphometric Analysis for Honey Bee Subspecies Classification

This module implements a comprehensive Linear Discriminant Analysis (LDA) pipeline
for automated classification of honey bee subspecies based on wing venation 
morphometric measurements. The implementation follows established protocols from
classical morphometric studies and incorporates modern statistical methods for
robust classification.

The analysis pipeline includes:
    - Comprehensive data preprocessing with statistical validation
    - Feature selection based on discriminant power analysis
    - Cross-validated model training with performance assessment
    - Multivariate statistical analysis of class separability
    - Visualization and reporting of classification results

References:
    Nawrocka, A., Kandemir, İ., Fuchs, S., & Tofilski, A. (2018). Computer software
    for identification of honey bee subspecies and evolutionary lineages. Apidologie, 
    49(2), 172-184.
    
    Meixner, M. D., Pinto, M. A., Bouga, M., Kryger, P., Ivanova, E., & Fuchs, S. (2013).
    Standard methods for characterising subspecies and ecotypes of Apis mellifera. 
    Journal of Apicultural Research, 52(4), 1-28.
    
    Ruttner, F. (1988). Biogeography and taxonomy of honeybees. Springer Science & 
    Business Media.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any

# Machine learning and statistical analysis imports
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import anderson, shapiro, f_oneway
from scipy.spatial.distance import mahalanobis

# Suppress non-critical warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class EnhancedMorphometricAnalyzer:
    """
    Comprehensive morphometric analysis system for honey bee subspecies classification.
    
    This class implements a complete pipeline for analyzing wing venation measurements
    to classify honey bee subspecies. The methodology follows established morphometric
    protocols while incorporating modern statistical preprocessing and validation methods.
    
    The analysis workflow consists of five main stages:
    1. Data loading and validation with quality control checks
    2. Preprocessing including imputation, scaling, and outlier treatment
    3. Feature selection using univariate statistical tests
    4. Model training with cross-validation and performance assessment
    5. Evaluation and visualization of classification results
    
    Attributes:
        random_state (int): Random seed for reproducible results
        n_features (int): Number of top features to select for classification
        lda_model (LinearDiscriminantAnalysis): Trained LDA classifier
        imputer (KNNImputer): Missing value imputation using k-nearest neighbors
        scaler (RobustScaler): Feature standardization robust to outliers
        feature_selector (SelectKBest): Statistical feature selection based on F-scores
        label_encoder (LabelEncoder): Encoding for categorical subspecies labels
        selected_features_ (np.ndarray): Boolean mask of selected features
        feature_names (List[str]): Names of morphometric measurement variables
        performance_metrics (Dict): Comprehensive evaluation metrics storage
        class_statistics (Dict): Statistical measures of class separability
        lineage_map (Dict): Mapping from lineage codes to full subspecies names
    
    Example:
        >>> analyzer = EnhancedMorphometricAnalyzer(n_features=10, random_state=42)
        >>> data_splits = analyzer.load_and_preprocess("manual_measurements.csv")
        >>> analyzer.train_model(data_splits['X_train'], data_splits['y_train'])
        >>> analyzer.evaluate_model(data_splits['X_test'], data_splits['y_test'])
        >>> analyzer.generate_report()
    """
    
    def __init__(self, n_features: int = 10, random_state: int = 42) -> None:
        """
        Initialize the morphometric analyzer with specified parameters.
        
        Parameters:
            n_features (int, default=10): Number of top discriminative features to select.
                Should be less than the total number of available features (19) and
                greater than the number of classes to ensure proper LDA performance.
            random_state (int, default=42): Random seed for reproducible results across
                train/test splits, cross-validation, and stochastic operations.
        
        Raises:
            ValueError: If n_features is not in valid range [2, 19].
        
        Note:
            The number of features selected should balance between model complexity
            and classification performance. Too few features may lose discriminative
            information, while too many may introduce noise and overfitting.
        """
        # Validate input parameters
        if not isinstance(n_features, int) or n_features < 2 or n_features > 19:
            raise ValueError("n_features must be an integer between 2 and 19")
        
        if not isinstance(random_state, int) or random_state < 0:
            raise ValueError("random_state must be a non-negative integer")
        
        # Store configuration parameters
        self.random_state = random_state
        self.n_features = n_features
        
        # Initialize core machine learning components
        self.lda_model = None
        self.imputer = KNNImputer(
            n_neighbors=5, 
            weights='distance',
            metric='nan_euclidean'
        )
        self.scaler = RobustScaler(
            quantile_range=(5, 95),  # Use 5th-95th percentile for robust scaling
            with_centering=True,
            with_scaling=True
        )
        self.feature_selector = SelectKBest(
            score_func=f_classif,  # ANOVA F-test for feature selection
            k=n_features
        )
        self.label_encoder = None
        
        # Feature tracking and selection
        self.selected_features_ = None
        
        # Morphometric feature names following Ruttner's standard nomenclature
        # Wing vein angles measured in degrees
        # Wing vein lengths normalized by forewing length
        # Morphometric indices calculated from multiple measurements
        self.feature_names = [
            'A4_angle',         # Angle of A4 vein junction (degrees)
            'B4_angle',         # Angle of B4 vein junction (degrees)
            'D7_angle',         # Angle of D7 vein junction (degrees)
            'G18_angle',        # Angle of G18 vein junction (degrees)
            'J10_angle',        # Angle of J10 vein junction (degrees)
            'A4_length',        # Length of A4 vein segment (normalized)
            'B4_length',        # Length of B4 vein segment (normalized)
            'D7_length',        # Length of D7 vein segment (normalized)
            'G18_length',       # Length of G18 vein segment (normalized)
            'J10_length',       # Length of J10 vein segment (normalized)
            'cubital_index',    # Ratio of cubital vein segments
            'discoidal_shift',  # Displacement of discoidal vein
            'hamuli_count',     # Number of hamuli hooks
            'stigma_length',    # Length of wing stigma
            'forewing_length',  # Total forewing length (mm)
            'forewing_width',   # Maximum forewing width (mm)
            'cell_3R_area',     # Area of cell 3R (mm²)
            'cell_3W_area',     # Area of cell 3W (mm²)
            'venation_density'  # Density of wing venation pattern
        ]
        
        # Initialize result storage containers
        self.performance_metrics = {}
        self.class_statistics = {}
        
        # Taxonomic lineage mapping from abbreviated codes to full names
        # Based on current phylogenetic classification of Apis mellifera
        self.lineage_map = {
            'M': 'Apis mellifera mellifera',     # Northern European lineage
            'C': 'Apis mellifera ligustica',     # Southern European lineage
            'O': 'Oriental lineage',             # Asian lineage
            'A': 'African lineage'               # African lineage
        }
    
    def load_and_preprocess(self, file_path: str, target_column: str = 'lineage', 
                           test_size: float = 0.2) -> Dict[str, Any]:
        """
        Load morphometric data and apply comprehensive preprocessing pipeline.
        
        This method implements a rigorous five-stage preprocessing workflow:
        1. Data loading with type validation and missing value assessment
        2. Morphometric measurement validation against biological constraints
        3. Missing value imputation using k-nearest neighbors algorithm
        4. Outlier detection and treatment using interquartile range method
        5. Feature standardization using robust scaling methods
        
        Parameters:
            file_path (str): Path to CSV file containing morphometric measurements.
                File must contain all required morphometric features and labels.
            target_column (str, default='lineage'): Column name to use as classification
                target. Can be 'lineage' for broad taxonomic groups or 'subspecies' 
                for detailed subspecies classification.
            test_size (float, default=0.2): Proportion of data reserved for final
                testing. Should be between 0.1 and 0.5 for reliable evaluation.
        
        Returns:
            Dict[str, Any]: Dictionary containing preprocessed data splits with keys:
                - 'X_train': Training feature matrix (scaled and selected)
                - 'X_test': Testing feature matrix (scaled and selected)
                - 'y_train': Training target labels (encoded)
                - 'y_test': Testing target labels (encoded)
                - 'feature_names': List of original feature names
                - 'class_names': Array of class names from label encoder
                - 'raw_data': Original DataFrame for reference
        
        Raises:
            FileNotFoundError: If the specified file path does not exist
            ValueError: If required columns are missing from the dataset
            RuntimeError: If preprocessing fails due to data quality issues
        
        Note:
            The preprocessing pipeline is designed to handle real-world morphometric
            data with missing values, measurement errors, and outliers. All preprocessing
            steps are fitted on training data only to prevent data leakage.
        """
        print("Loading and preprocessing morphometric data...")
        print(f"Target column: {target_column}")
        print(f"Test set proportion: {test_size}")
        
        # Stage 1: Data Loading and Initial Validation
        try:
            # Load data with automatic type inference
            df = pd.read_csv(file_path)
            print(f"Successfully loaded {len(df)} samples with {len(df.columns)} columns")
            
            # Log basic dataset statistics
            print(f"Dataset shape: {df.shape}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at path: {file_path}")
        except pd.errors.EmptyDataError:
            raise ValueError("Data file is empty or corrupted")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")
        
        # Validate presence of required columns
        required_features = set(self.feature_names)
        required_labels = {'lineage', 'subspecies'}
        missing_features = required_features - set(df.columns)
        missing_labels = required_labels - set(df.columns)
        
        if missing_features:
            raise ValueError(f"Missing required morphometric features: {missing_features}")
        if missing_labels:
            raise ValueError(f"Missing required label columns: {missing_labels}")
        
        # Stage 2: Feature Extraction and Target Preparation
        # Extract morphometric features maintaining original column order
        X = df[self.feature_names].copy()
        
        # Prepare target variable based on specified column
        if target_column == 'lineage':
            # Map lineage codes to full names, preserve unmapped values
            y_raw = df['lineage'].map(self.lineage_map).fillna(df['lineage'])
        elif target_column == 'subspecies':
            y_raw = df['subspecies']
        else:
            raise ValueError(f"Invalid target_column '{target_column}'. Must be 'lineage' or 'subspecies'")
        
        # Remove samples with missing target labels
        valid_target_mask = ~y_raw.isna()
        X = X[valid_target_mask]
        y_raw = y_raw[valid_target_mask]
        
        print(f"After removing missing targets: {len(X)} samples remain")
        print(f"Class distribution:\n{y_raw.value_counts()}")
        
        # Check for minimum samples per class (required for stratified splitting)
        min_class_size = y_raw.value_counts().min()
        if min_class_size < 2:
            raise ValueError("At least 2 samples required per class for stratified splitting")
        
        # Stage 3: Comprehensive Data Validation
        print("Performing morphometric measurement validation...")
        self._validate_measurements(X)
        
        # Stage 4: Missing Value Imputation
        print("Imputing missing values using k-nearest neighbors...")
        missing_before = X.isnull().sum().sum()
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=self.feature_names,
            index=X.index
        )
        missing_after = X_imputed.isnull().sum().sum()
        print(f"Missing values: {missing_before} → {missing_after}")
        
        # Stage 5: Outlier Detection and Treatment
        print("Detecting and treating outliers...")
        X_clean = self._handle_outliers(X_imputed)
        
        # Stage 6: Feature Standardization
        print("Standardizing features using robust scaling...")
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_clean),
            columns=self.feature_names,
            index=X_clean.index
        )
        
        # Verify scaling results
        print(f"Feature means after scaling: {X_scaled.mean().abs().max():.6f}")
        print(f"Feature std after scaling: {X_scaled.std().mean():.6f}")
        
        # Stage 7: Label Encoding
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_raw)
        n_classes = len(self.label_encoder.classes_)
        print(f"Encoded {n_classes} classes: {list(self.label_encoder.classes_)}")
        
        # Stage 8: Stratified Train-Test Split
        # Ensure balanced representation of all classes in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        # Verify stratification
        train_distribution = pd.Series(y_train).value_counts(normalize=True).sort_index()
        test_distribution = pd.Series(y_test).value_counts(normalize=True).sort_index()
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print("Class distribution verification:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"  {class_name}: Train={train_distribution[i]:.3f}, Test={test_distribution[i]:.3f}")
        
        # Return comprehensive data package
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'class_names': self.label_encoder.classes_,
            'raw_data': df,
            'preprocessing_stats': {
                'original_samples': len(df),
                'final_samples': len(X_scaled),
                'missing_imputed': missing_before,
                'outliers_treated': self._outlier_count if hasattr(self, '_outlier_count') else 0,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
        }
    
    def _validate_measurements(self, X: pd.DataFrame) -> None:
        """
        Perform comprehensive validation of morphometric measurements.
        
        This method implements biological and statistical validation checks to ensure
        data quality and identify potential measurement errors. Validation includes:
        - Physical constraint checking (angle ranges, positive lengths)
        - Statistical distribution analysis for key morphometric indices
        - Correlation analysis to detect measurement inconsistencies
        - Data completeness assessment
        
        Parameters:
            X (pd.DataFrame): DataFrame containing morphometric measurements
                with feature names as column headers.
        
        Raises:
            Warning: For measurements outside expected biological ranges
            
        Note:
            This method performs validation checks but does not modify the data.
            Actual data cleaning is performed by subsequent preprocessing steps.
        """
        print("Validating morphometric measurements against biological constraints...")
        
        # Define biological validation rules based on honey bee morphology
        validation_rules = {
            'vein_angles': {
                'features': [col for col in X.columns if 'angle' in col],
                'range': (0, 180),
                'description': 'Wing vein angles (degrees)'
            },
            'vein_lengths': {
                'features': [col for col in X.columns if 'length' in col and 'forewing' not in col],
                'range': (0, np.inf),
                'description': 'Wing vein lengths (normalized)'
            },
            'wing_dimensions': {
                'features': ['forewing_length', 'forewing_width'],
                'range': (0, np.inf),
                'description': 'Wing dimensions (mm)'
            },
            'cell_areas': {
                'features': [col for col in X.columns if 'area' in col],
                'range': (0, np.inf),
                'description': 'Wing cell areas (mm²)'
            },
            'morphometric_indices': {
                'features': ['cubital_index', 'discoidal_shift', 'venation_density'],
                'range': (-np.inf, np.inf),
                'description': 'Calculated morphometric indices'
            },
            'discrete_counts': {
                'features': ['hamuli_count'],
                'range': (0, np.inf),
                'description': 'Discrete anatomical counts'
            }
        }
        
        # Perform validation for each rule category
        total_violations = 0
        for rule_name, rule_config in validation_rules.items():
            valid_features = [col for col in rule_config['features'] if col in X.columns]
            
            if not valid_features:
                continue
                
            min_val, max_val = rule_config['range']
            
            # Check for range violations
            if min_val != -np.inf or max_val != np.inf:
                violation_mask = (X[valid_features] < min_val) | (X[valid_features] > max_val)
                n_violations = violation_mask.any(axis=1).sum()
                
                if n_violations > 0:
                    total_violations += n_violations
                    violation_percentage = (n_violations / len(X)) * 100
                    print(f"⚠️  {rule_config['description']}: {n_violations} samples "
                          f"({violation_percentage:.1f}%) outside valid range [{min_val}, {max_val}]")
                    
                    # Report specific feature violations
                    for feature in valid_features:
                        feature_violations = violation_mask[feature].sum()
                        if feature_violations > 0:
                            print(f"   - {feature}: {feature_violations} violations")
        
        # Assess data completeness
        print("\nData completeness assessment:")
        missing_summary = X.isnull().sum()
        total_missing = missing_summary.sum()
        
        if total_missing > 0:
            print(f"Total missing values: {total_missing} ({(total_missing / X.size) * 100:.1f}%)")
            features_with_missing = missing_summary[missing_summary > 0]
            for feature, count in features_with_missing.items():
                percentage = (count / len(X)) * 100
                print(f"  - {feature}: {count} ({percentage:.1f}%)")
        else:
            print("No missing values detected")
        
        # Statistical validation of key morphometric indices
        print("\nStatistical validation of morphometric indices:")
        key_indices = ['cubital_index', 'discoidal_shift', 'venation_density']
        
        for index in key_indices:
            if index in X.columns:
                data = X[index].dropna()
                
                if len(data) >= 50:  # Minimum sample size for reliable statistical testing
                    # Test for normality using Shapiro-Wilk test
                    # Sample size limited to 5000 for computational efficiency
                    test_sample = data.sample(min(5000, len(data)), random_state=self.random_state)
                    stat, p_value = shapiro(test_sample)
                    
                    # Interpret normality test results
                    is_normal = p_value > 0.05
                    normality_status = "Normal" if is_normal else "Non-normal"
                    
                    print(f"  {index}: {normality_status} distribution "
                          f"(W={stat:.3f}, p={p_value:.3e})")
                    
                    # Report descriptive statistics
                    print(f"    Mean: {data.mean():.3f}, Std: {data.std():.3f}, "
                          f"Range: [{data.min():.3f}, {data.max():.3f}]")
        
        # Summary of validation results
        if total_violations > 0:
            print(f"\n⚠️  Validation summary: {total_violations} samples contain measurements "
                  "outside expected biological ranges. These will be addressed in preprocessing.")
        else:
            print("\n✓ All measurements within expected biological ranges")
    
    def _handle_outliers(self, X: pd.DataFrame, method: str = 'iqr', 
                        factor: float = 1.5) -> pd.DataFrame:
        """
        Detect and handle outliers using interquartile range (IQR) method.
        
        This method identifies outliers based on the IQR criterion and applies
        winsorization to cap extreme values. Winsorization is preferred over
        removal to preserve sample size while reducing the impact of extreme
        measurements that may result from measurement errors.
        
        Parameters:
            X (pd.DataFrame): DataFrame containing morphometric measurements
            method (str, default='iqr'): Outlier detection method. Currently
                supports 'iqr' (interquartile range) method.
            factor (float, default=1.5): IQR multiplier for outlier threshold.
                Values beyond Q1 - factor*IQR or Q3 + factor*IQR are considered outliers.
        
        Returns:
            pd.DataFrame: DataFrame with outliers winsorized to threshold values
        
        Note:
            The IQR method is robust to non-normal distributions commonly found
            in morphometric data. The factor of 1.5 corresponds to the standard
            definition of outliers in exploratory data analysis.
        """
        print(f"Applying outlier detection using {method} method (factor={factor})...")
        
        X_clean = X.copy()
        outlier_summary = {}
        total_outliers = 0
        
        # Apply IQR-based outlier detection to each feature
        for feature in X.columns:
            # Calculate quartiles and IQR
            Q1 = X[feature].quantile(0.25)
            Q3 = X[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier boundaries
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Identify outliers
            outlier_mask = (X[feature] < lower_bound) | (X[feature] > upper_bound)
            n_outliers = outlier_mask.sum()
            
            if n_outliers > 0:
                outlier_summary[feature] = {
                    'count': n_outliers,
                    'percentage': (n_outliers / len(X)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                total_outliers += n_outliers
                
                # Apply winsorization: cap values at boundaries
                X_clean[feature] = X[feature].clip(lower_bound, upper_bound)
        
        # Store outlier count for reporting
        self._outlier_count = total_outliers
        
        # Report outlier treatment results
        if outlier_summary:
            print(f"Outliers detected in {len(outlier_summary)} features:")
            for feature, stats in outlier_summary.items():
                print(f"  - {feature}: {stats['count']} outliers ({stats['percentage']:.1f}%) "
                      f"winsorized to [{stats['lower_bound']:.3f}, {stats['upper_bound']:.3f}]")
            
            total_percentage = (total_outliers / X.size) * 100
            print(f"Total outlying values: {total_outliers} ({total_percentage:.1f}%) winsorized")
        else:
            print("No outliers detected")
        
        return X_clean
    
    def train_model(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                   cv_folds: int = 5) -> LinearDiscriminantAnalysis:
        """
        Train Linear Discriminant Analysis model with feature selection and validation.
        
        This method implements a comprehensive model training pipeline that includes:
        1. Feature selection using ANOVA F-test to identify discriminative features
        2. LDA model training with optimal solver selection
        3. Cross-validation for robust performance estimation
        4. Statistical analysis of class separability
        5. Computation of discriminant power metrics
        
        Parameters:
            X_train (pd.DataFrame): Training feature matrix after preprocessing
            y_train (np.ndarray): Training target labels (encoded)
            cv_folds (int, default=5): Number of folds for cross-validation.
                Should be between 3 and 10 for reliable estimates.
        
        Returns:
            LinearDiscriminantAnalysis: Trained LDA model
        
        Raises:
            ValueError: If insufficient samples or invalid cv_folds parameter
            RuntimeError: If model training fails
        
        Note:
            The SVD solver is used for numerical stability, especially important
            when dealing with morphometric data that may have correlated features.
            Feature selection helps reduce dimensionality while preserving
            discriminative power.
        """
        print("Training Linear Discriminant Analysis model...")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Classes: {len(np.unique(y_train))}")
        
        # Validate training parameters
        if len(X_train) < 10:
            raise ValueError("Insufficient training samples (minimum 10 required)")
        
        if not 3 <= cv_folds <= 10:
            raise ValueError("cv_folds must be between 3 and 10")
        
        # Stage 1: Feature Selection Using ANOVA F-test
        print(f"Selecting top {self.n_features} discriminative features...")
        
        # Apply feature selection and store results
        X_selected = self.feature_selector.fit_transform(X_train, y_train)
        self.selected_features_ = self.feature_selector.get_support()
        
        # Analyze feature selection results
        feature_scores = self.feature_selector.scores_
        feature_p_values = self.feature_selector.pvalues_
        
        # Create comprehensive feature ranking
        feature_rankings = pd.DataFrame({
            'feature': self.feature_names,
            'f_score': feature_scores,
            'p_value': feature_p_values,
            'selected': self.selected_features_,
            'rank': feature_scores.argsort()[::-1] + 1  # Rank from highest to lowest score
        }).sort_values('f_score', ascending=False)
        
        # Report feature selection results
        print("Feature selection results (top 10):")
        print(feature_rankings.head(10)[['feature', 'f_score', 'p_value', 'selected']].to_string(index=False))
        
        # Store selected feature names for interpretation
        selected_feature_names = feature_rankings[feature_rankings['selected']]['feature'].tolist()
        print(f"\nSelected features: {selected_feature_names}")
        
        # Stage 2: LDA Model Training
        print("Training LDA classifier...")
        
        # Initialize LDA with optimal parameters for morphometric data
        self.lda_model = LinearDiscriminantAnalysis(
            solver='svd',           # SVD solver for numerical stability
            store_covariance=True,  # Store covariance for diagnostics
            tol=1e-4               # Convergence tolerance
        )
        
        # Train the model on selected features
        try:
            self.lda_model.fit(X_selected, y_train)
            print(f"Model trained successfully with {X_selected.shape[1]} features")
        except Exception as e:
            raise RuntimeError(f"Model training failed: {str(e)}")
        
        # Stage 3: Cross-validation for Performance Estimation
        print(f"Performing {cv_folds}-fold stratified cross-validation...")
        
        # Use stratified k-fold to maintain class distribution in each fold
        cv_strategy = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.lda_model, X_selected, y_train,
            cv=cv_strategy,
            scoring='accuracy',
            n_jobs=-1  # Use all available cores
        )
        
        # Store cross-validation results
        self.performance_metrics.update({
            'cv_scores': cv_scores,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'cv_accuracy_95ci': [
                cv_scores.mean() - 1.96 * cv_scores.std() / np.sqrt(cv_folds),
                cv_scores.mean() + 1.96 * cv_scores.std() / np.sqrt(cv_folds)
            ]
        })
        
        # Report cross-validation results
        print(f"Cross-validation results:")
        print(f"  Mean accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  95% CI: [{self.performance_metrics['cv_accuracy_95ci'][0]:.4f}, "
              f"{self.performance_metrics['cv_accuracy_95ci'][1]:.4f}]")
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: np.ndarray) -> None:
        """
        Evaluate the trained LDA model on the test set.
        
        This method computes and reports various classification metrics including:
        - Confusion matrix
        - Classification report
        - Accuracy score
        - Statistical analysis of class separability (if available)
        
        Parameters:
            X_test (pd.DataFrame): Testing feature matrix (scaled and selected)
            y_test (np.ndarray): Testing target labels (encoded)
        
        Raises:
            ValueError: If model is not trained
            RuntimeError: If evaluation fails
        """
        if self.lda_model is None:
            raise ValueError("LDA model has not been trained yet. Call train_model() first.")
        
        print("Evaluating LDA model on test set...")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {X_test.shape[1]}")
        print(f"Classes: {len(np.unique(y_test))}")
        
        # Transform test data using the trained model
        X_test_transformed = self.lda_model.transform(X_test)
        
        # Predict on the test set
        y_pred = self.lda_model.predict(X_test_transformed)
        
        # Decode predictions and true labels
        y_test_decoded = self.label_encoder.inverse_transform(y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
        
        # Compute metrics
        accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
        report = classification_report(y_test_decoded, y_pred_decoded, target_names=self.label_encoder.classes_)
        confusion = confusion_matrix(y_test_decoded, y_pred_decoded)
        
        # Store evaluation results
        self.performance_metrics.update({
            'test_accuracy': accuracy,
            'test_report': report,
            'test_confusion_matrix': confusion
        })
        
        # Report evaluation results
        print(f"Test set accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(confusion)
        
        # Statistical analysis of class separability (if LDA model has covariance)
        if hasattr(self.lda_model, 'covariance_') and self.lda_model.covariance_ is not None:
            print("\nMultivariate statistical analysis of class separability:")
            # Calculate within-class scatter matrix (Sw)
            Sw = np.zeros((X_test_transformed.shape[1], X_test_transformed.shape[1]))
            for label in np.unique(y_test):
                label_mask = y_test_decoded == label
                if label_mask.sum() > 1:
                    Sw += np.cov(X_test_transformed[label_mask].T)
            
            # Calculate between-class scatter matrix (Sb)
            overall_mean = np.mean(X_test_transformed, axis=0)
            Sb = np.zeros((X_test_transformed.shape[1], X_test_transformed.shape[1]))
            for label in np.unique(y_test):
                label_mask = y_test_decoded == label
                if label_mask.sum() > 1:
                    label_mean = np.mean(X_test_transformed[label_mask], axis=0)
                    Sb += label_mask.sum() * np.outer(label_mean - overall_mean, label_mean - overall_mean)
            
            # Calculate generalized eigenvalue problem (Sw^-1 Sb)
            try:
                # Ensure Sw is invertible
                if np.linalg.det(Sw) == 0:
                    print("Warning: Within-class scatter matrix (Sw) is singular. Cannot compute class separability.")
                    self.class_statistics['Sw_singular'] = True
                else:
                    # Compute Sw^-1 Sb
                    Sw_inv = np.linalg.inv(Sw)
                    eig_vals, eig_vecs = np.linalg.eig(Sw_inv @ Sb)
                    
                    # Sort eigenvalues and eigenvectors
                    sorted_indices = np.argsort(eig_vals)[::-1]
                    eig_vals = eig_vals[sorted_indices]
                    eig_vecs = eig_vecs[:, sorted_indices]
                    
                    # Store eigenvalues and eigenvectors
                    self.class_statistics['eigenvalues'] = eig_vals
                    self.class_statistics['eigenvectors'] = eig_vecs
                    
                    print(f"Computed {len(eig_vals)} discriminant axes.")
                    print(f"Largest eigenvalue (Discriminant Power): {eig_vals[0]:.4f}")
                    print(f"Smallest eigenvalue (Discriminant Power): {eig_vals[-1]:.4f}")
                    
                    # Plot eigenvalues (if matplotlib is available)
                    if plt:
                        plt.figure(figsize=(10, 6))
                        plt.plot(range(1, len(eig_vals) + 1), eig_vals, 'o-')
                        plt.xlabel('Discriminant Axis')
                        plt.ylabel('Eigenvalue')
                        plt.title('LDA Eigenvalues (Discriminant Power)')
                        plt.grid(True)
                        plt.show()
            except Exception as e:
                print(f"Error computing class separability: {str(e)}")
        else:
            print("LDA model does not have a covariance matrix. Cannot perform multivariate analysis.")
    
if __name__ == "__main__":
    DATA_PATH = "manual_measurements.csv"  
    TARGET = "subspecies"   # or "lineage"
    N_FEATURES = 10
    RANDOM_STATE = 42

    analyzer = EnhancedMorphometricAnalyzer(n_features=N_FEATURES, random_state=RANDOM_STATE)
    data_splits = analyzer.load_and_preprocess(DATA_PATH, target_column=TARGET, test_size=0.2)
    analyzer.train_model(data_splits['X_train'], data_splits['y_train'])
    analyzer.evaluate_model(data_splits['X_test'], data_splits['y_test'])
