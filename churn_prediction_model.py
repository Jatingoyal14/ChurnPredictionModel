# Enhanced Churn Prediction Model
# Complete ML Pipeline with Multiple Models and Advanced Features

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictionSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.results = {}
        
    def load_data(self, file_path):
        """Load and initial data exploration"""
        print("📊 Loading data...")
        self.data = pd.read_csv(file_path)
        print(f"Data shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        return self.data
    
    def explore_data(self):
        """Comprehensive EDA"""
        print("\n📈 Data Exploration")
        print("="*50)
        
        # Basic info
        print(f"Dataset shape: {self.data.shape}")
        print(f"Missing values:\n{self.data.isnull().sum()}")
        print(f"Data types:\n{self.data.dtypes}")
        
        # Target distribution
        target_dist = self.data['Closed'].value_counts()
        print(f"\nChurn Distribution:")
        print(f"Retained (0): {target_dist[0]} ({target_dist[0]/len(self.data)*100:.1f}%)")
        print(f"Churned (1): {target_dist[1]} ({target_dist[1]/len(self.data)*100:.1f}%)")
        
        return self.data.describe()
    
    def engineer_features(self):
        """Advanced feature engineering"""
        print("\n⚙️ Engineering Features...")
        
        # Create feature engineered dataset
        df = self.data.copy()
        
        # Age groups
        df['Age_Group'] = pd.cut(df['Age'], 
                                bins=[0, 30, 45, 60, 120], 
                                labels=['Young', 'Middle', 'Senior', 'Elder'])
        
        # Credit score categories
        df['Credit_Category'] = pd.cut(df['Credit Score'], 
                                      bins=[0, 500, 650, 750, 900], 
                                      labels=['Poor', 'Fair', 'Good', 'Excellent'])
        
        # Income categories
        df['Income_Category'] = pd.cut(df['Estimated Yearly Income'], 
                                      bins=[0, 100000, 300000, 500000, 999999999], 
                                      labels=['Low', 'Medium', 'High', 'Premium'])
        
        # Loyalty score (customer since / age)
        df['Loyalty_Score'] = df['Customer Since'] / df['Age']
        df['Loyalty_Score'] = df['Loyalty_Score'].fillna(0)
        
        # High value customer
        df['High_Value_Customer'] = (df['Estimated Yearly Income'] > 400000).astype(int)
        
        # Has current account
        df['Has_Current_Account'] = (df['Current Account'] > 0).astype(int)
        
        self.engineered_data = df
        print(f"Features after engineering: {df.shape[1]}")
        return df
    
    def preprocess_data(self):
        """Preprocessing pipeline"""
        print("\n🔧 Preprocessing Data...")
        
        df = self.engineered_data.copy()
        
        # Separate features and target
        X = df.drop('Closed', axis=1)
        y = df['Closed']
        
        # Identify categorical and numerical columns
        categorical_features = ['Geography', 'Gender', 'Age_Group', 'Credit_Category', 'Income_Category']
        numerical_features = ['Credit Score', 'Age', 'Customer Since', 'Current Account', 
                             'Num of products', 'UPI Enabled', 'Estimated Yearly Income',
                             'Loyalty_Score', 'High_Value_Customer', 'Has_Current_Account']
        
        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ])
        
        # Fit and transform
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names
        cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
        self.feature_names = numerical_features + cat_feature_names
        
        # Convert to DataFrame for easier handling
        X_processed = pd.DataFrame(X_processed, columns=self.feature_names)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        self.X_train, self.X_test = X_train_balanced, X_test
        self.y_train, self.y_test = y_train_balanced, y_test
        self.preprocessor = preprocessor
        
        print(f"Training set shape after SMOTE: {X_train_balanced.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train_balanced, X_test, y_train_balanced, y_test
    
    def build_neural_network(self, input_dim, architecture='deep'):
        """Build different NN architectures"""
        model = Sequential()
        
        if architecture == 'deep':
            model.add(Dense(256, activation='relu', input_shape=(input_dim,)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            
            model.add(Dense(64, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))
            
        elif architecture == 'simple':
            model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
            model.add(Dropout(0.2))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))
        
        elif architecture == 'wide':
            model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
            model.add(Dropout(0.4))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.3))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def train_models(self):
        """Train multiple models"""
        print("\n🚀 Training Models...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # 1. Deep Neural Network
        print("Training Deep Neural Network...")
        nn_deep = self.build_neural_network(self.X_train.shape[1], 'deep')
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = nn_deep.fit(
            self.X_train, self.y_train,
            batch_size=32,
            epochs=100,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # Predictions
        y_pred_nn = (nn_deep.predict(self.X_test) > 0.5).astype(int)
        y_prob_nn = nn_deep.predict(self.X_test)
        
        self.models['Deep_NN'] = nn_deep
        self.results['Deep_NN'] = {
            'predictions': y_pred_nn.flatten(),
            'probabilities': y_prob_nn.flatten(),
            'accuracy': np.mean(y_pred_nn.flatten() == self.y_test),
            'auc': roc_auc_score(self.y_test, y_prob_nn.flatten())
        }
        
        # 2. Logistic Regression
        print("Training Logistic Regression...")
        lr = LogisticRegression(random_state=42)
        lr.fit(self.X_train, self.y_train)
        
        y_pred_lr = lr.predict(self.X_test)
        y_prob_lr = lr.predict_proba(self.X_test)[:, 1]
        
        self.models['Logistic_Regression'] = lr
        self.results['Logistic_Regression'] = {
            'predictions': y_pred_lr,
            'probabilities': y_prob_lr,
            'accuracy': lr.score(self.X_test, self.y_test),
            'auc': roc_auc_score(self.y_test, y_prob_lr)
        }
        
        # 3. Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)
        
        y_pred_rf = rf.predict(self.X_test)
        y_prob_rf = rf.predict_proba(self.X_test)[:, 1]
        
        self.models['Random_Forest'] = rf
        self.results['Random_Forest'] = {
            'predictions': y_pred_rf,
            'probabilities': y_prob_rf,
            'accuracy': rf.score(self.X_test, self.y_test),
            'auc': roc_auc_score(self.y_test, y_prob_rf)
        }
        
        # Save models
        joblib.dump(self.models['Logistic_Regression'], 'models/logistic_regression.pkl')
        joblib.dump(self.models['Random_Forest'], 'models/random_forest.pkl')
        joblib.dump(self.preprocessor, 'models/preprocessor.pkl')
        nn_deep.save('models/deep_nn.h5')
        
        # Save feature names
        joblib.dump(self.feature_names, 'models/feature_names.pkl')
        
        print("✅ All models trained and saved!")
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n📊 Model Evaluation")
        print("="*60)
        
        for model_name, results in self.results.items():
            print(f"\n{model_name}:")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"AUC-ROC: {results['auc']:.4f}")
            print(f"Classification Report:")
            print(classification_report(self.y_test, results['predictions']))
    
    def predict_single_customer(self, customer_data):
        """Predict for a single customer"""
        # Use the best model (you can customize this)
        best_model = self.models['Random_Forest']  # Change as needed
        
        # Preprocess the data
        customer_df = pd.DataFrame([customer_data])
        customer_processed = self.preprocessor.transform(customer_df)
        
        # Make prediction
        prediction = best_model.predict(customer_processed)[0]
        probability = best_model.predict_proba(customer_processed)[0][1]
        
        return prediction, probability
    
    def run_complete_pipeline(self, file_path):
        """Run the complete ML pipeline"""
        print("🚀 Starting Complete Churn Prediction Pipeline")
        print("="*60)
        
        # Load and explore data
        self.load_data(file_path)
        self.explore_data()
        
        # Feature engineering and preprocessing
        self.engineer_features()
        self.preprocess_data()
        
        # Train models
        self.train_models()
        
        # Evaluate models
        self.evaluate_models()
        
        print("\n✅ Pipeline Complete!")
        return self.results

# Main execution
if __name__ == "__main__":
    system = ChurnPredictionSystem()
    results = system.run_complete_pipeline('data/churn_prediction_data.csv')