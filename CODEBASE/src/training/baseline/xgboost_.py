import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load cleaned train data
try:
    train_cleaned = pd.read_csv('../../../dataset/train_cleaned.csv')
except FileNotFoundError:
    print("train_cleaned.csv not found. Make sure you have run the dataset_cleaning.py script.")
    exit()

# Parse bullet_points from JSON string to list
train_cleaned['bullet_points'] = train_cleaned['bullet_points'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

# Create text feature by concatenating item_name and bullet_points
train_cleaned['text'] = train_cleaned['item_name'].fillna('') + ' ' + train_cleaned['bullet_points'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

# Drop rows with missing price or essential features
train_cleaned = train_cleaned.dropna(subset=['price', 'text', 'value', 'unit'])
train_cleaned = train_cleaned[train_cleaned['text'].str.strip() != '']

# Features and target
X = train_cleaned[['text', 'value', 'unit']]
y = train_cleaned['price']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numeric, categorical, and text features
numeric_features = ['value']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['unit']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

text_feature = 'text'
text_transformer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', text_transformer, text_feature)
    ],
    remainder='drop'
)

# Create the model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        # tree_method='gpu_hist'  # Use GPU for training
    ))
])

# Train the model
print("Training XGBoost model on combined features (GPU)...")
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Plot predictions vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price (XGBoost - Combined Features)')
plt.show()

# Save the model
joblib.dump(model, 'xgboost_combined_gpu_model.pkl')
print("Model saved as xgboost_combined_gpu_model.pkl")