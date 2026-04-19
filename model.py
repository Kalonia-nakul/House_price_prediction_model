import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load data
df = pd.read_csv("Housing.csv")

# Separate features and target
X = df[['area', 'bedrooms', 'bathrooms', 'stories',
        'mainroad', 'guestroom', 'basement',
        'hotwaterheating', 'airconditioning',
        'parking', 'prefarea', 'furnishingstatus']]
y = df['price']

# Define categorical and numeric columns
categorical = ['mainroad', 'guestroom', 'basement',
               'hotwaterheating', 'airconditioning',
               'prefarea', 'furnishingstatus']
numeric = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Preprocess: One-hot encode categorical, pass numeric unchanged
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical),
        ('num', 'passthrough', numeric)
    ]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
model = LinearRegression()
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Train
model.fit(X_train_transformed, y_train)

# Predict
y_pred = model.predict(X_test_transformed)

# Evaluate
rmse = mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)
print("R² Score:", r2_score(y_test, y_pred))
