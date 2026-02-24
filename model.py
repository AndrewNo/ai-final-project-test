import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def generate_synthetic_data(n_samples=100, n_features=1, noise=10, random_state=42):
    """
    Generates a synthetic dataset for regression.
    """
    X, y = np.random.rand(n_samples, n_features) * 100, np.random.rand(n_samples) * 100
    y = 2 * X.squeeze() + 1 + np.random.randn(n_samples) * noise
    return X, y

def train_regression_model(X_train, y_train):
    """
    Trains a linear regression model.

    Parameters:
    - X_train: Training features.
    - y_train: Training target values.

    Returns:
    - A trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def save_regression_model(model, filename="linear_regression_model.joblib"):
    """
    Serializes and saves a trained regression model.

    Parameters:
    - model: The trained regression model to save.
    - filename: The file to save the model to.
    """
    joblib.dump(model, filename)

def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluates the regression model using mean squared error.

    Parameters:
    - model: The trained regression model.
    - X_test: Test features.
    - y_test: Test target values.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

def save_initial_datasets(X, y, X_filename="X.joblib", y_filename="y.joblib"):
    """
    Serializes and saves the initial datasets.

    Parameters:
    - X: The feature matrix.
    - y: The target values.
    - X_filename: The file to save the feature matrix to.
    - y_filename: The file to save the target values to.
    """
    joblib.dump(X, X_filename)
    joblib.dump(y, y_filename)

if __name__ == "__main__":
    # Generate synthetic data
    X, y = generate_synthetic_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the regression model
    model = train_regression_model(X_train, y_train)

    # Save the trained model
    save_regression_model(model)

    # Evaluate the model
    evaluate_regression_model(model, X_test, y_test)

    # Save the initial datasets
    save_initial_datasets(X, y)

    print("Model training, evaluation, and serialization complete.")
