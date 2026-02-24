import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

def load_and_predict(model_filename, input_data):
    """
    Loads a saved regression model and uses it to predict a target value.

    Parameters:
    - model_filename: The file where the model is saved.
    - input_data: The user-provided input data for prediction.

    Returns:
    - The model's prediction.
    """
    model = joblib.load(model_filename)
    prediction = model.predict(input_data)
    return prediction

def visualize_difference(input_feature, predicted_value, X_filename="X.joblib", y_filename="y.joblib"):
    """
    Visualizes the difference between the actual and predicted values.

    Parameters:
    - input_feature: The feature value selected by the user.
    - predicted_value: The value predicted by the model.
    - X_filename: The file where the feature matrix is saved.
    - y_filename: The file where the target values are saved.
    """
    X = np.asarray(joblib.load(X_filename)).reshape(-1)
    y = np.asarray(joblib.load(y_filename)).reshape(-1)

    # Find the index of the closest feature in the original dataset
    input_feature = float(input_feature)
    closest_index = np.argmin(np.abs(X - input_feature))
    actual_value = float(y[closest_index])
    predicted_value = float(np.asarray(predicted_value).reshape(-1)[0])

    # Create the plot
    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Dataset")
    ax.scatter([input_feature], [actual_value], color='red', label="Actual Value")
    ax.scatter([input_feature], [predicted_value], color='green', label="Predicted Value")

    # Draw a line between the actual and predicted values
    x_line = np.array([input_feature, input_feature], dtype=float)
    y_line = np.array([actual_value, predicted_value], dtype=float)
    ax.plot(x_line, y_line, 'k--')

    # Annotate the difference
    difference = predicted_value - actual_value
    ax.text(input_feature, (actual_value + predicted_value) / 2, f"Difference: {difference:.2f}", ha='center')

    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    ax.legend()
    st.pyplot(fig)

def create_streamlit_app():
    """
    Creates and runs the Streamlit web application.
    """
    st.title("Linear Regression Predictor")

    # Sidebar for user input
    st.sidebar.header("Input Feature")
    input_feature = st.sidebar.slider("Select a value", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

    if st.sidebar.button("Predict value"):
        # Predict the value
        predicted_value = load_and_predict("linear_regression_model.joblib", np.array([[input_feature]]))
        st.write(f"Predicted value: {predicted_value[0]:.2f}")

        # Visualize the difference
        visualize_difference(input_feature, predicted_value)

if __name__ == "__main__":
    create_streamlit_app()
