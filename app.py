import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import model as model_module

def retrain_model():
    close_prices = model_module.load_commodity_data()
    X, y = model_module.build_features_targets(close_prices)
    model = model_module.train_regression_model(X, y)
    model_module.save_regression_model(model)
    model_module.save_initial_datasets(X, y)
    return model

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
    if hasattr(model, "n_features_in_") and input_data.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Model expects {model.n_features_in_} features, got {input_data.shape[1]}."
        )
    prediction = model.predict(input_data)
    return prediction

def visualize_difference(input_features, predicted_values, X_filename="X.joblib", y_filename="y.joblib"):
    """
    Visualizes the difference between actual and predicted values.

    Parameters:
    - input_features: The feature values selected by the user.
    - predicted_values: The values predicted by the model.
    - X_filename: The file where the feature matrix is saved.
    - y_filename: The file where the target values are saved.
    """
    X = np.asarray(joblib.load(X_filename))
    y = np.asarray(joblib.load(y_filename))

    input_features = np.asarray(input_features, dtype=float).reshape(1, -1)
    predicted_values = np.asarray(predicted_values, dtype=float).reshape(-1)

    # Find the index of the closest feature row in the original dataset
    diffs = X - input_features
    closest_index = int(np.argmin(np.linalg.norm(diffs, axis=1)))
    actual_values = np.asarray(y[closest_index], dtype=float).reshape(-1)

    labels = ["Gold", "Silver", "Oil"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, actual_values, width, label="Actual")
    ax.bar(x + width / 2, predicted_values, width, label="Predicted")

    for i, (act, pred) in enumerate(zip(actual_values, predicted_values)):
        ax.text(i - width / 2, act, f"{act:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, pred, f"{pred:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Price (USD)")
    ax.set_title("Actual vs Predicted (Closest Historical Match)")
    ax.legend()
    st.pyplot(fig)

def create_streamlit_app():
    """
    Creates and runs the Streamlit web application.
    """
    st.title("Commodity Price Predictor")

    # Sidebar for user input
    st.sidebar.header("Input Features (Previous Day Close)")
    gold = st.sidebar.number_input("Gold (USD)", min_value=0.0, value=2000.0, step=1.0)
    silver = st.sidebar.number_input("Silver (USD)", min_value=0.0, value=25.0, step=0.1)
    oil = st.sidebar.number_input("Oil (USD)", min_value=0.0, value=80.0, step=0.5)

    if st.sidebar.button("Predict value"):
        # Predict the values
        input_features = np.array([[gold, silver, oil]], dtype=float)
        try:
            predicted_value = load_and_predict("linear_regression_model.joblib", input_features)
        except ValueError:
            with st.spinner("Model/data mismatch. Retraining from latest data..."):
                try:
                    retrain_model()
                    predicted_value = load_and_predict("linear_regression_model.joblib", input_features)
                except Exception as exc:
                    st.error(
                        "Model/data mismatch and retraining failed. "
                        "Please rerun: `pip install -r requirements.txt` then `python model.py`."
                    )
                    st.stop()
        predicted_value = np.asarray(predicted_value).reshape(-1)
        st.write(
            f"Predicted next-day close — Gold: {predicted_value[0]:.2f}, "
            f"Silver: {predicted_value[1]:.2f}, Oil: {predicted_value[2]:.2f}"
        )

        # Visualize the difference
        visualize_difference(input_features, predicted_value)

if __name__ == "__main__":
    create_streamlit_app()
