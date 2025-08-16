import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

st.title("Welcome to Machine Learning")

# Dropdown with no pre-selection
models = ["", 'Logistic', 'Linear', 'SVM', 'DNN', 'Decision Tree']
options = st.selectbox('Which model you want to use', models)

# Save selected model in session_state
if st.button('Select') and options:
    st.session_state.selected_model = options

# Check if a model was selected previously
if 'selected_model' in st.session_state:
    selected = st.session_state.selected_model

    # Load necessary objects for each model
    if selected == 'Logistic':
        with open('logistic_model.pkl', 'rb') as file:
            logistic_data = pkl.load(file)  # should contain {'model': model, 'scaler': scaler, 'X_columns': X.columns, 'X_test': X_test, 'y_test': y_test}
        model = logistic_data['model']
        scaler = logistic_data['scaler']
        X_columns = logistic_data['X_columns']
        X_test = logistic_data['X_test']
        y_test = logistic_data['y_test']

        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.subheader("Logistic Regression Prediction")
        st.write(f"Accuracy: **{accuracy:.2f}**")

    elif selected == 'Linear':
        with open('linear_model.pkl', 'rb') as file:
            linear_data = pkl.load(file)
        model = linear_data['model']
        scaler = linear_data['scaler']
        X_columns = linear_data['X_columns']
        X_test = linear_data['X_test']
        y_test = linear_data['y_test']

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Linear Regression Prediction")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

    elif selected == 'SVM':
        with open('svm_model.pkl', 'rb') as file:
            svm_data = pkl.load(file)
        model = svm_data['model']
        scaler = svm_data['scaler']
        X_columns = svm_data['X_columns']
        X_test = svm_data['X_test']
        y_test = svm_data['y_test']

        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.subheader("SVM Accuracy Prediction")
        st.write(f"Accuracy: **{accuracy:.2f}**")
    elif selected == 'DNN':
        dnn_model = load_model("dnn_model.h5")
        with open("dnn_model_meta.pkl", "rb") as file:
            dnn_meta = pkl.load(file)
        model = dnn_model
        scaler = dnn_meta['scaler']
        X_columns = dnn_meta['X_columns']
        X_test = dnn_meta['X_test']
        y_test = dnn_meta['y_test']

        # Accuracy on test data
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"DNN Accuracy: **{accuracy:.2f}**")
        
    elif selected == 'Decision Tree':
        with open('decision_tree_model.pkl', 'rb') as file:
            dt_data = pkl.load(file)
        model = dt_data['model']
        scaler = dt_data['scaler']
        X_columns = dt_data['X_columns']
        X_test = dt_data['X_test']
        y_test = dt_data['y_test']

        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.subheader("Decision Tree Prediction")
        st.write(f"Accuracy: **{accuracy:.2f}**")


            # Common user input section
    specific_columns = ["Age", "Avg_Daily_Usage_Hours", "Anxiety", "Depression",
                        "Device Model", "User Behavior Class", "Attack Type",
                        "Attack Source", "Price", "Discount"]

    user_data = {}
    for col in specific_columns:
        user_data[col] = st.text_input(f"Enter {col}:")

    if st.button("Predict"):
        try:
            if any(v.strip() == "" for v in user_data.values()):
                st.error("Please fill all fields.")
            else:
                user_values = [float(user_data[col]) for col in specific_columns]
                mean_values = pd.DataFrame(X_test, columns=X_columns).mean()

                full_input = [user_values[specific_columns.index(col)] if col in specific_columns else mean_values[col] for col in X_columns]

                user_df = pd.DataFrame([full_input], columns=X_columns)
                user_scaled = scaler.transform(user_df)
                prediction = model.predict(user_scaled)[0]

                if selected in ['Logistic', 'SVM', 'DNN', 'Decision Tree']:
                    if prediction == 0:
                        st.success("No Affects On Academic Performance")
                    else:
                        st.success("Yes, Affects On Academic Performance")
                elif selected == 'Linear':
                    st.success(f"Predicted Value (Data usage): **{prediction:.2f}**")
        except ValueError:
            st.error("Please enter valid numeric values.")
