import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, r2_score

from helper import preprocess
from analysis import visualization
import classification
import regression

os.makedirs('saved_models/classification', exist_ok=True)
os.makedirs('saved_models/regression', exist_ok=True)

st.set_page_config(page_title="Online Games Project", layout="wide")
st.title("Online Games Project")

tab1, tab2, tab3 = st.tabs(["Analysis", "Train", "Test"])


with tab1:
    st.header("Data Analysis")
    plot_size = st.slider("Adjust Chart Display Size (%)", min_value=20, max_value=100, value=50)

    if st.button("Run Full Data Visualization"):
        try:
            st.info("Loading training data and generating plots... This might take a moment.")
            df_eda = pd.read_csv('train_data_regression.csv')

            original_show = plt.show
            if plot_size < 100:
                spacer = (100 - plot_size) / 2
                _, center_col, _ = st.columns([spacer, plot_size, spacer])
            else:
                center_col = st.container()

            with center_col:
                plt.show = lambda **kwargs: st.pyplot(plt.gcf(), clear_figure=True, use_container_width=True)
                visualization(df_eda)

            plt.show = original_show
            st.success("Analysis Complete!")

        except FileNotFoundError:
            st.error("Could not find 'train_data_regression.csv' in the directory.")


with tab2:
    st.header("Train a New Model")

    col1, col2 = st.columns(2)
    with col1:
        problem_type = st.selectbox("Select Problem Type", ["Classification (Predict Popularity)", "Regression (Predict Rec. Count)"])

    with col2:
        if "Classification" in problem_type:
            model_options = [
                "Random Forest Classifier", "Extra Trees Classifier",
                "Gradient Boosting Classifier", "AdaBoost Classifier",
                "Logistic Regression", "Naive Bayes",
                "XGBoost Classifier", "LightGBM Classifier",
                "Support Vector Classifier", "Decision Tree Classifier"
            ]
            model_choice = st.selectbox("Select Model", model_options)
        else:
            model_options = ["Random Forest Regressor", "Gradient Boosting Regressor", "Ridge Regression", "Lasso Regression", "Support Vector Regressor"]
            model_choice = st.selectbox("Select Model", model_options)

    # Default Hyperparameters
    def_n_est = 100; def_max_d = 0; def_lr = 0.1; def_sub = 1.0
    def_min_s = 2; def_min_l = 1; def_max_f = "None"; def_boot = True
    def_alpha = 1.0; def_C = 1.0; def_max_iter = 1000; def_kernel = "rbf"

    # Classification
    if model_choice == "Random Forest Classifier":
        def_n_est = 300; def_max_d = 12; def_max_f = "sqrt"
    elif model_choice == "Extra Trees Classifier":
        def_n_est = 300; def_max_d = 12; def_max_f = "sqrt"; def_boot = False
    elif model_choice == "Gradient Boosting Classifier":
        def_n_est = 200; def_max_d = 5; def_sub = 0.8
    elif model_choice == "AdaBoost Classifier":
        def_n_est = 200; def_lr = 1.0
    elif model_choice == "XGBoost Classifier":
        def_n_est = 100; def_max_d = 3; def_lr = 0.05; def_sub = 0.8
    elif model_choice == "LightGBM Classifier":
        def_n_est = 100; def_max_d = 7; def_lr = 0.05; def_sub = 0.8
    elif model_choice == "Logistic Regression":
        def_C = 1.0; def_max_iter = 1000
    elif model_choice == "Support Vector Classifier":
        def_C = 10.0; def_kernel = "rbf"
    elif model_choice == "Decision Tree Classifier":
        def_max_d = 5; def_min_s = 10

    # Regression
    elif model_choice == "Random Forest Regressor":
        def_n_est = 200; def_max_d = 20; def_min_l = 1
    elif model_choice == "Gradient Boosting Regressor":
        def_n_est = 200; def_max_d = 5; def_lr = 0.05
    elif model_choice in ["Ridge Regression", "Lasso Regression"]:
        def_alpha = 0.01
        
    elif model_choice == "Support Vector Regressor":
        def_C = 1.0; def_kernel = "linear"

    st.subheader("Hyperparameters")
    model_kwargs = {}
    h1, h2, h3, h4 = st.columns(4)

    if model_choice in ["Random Forest Classifier", "Extra Trees Classifier", "Random Forest Regressor"]:
        with h1:
            model_kwargs['n_estimators'] = st.number_input("n_estimators", 10, 2000, def_n_est, 10)
            model_kwargs['max_depth'] = st.number_input("max_depth (0=None)", 0, 100, def_max_d)
            if model_kwargs['max_depth'] == 0: model_kwargs['max_depth'] = None
        with h2:
            model_kwargs['min_samples_split'] = st.number_input("min_samples_split", 2, 20, def_min_s)
            model_kwargs['min_samples_leaf'] = st.number_input("min_samples_leaf", 1, 20, def_min_l)
        with h3:
            max_feat = st.selectbox("max_features", ["sqrt", "log2", "None"], index=0 if def_max_f=="sqrt" else 2)
            model_kwargs['max_features'] = None if max_feat == "None" else max_feat
            model_kwargs['bootstrap'] = st.selectbox("bootstrap", [True, False], index=0 if def_boot else 1)
        with h4:
            if "Classifier" in model_choice:
                model_kwargs['criterion'] = st.selectbox("criterion", ["gini", "entropy", "log_loss"])
            else:
                model_kwargs['criterion'] = st.selectbox("criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"])

    elif model_choice in ["Gradient Boosting Classifier", "Gradient Boosting Regressor"]:
        with h1:
            model_kwargs['n_estimators'] = st.number_input("n_estimators", 10, 2000, def_n_est, 10)
            model_kwargs['learning_rate'] = st.number_input("learning_rate", 0.001, 2.0, def_lr, 0.01)
        with h2:
            model_kwargs['max_depth'] = st.number_input("max_depth", 1, 100, def_max_d)
            model_kwargs['subsample'] = st.slider("subsample", 0.1, 1.0, def_sub)
        with h3:
            model_kwargs['min_samples_split'] = st.number_input("min_samples_split", 2, 20, def_min_s)
            model_kwargs['min_samples_leaf'] = st.number_input("min_samples_leaf", 1, 20, def_min_l)
        with h4:
            max_feat = st.selectbox("max_features", ["None", "sqrt", "log2"])
            model_kwargs['max_features'] = None if max_feat == "None" else max_feat
            if "Classifier" in model_choice:
                model_kwargs['loss'] = st.selectbox("loss", ["log_loss", "exponential"])

    elif model_choice in ["XGBoost Classifier", "LightGBM Classifier"]:
        with h1:
            model_kwargs['n_estimators'] = st.number_input("n_estimators", 10, 2000, def_n_est, 10)
            model_kwargs['learning_rate'] = st.number_input("learning_rate", 0.001, 2.0, def_lr, 0.01)
        with h2:
            model_kwargs['max_depth'] = st.number_input("max_depth", 1, 100, def_max_d)
            model_kwargs['subsample'] = st.slider("subsample", 0.1, 1.0, def_sub)
        with h3:
            model_kwargs['colsample_bytree'] = st.slider("colsample_bytree", 0.1, 1.0, 0.8)
            if model_choice == "LightGBM Classifier":
                model_kwargs['num_leaves'] = st.number_input("num_leaves", 10, 200, 31)
        with h4:
            model_kwargs['reg_alpha'] = st.number_input("reg_alpha (L1)", 0.0, 10.0, 0.5)
            model_kwargs['reg_lambda'] = st.number_input("reg_lambda (L2)", 0.0, 10.0, 1.5)

    elif model_choice == "AdaBoost Classifier":
        with h1:
            model_kwargs['n_estimators'] = st.number_input("n_estimators", 10, 2000, def_n_est, 10)
        with h2:
            model_kwargs['learning_rate'] = st.number_input("learning_rate", 0.001, 2.0, def_lr, 0.01)
        with h3:
            model_kwargs['algorithm'] = st.selectbox("algorithm", ["SAMME"])

    elif model_choice == "Logistic Regression":
        with h1:
            model_kwargs['C'] = st.number_input("C (Inverse Regularization)", 0.01, 100.0, def_C)
            model_kwargs['max_iter'] = st.number_input("max_iter", 100, 5000, def_max_iter)
        with h2:
            model_kwargs['solver'] = st.selectbox("solver", ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"])
        with h3:
            model_kwargs['penalty'] = st.selectbox("penalty", ["l2", "l1", "elasticnet", "None"])
            if model_kwargs['penalty'] == "None": model_kwargs['penalty'] = None
        with h4:
            if model_kwargs['penalty'] == "elasticnet":
                model_kwargs['l1_ratio'] = st.slider("l1_ratio (For ElasticNet)", 0.0, 1.0, 0.5)

    elif model_choice == "Naive Bayes":
        with h1:
            model_kwargs['var_smoothing'] = st.number_input("var_smoothing", 1e-12, 1e-5, 1e-9, format="%.1e")

    elif model_choice in ["Ridge Regression", "Lasso Regression"]:
        with h1:
            model_kwargs['alpha'] = st.number_input("alpha", 0.01, 100.0, def_alpha)
            model_kwargs['max_iter'] = st.number_input("max_iter (0=None)", 0, 10000, 0)
            if model_kwargs['max_iter'] == 0: model_kwargs['max_iter'] = None
        if model_choice == "Ridge Regression":
            with h2:
                model_kwargs['solver'] = st.selectbox("solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])
        else:
            with h2:
                model_kwargs['selection'] = st.selectbox("selection", ["cyclic", "random"])

    elif model_choice == "Support Vector Regressor":
        with h1:
            model_kwargs['C'] = st.number_input("C", 0.01, 100.0, def_C)
            model_kwargs['epsilon'] = st.number_input("epsilon", 0.01, 10.0, 0.1)
        with h2:
            kernel_options = ["rbf", "linear", "poly", "sigmoid"]
            model_kwargs['kernel'] = st.selectbox("kernel", kernel_options, index=kernel_options.index(def_kernel))

            if model_kwargs['kernel'] == 'poly':
                model_kwargs['degree'] = st.number_input("degree (Poly)", 2, 10, 3)
        with h3:
            model_kwargs['gamma'] = st.selectbox("gamma", ["scale", "auto"])

    elif model_choice == "Decision Tree Classifier":
        with h1:
            model_kwargs['max_depth'] = st.number_input("max_depth (0=None)", 0, 100, def_max_d)
            if model_kwargs['max_depth'] == 0: model_kwargs['max_depth'] = None
        with h2:
            model_kwargs['min_samples_split'] = st.number_input("min_samples_split", 2, 100, def_min_s)
            model_kwargs['min_samples_leaf'] = st.number_input("min_samples_leaf", 1, 100, def_min_l)
        with h3:
            max_feat = st.selectbox("max_features", ["None", "sqrt", "log2"])
            model_kwargs['max_features'] = None if max_feat == "None" else max_feat
        with h4:
            model_kwargs['criterion'] = st.selectbox("criterion", ["gini", "entropy", "log_loss"])


    elif model_choice == "Support Vector Classifier":
        with h1:
            model_kwargs['C'] = st.number_input("C", 0.01, 1000.0, def_C)
        with h2:
            kernel_options = ["rbf", "linear", "poly", "sigmoid"]
            model_kwargs['kernel'] = st.selectbox("kernel", kernel_options, index=kernel_options.index(def_kernel))
            if model_kwargs['kernel'] == 'poly':
                model_kwargs['degree'] = st.number_input("degree (Poly)", 2, 10, 3)
        with h3:
            model_kwargs['gamma'] = st.selectbox("gamma", ["scale", "auto"])


    st.markdown("---")

    if st.button("Start", use_container_width=True):
        with st.spinner(f"Training {model_choice}..."):
            try:
                task_folder = "classification" if "Classification" in problem_type else "regression"

                if "Classification" in problem_type:
                    X_train, y_train, X_test, y_test, weights = classification.prep_classification_data(use_smote=True)
                    model = classification.get_classification_model(model_choice, **model_kwargs)

                    model, train_time, test_time, acc_test, f1_test, y_pred_test = classification.train_and_eval_classification(
                        model, X_train, y_train, X_test, y_test, weights
                    )

                    st.subheader("Classification Performance")
                    metric_col1, metric_col2 = st.columns(2)
                    metric_col1.metric("Test Accuracy", f"{acc_test:.4f}")
                    metric_col2.metric("Test F1 Score", f"{f1_test:.4f}")

                    _, chart_col, _ = st.columns([1, 4, 1])
                    with chart_col:

                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                        axes[0].bar(['Accuracy'], [acc_test], color='steelblue')
                        axes[0].set_title(f"Test Accuracy ({acc_test:.4f})")
                        axes[0].set_ylim(0, 1.0)

                        axes[1].bar(['Training'], [train_time], color='forestgreen')
                        axes[1].set_title(f"Train Time ({train_time:.2f}s)")

                        axes[2].bar(['Testing'], [test_time], color='purple')
                        axes[2].set_title(f"Test Time ({test_time:.4f}s)")

                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=False)

                else:
                    X_train, y_train_log, X_test, y_test_log = regression.prep_regression_data()
                    model = regression.get_regression_model(model_choice, **model_kwargs)

                    model, train_time, test_time, mse_test, r2_test, mae_test, y_test_act, y_pred_act = regression.train_and_eval_regression(
                        model, X_train, y_train_log, X_test, y_test_log
                    )

                    st.subheader("Regression Performance")
                    r_col1, r_col2, r_col3 = st.columns(3)
                    r_col1.metric("Test MAE", f"{mae_test:.2f}")
                    r_col2.metric("Test RMSE", f"{np.sqrt(mse_test):.2f}")
                    r_col3.metric("Test R2", f"{r2_test:.4f}")

                    _, chart_col, _ = st.columns([1, 4, 1])
                    with chart_col:
                        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                        axes[0].bar(['Training'], [train_time], color='forestgreen')
                        axes[0].set_title(f"Train Time ({train_time:.2f}s)")
                        axes[1].bar(['Testing'], [test_time], color='purple')
                        axes[1].set_title(f"Test Time ({test_time:.4f}s)")
                        st.pyplot(fig, use_container_width=False)

                        st.write("### Actual vs Predicted (Test Set)")

                        plot_limit = np.percentile(y_test_act, 80)

                        fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
                        ax_scatter.scatter(y_test_act, y_pred_act, alpha=0.5, color='purple')

                        ax_scatter.plot([0, plot_limit], [0, plot_limit], 'k--', lw=2)

                        ax_scatter.set_xlim(0, plot_limit)
                        ax_scatter.set_ylim(0, plot_limit)
                        ax_scatter.set_xlabel("Actual Recommendations")
                        ax_scatter.set_ylabel("Predicted Recommendations")

                        st.pyplot(fig_scatter, use_container_width=False)

                safe_name = model_choice.replace(" ", "_").lower()
                save_path = f"saved_models/{task_folder}/{safe_name}.pkl"
                joblib.dump(model, save_path)
                st.success(f"save path: `{save_path}`")

            except ValueError as ve:
                st.error(f"Incompatible Hyperparameters: {str(ve)}")
            except FileNotFoundError as e:
                st.error(f"Missing preprocessed files! {str(e)}")


with tab3:
    st.header("Test")
    test_problem_type = st.selectbox("Which phase ?", ["Classification", "Regression"])

    folder_to_scan = 'saved_models/classification' if test_problem_type == "Classification" else 'saved_models/regression'
    saved_files = [f for f in os.listdir(folder_to_scan) if f.endswith('.pkl')]

    if not saved_files:
        st.warning(f"No saved models found in the `{folder_to_scan}` folder. Please train a model first.")
    else:
        selected_model_file = st.selectbox("Select Pre-trained Model", saved_files)
        uploaded_file = st.file_uploader("Upload test.csv", type=['csv'])

        if uploaded_file is not None and st.button("Predict & Evaluate"):
            with st.spinner("Processing data and generating predictions..."):
                try:
                    raw_df = pd.read_csv(uploaded_file)
                    task_suffix = test_problem_type.lower()
                    tools = joblib.load(f'preprocessing_tools_{task_suffix}.pkl')

                    has_target = False
                    if test_problem_type == "Classification" and 'GamePopularity' in raw_df.columns:
                        has_target = True
                        y_true = raw_df['GamePopularity'].map({'Low': 0, 'Medium': 1, 'High': 2})
                        X_raw = raw_df.drop(columns=['GamePopularity'])
                    elif test_problem_type == "Regression" and 'RecommendationCount' in raw_df.columns:
                        has_target = True
                        y_true = raw_df['RecommendationCount']
                        X_raw = raw_df.drop(columns=['RecommendationCount'])
                    else:
                        X_raw = raw_df

                    X_test_prep = preprocess(
                        X_raw,
                        is_train=False,
                        scaler=tools['scaler'],
                        text_pcas=tools['text_pcas'],
                        train_cols=tools['columns']
                    )

                    leaky_cols = ['SteamSpyOwners', 'SteamSpyOwnersVariance', 'PriceInitial', 'ReleaseYear', 'date_extraction_failure']
                    X_test_prep = X_test_prep.drop(columns=leaky_cols, errors='ignore')

                    model = joblib.load(f"{folder_to_scan}/{selected_model_file}")
                    predictions = model.predict(X_test_prep)

                    if not has_target:
                        st.dataframe(pd.DataFrame({"Prediction": predictions}))
                    else:
                        _, eval_col, _ = st.columns([1, 2, 1])
                        with eval_col:
                            if test_problem_type == "Classification":
                                st.write(f"**Test Accuracy:** {accuracy_score(y_true, predictions):.4f}")
                                fig, ax = plt.subplots(figsize=(6, 4))
                                sns.heatmap(confusion_matrix(y_true, predictions), annot=True, fmt='d', cmap='Greens', ax=ax)
                                st.pyplot(fig, use_container_width=False)
                            else:
                                preds_act = np.expm1(predictions)
                                st.write(f"**MAE:** {mean_absolute_error(y_true, preds_act):.4f}")
                                st.write(f"**R2 Score:** {r2_score(y_true, preds_act):.4f}")
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.scatter(y_true, preds_act, alpha=0.5, color='purple')
                                ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
                                st.pyplot(fig, use_container_width=False)

                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")