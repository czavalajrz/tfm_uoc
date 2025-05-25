# Python script for WiFi Indoor Localization (Batch Processing)
# Modified version including Random Forest model

# Import necessary libraries
import pandas as pd
import numpy as np
import time
import os
import shutil # For cleaning up output directories if needed
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- Global Configurations --- #
K_NEIGHBORS = 5
RSSI_NO_SIGNAL_REPLACEMENT = -105 # Value to replace 100 (no signal)
# Random Forest parameters
N_ESTIMATORS = 100
MAX_DEPTH = 20
MIN_SAMPLES_SPLIT = 5
RANDOM_STATE = 42

# Floor mapping - assuming this might be common, but can be adapted if datasets differ
FLOOR_MAPPING = {
    0.0: 0,
    3.7: 1,
    7.4: 2,
    11.1: 3,
    14.8: 4
}

# --- Helper Functions --- #

def create_output_directory(base_path, dir_name):
    """Creates an output directory, removing it if it already exists."""
    output_dir = os.path.join(base_path, dir_name)
    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists. Removing it.")
        shutil.rmtree(output_dir) # Use with caution, removes directory and its contents
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")
    return output_dir

def load_single_dataset(dataset_path, dataset_name):
    """Loads RSSI and coordinate data for a single dataset."""
    print(f"\n--- Loading data for dataset: {dataset_name} ---")
    
    coord_columns_actual = ["LONGITUDE", "LATITUDE", "HEIGHT", "FLOOR", "BUILDINGID"]
    try:
        trn_rss_path = os.path.join(dataset_path, f"{dataset_name}_trnrss.csv")
        trn_rss_df = pd.read_csv(trn_rss_path, header=None)
        # Dynamically determine WAP columns for training RSS data
        num_wap_cols_train = trn_rss_df.shape[1]
        wap_columns_train = [f'WAP{str(i).zfill(3)}' for i in range(1, num_wap_cols_train + 1)]
        trn_rss_df.columns = wap_columns_train

        trn_crd_path = os.path.join(dataset_path, f"{dataset_name}_trncrd.csv")
        trn_crd_df = pd.read_csv(trn_crd_path, header=None)
        trn_crd_df.columns = coord_columns_actual

        tst_rss_path = os.path.join(dataset_path, f"{dataset_name}_tstrss.csv")
        tst_rss_df = pd.read_csv(tst_rss_path, header=None)
        # Dynamically determine WAP columns for testing RSS data
        num_wap_cols_test = tst_rss_df.shape[1]
        wap_columns_test = [f'WAP{str(i).zfill(3)}' for i in range(1, num_wap_cols_test + 1)]
        # It's crucial that train and test WAP columns match in number and naming convention for consistency
        # However, the actual number of WAPs detected might differ if files are truly different.
        # For now, we assume the *intent* is that they should match the training data's WAP count if they are part of the same dataset definition.
        # If they can genuinely differ, the downstream feature selection needs to be robust to this (e.g. use intersection of columns or pad missing ones).
        # For this implementation, we will use the training WAP columns for the test set, assuming they should conform.
        # If num_wap_cols_test is different from num_wap_cols_train, this might indicate an issue or require padding/selection.
        # For now, let's assume they should be the same and use wap_columns_train for tst_rss_df as well.
        # This ensures consistency in feature sets. If a WAP is in train but not test, it will be NaN and handled by model or preprocessing.
        # If a WAP is in test but not train, it would be an extra column not used by model trained on train_wap_columns.
        # The most robust way is to ensure they have the same columns. If test has more, select only train_wap_columns. If test has less, it's an issue.
        if num_wap_cols_test != num_wap_cols_train:
            print(f"Warning: Dataset {dataset_name} has {num_wap_cols_train} WAPs in training and {num_wap_cols_test} WAPs in testing. Using training WAP definition.")
            # This part needs careful consideration. If test has fewer, we might need to pad or error.
            # If test has more, we should select only the ones present in training.
            # For now, we'll assign based on its own shape and let feature selection handle it, but this is a potential issue.
            tst_rss_df.columns = wap_columns_test # Assign based on its own shape for now
        else:
            tst_rss_df.columns = wap_columns_train # If same number, use train's definition for consistency

        tst_crd_path = os.path.join(dataset_path, f"{dataset_name}_tstcrd.csv")
        tst_crd_df = pd.read_csv(tst_crd_path, header=None)
        tst_crd_df.columns = coord_columns_actual

        train_df = pd.concat([trn_rss_df, trn_crd_df], axis=1)
        test_df = pd.concat([tst_rss_df, tst_crd_df], axis=1)
        print(f"Data loaded and merged successfully for {dataset_name}.")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error loading data for {dataset_name}: {e}")
        return None, None
    except ValueError as e: # Handles column length mismatch if WAP count is different
        print(f"ValueError during data loading for {dataset_name} (check WAP column count or file format): {e}")
        return None, None

def preprocess_dataset_data(train_df, test_df, dataset_name):
    """Preprocesses data: handles target encoding and RSSI values."""
    print(f"\n--- Preprocessing data for dataset: {dataset_name} ---")
    
    # Correct Target Variable Encoding for FLOOR and BUILDINGID
    # Check if FLOOR column needs mapping (i.e., contains float values)
    if train_df['FLOOR'].dtype == float or test_df['FLOOR'].dtype == float:
        print(f"Mapping FLOOR values for {dataset_name}...")
        train_df['FLOOR'] = train_df['FLOOR'].map(FLOOR_MAPPING).fillna(train_df['FLOOR']).astype(int)
        test_df['FLOOR'] = test_df['FLOOR'].map(FLOOR_MAPPING).fillna(test_df['FLOOR']).astype(int)
    else:
        train_df['FLOOR'] = train_df['FLOOR'].astype(int)
        test_df['FLOOR'] = test_df['FLOOR'].astype(int)
    
    train_df['BUILDINGID'] = train_df['BUILDINGID'].astype(int)
    test_df['BUILDINGID'] = test_df['BUILDINGID'].astype(int)
    print(f"Target encoding for FLOOR and BUILDINGID complete for {dataset_name}.")

    # Feature columns (WAPs)
    feature_columns = [col for col in train_df.columns if 'WAP' in col]

    X_train = train_df[feature_columns].copy()
    y_train_building = train_df['BUILDINGID'].copy()
    y_train_floor = train_df['FLOOR'].copy()
    y_train_coords = train_df[['LONGITUDE', 'LATITUDE']].copy()

    X_test = test_df[feature_columns].copy()
    y_test_building = test_df['BUILDINGID'].copy()
    y_test_floor = test_df['FLOOR'].copy()
    y_test_coords = test_df[['LONGITUDE', 'LATITUDE']].copy()

    # Replace 100 (no signal) with RSSI_NO_SIGNAL_REPLACEMENT
    X_train.replace(100, RSSI_NO_SIGNAL_REPLACEMENT, inplace=True)
    X_test.replace(100, RSSI_NO_SIGNAL_REPLACEMENT, inplace=True)
    print(f"RSSI 'no signal' values replaced for {dataset_name}.")
    
    return X_train, y_train_building, y_train_floor, y_train_coords, X_test, y_test_building, y_test_floor, y_test_coords, test_df # Return full test_df for error analysis

def train_and_predict_model(model_name_suffix, X_train, y_train_building, y_train_floor, y_train_coords, X_test, y_test_building, y_test_floor, y_test_coords, k_neighbors, use_pca=False, metric='minkowski', weights='uniform'):
    """Trains building, floor, and coordinate models and returns predictions and timings."""
    print(f"\n--- Implementing {model_name_suffix} (PCA: {use_pca}, Metric: {metric}, Weights: {weights}) ---")
    
    current_X_train, current_X_test = X_train, X_test
    pca_components = None

    if use_pca:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        pca = PCA(n_components=0.95) # Retain 95% of variance
        current_X_train = pca.fit_transform(X_train_scaled)
        current_X_test = pca.transform(X_test_scaled)
        pca_components = pca.n_components_
        print(f"PCA applied. Number of components: {pca_components}")

    # Building Model
    building_model = KNeighborsClassifier(n_neighbors=k_neighbors, weights=weights, metric=metric)
    start_time = time.time(); building_model.fit(current_X_train, y_train_building); t_time_b = time.time() - start_time
    start_time = time.time(); y_pred_building = building_model.predict(current_X_test); p_time_b = time.time() - start_time

    # Floor Model
    floor_model = KNeighborsClassifier(n_neighbors=k_neighbors, weights=weights, metric=metric)
    start_time = time.time(); floor_model.fit(current_X_train, y_train_floor); t_time_f = time.time() - start_time
    start_time = time.time(); y_pred_floor = floor_model.predict(current_X_test); p_time_f = time.time() - start_time

    # Coordinates Model
    coords_model = KNeighborsRegressor(n_neighbors=k_neighbors, weights=weights, metric=metric)
    start_time = time.time(); coords_model.fit(current_X_train, y_train_coords); t_time_c = time.time() - start_time
    start_time = time.time(); y_pred_coords = coords_model.predict(current_X_test); p_time_c = time.time() - start_time

    predictions_df = pd.DataFrame({
        'BUILDINGID_True': y_test_building,
        f'BUILDINGID_Pred_{model_name_suffix}': y_pred_building,
        'FLOOR_True': y_test_floor,
        f'FLOOR_Pred_{model_name_suffix}': y_pred_floor,
        'LONGITUDE_True': y_test_coords['LONGITUDE'],
        'LATITUDE_True': y_test_coords['LATITUDE'],
        f'LONGITUDE_Pred_{model_name_suffix}': y_pred_coords[:, 0],
        f'LATITUDE_Pred_{model_name_suffix}': y_pred_coords[:, 1]
    })

    timing_info = {
        'train_time_building': t_time_b, 'predict_time_building': p_time_b,
        'train_time_floor': t_time_f, 'predict_time_floor': p_time_f,
        'train_time_coords': t_time_c, 'predict_time_coords': p_time_c,
        'total_train_time': t_time_b + t_time_f + t_time_c,
        'total_predict_time': p_time_b + p_time_f + p_time_c,
        'pca_components': pca_components
    }
    print(f"{model_name_suffix} implementation complete.")
    return predictions_df, timing_info

def train_and_predict_random_forest(model_name_suffix, X_train, y_train_building, y_train_floor, y_train_coords, X_test, y_test_building, y_test_floor, y_test_coords):
    """Trains Random Forest models for building, floor, and coordinate prediction and returns predictions and timings."""
    print(f"\n--- Implementing {model_name_suffix} (n_estimators: {N_ESTIMATORS}, max_depth: {MAX_DEPTH}) ---")
    
    # Building Model
    building_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, 
        max_depth=MAX_DEPTH, 
        min_samples_split=MIN_SAMPLES_SPLIT, 
        random_state=RANDOM_STATE
    )
    start_time = time.time(); building_model.fit(X_train, y_train_building); t_time_b = time.time() - start_time
    start_time = time.time(); y_pred_building = building_model.predict(X_test); p_time_b = time.time() - start_time

    # Floor Model
    floor_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, 
        max_depth=MAX_DEPTH, 
        min_samples_split=MIN_SAMPLES_SPLIT, 
        random_state=RANDOM_STATE
    )
    start_time = time.time(); floor_model.fit(X_train, y_train_floor); t_time_f = time.time() - start_time
    start_time = time.time(); y_pred_floor = floor_model.predict(X_test); p_time_f = time.time() - start_time

    # Coordinates Model
    coords_model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, 
        max_depth=MAX_DEPTH, 
        min_samples_split=MIN_SAMPLES_SPLIT, 
        random_state=RANDOM_STATE
    )
    start_time = time.time(); coords_model.fit(X_train, y_train_coords); t_time_c = time.time() - start_time
    start_time = time.time(); y_pred_coords = coords_model.predict(X_test); p_time_c = time.time() - start_time

    # Get feature importances for analysis
    building_importances = building_model.feature_importances_
    floor_importances = floor_model.feature_importances_
    
    # Store top 10 most important features for each task
    feature_names = X_train.columns
    building_top_features = pd.DataFrame({
        'feature': feature_names,
        'importance': building_importances
    }).sort_values('importance', ascending=False).head(10)
    
    floor_top_features = pd.DataFrame({
        'feature': feature_names,
        'importance': floor_importances
    }).sort_values('importance', ascending=False).head(10)

    predictions_df = pd.DataFrame({
        'BUILDINGID_True': y_test_building,
        f'BUILDINGID_Pred_{model_name_suffix}': y_pred_building,
        'FLOOR_True': y_test_floor,
        f'FLOOR_Pred_{model_name_suffix}': y_pred_floor,
        'LONGITUDE_True': y_test_coords['LONGITUDE'],
        'LATITUDE_True': y_test_coords['LATITUDE'],
        f'LONGITUDE_Pred_{model_name_suffix}': y_pred_coords[:, 0],
        f'LATITUDE_Pred_{model_name_suffix}': y_pred_coords[:, 1]
    })

    timing_info = {
        'train_time_building': t_time_b, 'predict_time_building': p_time_b,
        'train_time_floor': t_time_f, 'predict_time_floor': p_time_f,
        'train_time_coords': t_time_c, 'predict_time_coords': p_time_c,
        'total_train_time': t_time_b + t_time_f + t_time_c,
        'total_predict_time': p_time_b + p_time_f + p_time_c,
        'building_top_features': building_top_features.to_dict(),
        'floor_top_features': floor_top_features.to_dict()
    }
    print(f"{model_name_suffix} implementation complete.")
    return predictions_df, timing_info

def calculate_2d_error(lon_true_series, lat_true_series, lon_pred_series, lat_pred_series):
    return np.sqrt((lon_true_series.values - lon_pred_series.values)**2 + (lat_true_series.values - lat_pred_series.values)**2)

def evaluate_model_performance(model_name, predictions_df, timing_info, results_output_path):
    """Evaluates a single model's performance and saves detailed metrics."""
    print(f"\n--- Evaluating {model_name} ---")
    model_eval = {}
    pred_suffix = model_name # Already includes PCA, Cosine, etc.

    # Ensure columns exist before accessing
    true_building_col = 'BUILDINGID_True'
    pred_building_col = f'BUILDINGID_Pred_{pred_suffix}'
    true_floor_col = 'FLOOR_True'
    pred_floor_col = f'FLOOR_Pred_{pred_suffix}'
    lon_true_col = 'LONGITUDE_True'
    lat_true_col = 'LATITUDE_True'
    lon_pred_col = f'LONGITUDE_Pred_{pred_suffix}'
    lat_pred_col = f'LATITUDE_Pred_{pred_suffix}'

    # Check if all necessary prediction columns are in predictions_df
    required_cols = [pred_building_col, pred_floor_col, lon_pred_col, lat_pred_col]
    if not all(col in predictions_df.columns for col in required_cols):
        print(f"Error: Missing one or more prediction columns for {model_name} in predictions_df. Available: {predictions_df.columns}")
        # Add placeholder for missing metrics to avoid crashing the summary
        model_eval["Building_Accuracy"] = np.nan
        model_eval["Floor_Accuracy_Overall"] = np.nan
        model_eval["Floor_Accuracy_Conditional_Correct_Building"] = np.nan
        model_eval["Mean_2D_Position_Error"] = np.nan
        model_eval["Median_2D_Position_Error"] = np.nan
        model_eval["Std_2D_Position_Error"] = np.nan
        for threshold in [1, 3, 5, 7.5, 10, 15, 20]:
            model_eval[f"Error_Below_{threshold}m_Percentage"] = np.nan
        model_eval["Timing"] = timing_info
        return model_eval, predictions_df # Return original df if error

    model_eval["Building_Accuracy"] = accuracy_score(predictions_df[true_building_col], predictions_df[pred_building_col])
    model_eval["Floor_Accuracy_Overall"] = accuracy_score(predictions_df[true_floor_col], predictions_df[pred_floor_col])

    correct_building_mask = (predictions_df[true_building_col] == predictions_df[pred_building_col])
    if correct_building_mask.sum() > 0:
        model_eval["Floor_Accuracy_Conditional_Correct_Building"] = accuracy_score(
            predictions_df[true_floor_col][correct_building_mask],
            predictions_df[pred_floor_col][correct_building_mask]
        )
    else:
        model_eval["Floor_Accuracy_Conditional_Correct_Building"] = 0.0

    position_errors = calculate_2d_error(predictions_df[lon_true_col], predictions_df[lat_true_col], predictions_df[lon_pred_col], predictions_df[lat_pred_col])
    predictions_df["Position_Error"] = position_errors # Add error column to this specific model's prediction_df

    model_eval["Mean_2D_Position_Error"] = position_errors.mean()
    model_eval["Median_2D_Position_Error"] = np.median(position_errors)
    model_eval["Std_2D_Position_Error"] = position_errors.std()

    for threshold in [1, 3, 5, 7.5, 10, 15, 20]:
        model_eval[f"Error_Below_{threshold}m_Percentage"] = (position_errors < threshold).mean() * 100
    
    model_eval["Mean_Error_Per_Building"] = predictions_df.groupby(true_building_col)["Position_Error"].mean().to_dict()
    model_eval["Mean_Error_Per_Floor_Overall"] = predictions_df.groupby(true_floor_col)["Position_Error"].mean().to_dict()
    # More detailed stats per building/floor
    error_stats_bf = predictions_df.groupby([true_building_col, true_floor_col])["Position_Error"].agg(["mean", "median", "std", "count", "max"])
    model_eval["Error_Stats_Per_Building_Floor"] = error_stats_bf.reset_index().to_dict(orient="records")
    
    model_eval["Timing"] = timing_info
    return model_eval, predictions_df # Return predictions_df with Position_Error column

def generate_visualizations(evaluation_summary_df, all_model_predictions_with_error, dataset_name, output_dir):
    """Generates and saves comparison plots for a dataset."""
    print(f"\n--- Generating visualizations for {dataset_name} ---")
    visualization_paths = {}
    plot_params = {'rotation': 45, 'ha': "right"}

    plt.figure(figsize=(12, 7)); sns.barplot(x=evaluation_summary_df.index, y="Mean_2D_Position_Error", data=evaluation_summary_df.sort_values(by="Mean_2D_Position_Error")); plt.title(f"Mean 2D Position Error - {dataset_name}"); plt.ylabel("Mean Error (meters)"); plt.xlabel("Model"); plt.xticks(**plot_params); plt.tight_layout(); path = os.path.join(output_dir, f"{dataset_name}_mean_pos_error_v1.png"); plt.savefig(path); visualization_paths["mean_error_plot"] = path; plt.close()
    plt.figure(figsize=(12, 7)); sns.barplot(x=evaluation_summary_df.index, y="Building_Accuracy", data=evaluation_summary_df.sort_values(by="Building_Accuracy", ascending=False)); plt.title(f"Building ID Accuracy - {dataset_name}"); plt.ylabel("Accuracy"); plt.xlabel("Model"); plt.xticks(**plot_params); plt.ylim(0, 1); plt.tight_layout(); path = os.path.join(output_dir, f"{dataset_name}_building_accuracy_v1.png"); plt.savefig(path); visualization_paths["building_accuracy_plot"] = path; plt.close()
    plt.figure(figsize=(12, 7)); sns.barplot(x=evaluation_summary_df.index, y="Floor_Accuracy_Overall", data=evaluation_summary_df.sort_values(by="Floor_Accuracy_Overall", ascending=False)); plt.title(f"Overall Floor Accuracy - {dataset_name}"); plt.ylabel("Accuracy"); plt.xlabel("Model"); plt.xticks(**plot_params); plt.ylim(0, 1); plt.tight_layout(); path = os.path.join(output_dir, f"{dataset_name}_floor_accuracy_v1.png"); plt.savefig(path); visualization_paths["floor_accuracy_plot"] = path; plt.close()
    plt.figure(figsize=(12, 7)); sns.barplot(x=evaluation_summary_df.index, y="total_predict_time", data=evaluation_summary_df.sort_values(by="total_predict_time")); plt.title(f"Prediction Time - {dataset_name}"); plt.ylabel("Time (seconds)"); plt.xlabel("Model"); plt.xticks(**plot_params); plt.tight_layout(); path = os.path.join(output_dir, f"{dataset_name}_prediction_time_v1.png"); plt.savefig(path); visualization_paths["prediction_time_plot"] = path; plt.close()

    # CDF of errors
    plt.figure(figsize=(12, 7))
    for model_name, model_df in all_model_predictions_with_error.items():
        errors = model_df["Position_Error"].sort_values().values
        y = np.arange(1, len(errors) + 1) / len(errors)
        plt.plot(errors, y, label=model_name)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("2D Position Error (meters)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"CDF of 2D Position Error - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_error_cdf_v1.png")
    plt.savefig(path)
    visualization_paths["error_cdf_plot"] = path
    plt.close()

    # If Random Forest is included, plot feature importances
    if 'RandomForest' in evaluation_summary_df.index:
        rf_timing_info = evaluation_summary_df.loc['RandomForest', 'Timing']
        if 'building_top_features' in rf_timing_info and 'floor_top_features' in rf_timing_info:
            # Building feature importance
            building_features = pd.DataFrame(rf_timing_info['building_top_features'])
            plt.figure(figsize=(12, 7))
            sns.barplot(x='importance', y='feature', data=building_features)
            plt.title(f"Top 10 Features for Building Prediction - {dataset_name}")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()
            path = os.path.join(output_dir, f"{dataset_name}_rf_building_features_v1.png")
            plt.savefig(path)
            visualization_paths["rf_building_features_plot"] = path
            plt.close()
            
            # Floor feature importance
            floor_features = pd.DataFrame(rf_timing_info['floor_top_features'])
            plt.figure(figsize=(12, 7))
            sns.barplot(x='importance', y='feature', data=floor_features)
            plt.title(f"Top 10 Features for Floor Prediction - {dataset_name}")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()
            path = os.path.join(output_dir, f"{dataset_name}_rf_floor_features_v1.png")
            plt.savefig(path)
            visualization_paths["rf_floor_features_plot"] = path
            plt.close()

    print(f"Visualizations generated for {dataset_name}.")
    return visualization_paths

def process_single_dataset(dataset_name, base_project_dir, results_base_path):
    """Processes a single dataset through the entire pipeline."""
    print(f"\n\n=== Processing Dataset: {dataset_name} ===")
    
    # Create output directory for this dataset
    dataset_output_path = os.path.join(results_base_path, dataset_name)
    create_output_directory(results_base_path, dataset_name)
    
    # Load data
    dataset_input_path = os.path.join(base_project_dir, dataset_name)
    train_df, test_df = load_single_dataset(dataset_input_path, dataset_name)
    if train_df is None or test_df is None:
        print(f"Error: Could not load data for {dataset_name}. Skipping.")
        return None
    
    # Preprocess data
    X_train, y_train_building, y_train_floor, y_train_coords, X_test, y_test_building, y_test_floor, y_test_coords, test_df = preprocess_dataset_data(train_df, test_df, dataset_name)
    
    # Train and evaluate models
    models_to_evaluate = {
        "KNN": {"use_pca": False, "metric": "minkowski", "weights": "uniform"},
        "WKNN": {"use_pca": False, "metric": "minkowski", "weights": "distance"},
        "KNN_PCA": {"use_pca": True, "metric": "minkowski", "weights": "uniform"},
        "WKNN_Cosine": {"use_pca": False, "metric": "cosine", "weights": "distance"},
        "WKNN_Manhattan": {"use_pca": False, "metric": "manhattan", "weights": "distance"}
    }
    
    model_evaluations = {}
    all_model_predictions = {}
    
    for model_name, model_params in models_to_evaluate.items():
        predictions_df, timing_info = train_and_predict_model(
            model_name, 
            X_train, y_train_building, y_train_floor, y_train_coords,
            X_test, y_test_building, y_test_floor, y_test_coords,
            K_NEIGHBORS,
            use_pca=model_params["use_pca"],
            metric=model_params["metric"],
            weights=model_params["weights"]
        )
        model_eval, predictions_with_error = evaluate_model_performance(model_name, predictions_df, timing_info, dataset_output_path)
        model_evaluations[model_name] = model_eval
        all_model_predictions[model_name] = predictions_with_error
    
    # Add Random Forest model
    rf_predictions_df, rf_timing_info = train_and_predict_random_forest(
        "RandomForest",
        X_train, y_train_building, y_train_floor, y_train_coords,
        X_test, y_test_building, y_test_floor, y_test_coords
    )
    rf_model_eval, rf_predictions_with_error = evaluate_model_performance("RandomForest", rf_predictions_df, rf_timing_info, dataset_output_path)
    model_evaluations["RandomForest"] = rf_model_eval
    all_model_predictions["RandomForest"] = rf_predictions_with_error
    
    # Create summary dataframe
    evaluation_summary = pd.DataFrame(model_evaluations).T
    
    # Extract timing information to separate columns
    timing_columns = ['total_train_time', 'total_predict_time', 'pca_components']
    for col in timing_columns:
        evaluation_summary[col] = evaluation_summary['Timing'].apply(lambda x: x.get(col, None))
    
    # Save evaluation summary to CSV
    evaluation_summary_path = os.path.join(dataset_output_path, f"{dataset_name}_evaluation_summary_v1.csv")
    evaluation_summary.to_csv(evaluation_summary_path)
    print(f"Evaluation summary saved to {evaluation_summary_path}")
    
    # Generate visualizations
    visualization_paths = generate_visualizations(evaluation_summary, all_model_predictions, dataset_name, dataset_output_path)
    
    # Determine best model based on mean position error
    best_model_name_eval = evaluation_summary["Mean_2D_Position_Error"].idxmin()
    print(f"Best model for {dataset_name} based on Mean 2D Position Error: {best_model_name_eval}")
    
    # Analyze errors for best model
    pred_building_col_best_eval = f'BUILDINGID_Pred_{best_model_name_eval}'
    pred_floor_col_best_eval = f'FLOOR_Pred_{best_model_name_eval}'
    
    # Analyze errors by floor for best model
    floor_errors = all_model_predictions[best_model_name_eval].groupby('FLOOR_True')['Position_Error'].agg(['mean', 'median', 'std', 'count'])
    floor_errors_path = os.path.join(dataset_output_path, f"{dataset_name}_{best_model_name_eval}_floor_errors_v1.csv")
    floor_errors.to_csv(floor_errors_path)
    print(f"Floor errors analysis saved to {floor_errors_path}")
    
    # Return summary for global comparison
    return {
        "dataset_name": dataset_name,
        "evaluation_summary": evaluation_summary,
        "best_model": best_model_name_eval,
        "best_model_error": evaluation_summary.loc[best_model_name_eval, "Mean_2D_Position_Error"],
        "visualization_paths": visualization_paths
    }

def main():
    """Main function to process all datasets."""
    print("=== WiFi Indoor Localization Batch Analysis with Random Forest ===")
    
    # Base paths
    base_project_dir = "/home/cariik4t/tfm_v1/mtfm/data"
    results_base_dir = os.path.join(base_project_dir, "results_v1")
    
    # Create results directory
    if not os.path.exists(results_base_dir):
        os.makedirs(results_base_dir)
    
    # Datasets to process
    datasets = ["UJI1", "UTS1", "TUT1", "TUT2", "TUT3", "TUT4", "TUT5", "TUT6", "TUT7"]
    
    # Process each dataset
    all_datasets_results = []
    for dataset in datasets:
        result = process_single_dataset(dataset, base_project_dir, results_base_dir)
        if result:
            all_datasets_results.append(result)
    
    # Create global summary
    if all_datasets_results:
        # Extract evaluation summaries
        all_summaries = []
        for result in all_datasets_results:
            summary = result["evaluation_summary"].copy()
            summary["Dataset"] = result["dataset_name"]
            all_summaries.append(summary)
        
        # Combine all summaries
        global_summary = pd.concat(all_summaries)
        global_summary_path = os.path.join(base_project_dir, "all_datasets_evaluation_summary_v1.csv")
        global_summary.to_csv(global_summary_path)
        print(f"Global evaluation summary saved to {global_summary_path}")
        
        # Create comparison visualizations across datasets
        plt.figure(figsize=(14, 8))
        #pivot_data = global_summary.pivot(index="Dataset", columns=global_summary.index, values="Mean_2D_Position_Error")
        pivot_data = global_summary.reset_index().pivot(index="Dataset", columns="index", values="Mean_2D_Position_Error")
        # Asegura que los datos del heatmap sean float
        pivot_data = pivot_data.astype(float)
        sns.heatmap(pivot_data, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Mean 2D Position Error Across All Datasets and Models")
        plt.tight_layout()
        heatmap_path = os.path.join(base_project_dir, "mean_error_heatmap_v1.png")
        plt.savefig(heatmap_path)
        plt.close()
        
        # Create bar chart of best model for each dataset
        best_models = pd.DataFrame([
            {"Dataset": result["dataset_name"], 
             "Best_Model": result["best_model"], 
             "Error": result["best_model_error"]} 
            for result in all_datasets_results
        ])
        plt.figure(figsize=(12, 7))
        bars = sns.barplot(x="Dataset", y="Error", data=best_models, hue="Best_Model", palette="Set2")
        plt.title("Best Model and Error for Each Dataset")
        plt.ylabel("Mean 2D Position Error (meters)")
        plt.xticks(rotation=45, ha="right")
        
        # Add value labels on bars
        for i, bar in enumerate(bars.patches):
            bars.text(
                bar.get_x() + bar.get_width()/2., 
                bar.get_height() + 0.3, 
                f"{bar.get_height():.2f}m", 
                ha='center', va='bottom', 
                color='black', fontweight='bold'
            )
        
        plt.tight_layout()
        best_models_path = os.path.join(base_project_dir, "best_models_comparison_v1.png")
        plt.savefig(best_models_path)
        plt.close()
        
        # Create model ranking summary
        model_ranks = global_summary.groupby(global_summary.index)["Mean_2D_Position_Error"].rank()
        avg_ranks = model_ranks.groupby(level=0).mean()
        win_counts = (model_ranks == 1).groupby(level=0).sum()
        
        ranking_summary = pd.DataFrame({
            "Average_Rank": avg_ranks,
            "Win_Count": win_counts
        }).sort_values("Average_Rank")
        
        ranking_path = os.path.join(base_project_dir, "model_ranking_summary_v1.csv")
        ranking_summary.to_csv(ranking_path)
        print(f"Model ranking summary saved to {ranking_path}")
        
        # Create model ranking visualization
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x=ranking_summary.index, y="Average_Rank", data=ranking_summary)
        plt.title("Average Rank of Models Across All Datasets")
        plt.ylabel("Average Rank (lower is better)")
        plt.xlabel("Model")
        plt.xticks(rotation=45, ha="right")
        
        # Add win count labels
        for i, p in enumerate(ax.patches):
            model_name = ranking_summary.index[i]
            wins = ranking_summary.loc[model_name, "Win_Count"]
            ax.annotate(f"Wins: {int(wins)}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'bottom',
                       xytext = (0, 5), textcoords = 'offset points')
        
        plt.tight_layout()
        ranking_viz_path = os.path.join(base_project_dir, "model_ranking_visualization_v1.png")
        plt.savefig(ranking_viz_path)
        plt.close()
        
        print("All datasets processed successfully.")
    else:
        print("No datasets were processed successfully.")

if __name__ == "__main__":
    main()
