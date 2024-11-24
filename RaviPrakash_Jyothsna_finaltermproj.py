import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin




def preprocess_data(df):
    """
    Preprocess the input DataFrame for machine learning tasks.

    Args:
        df (pd.DataFrame): The input dataset with features and target.

    Returns:
        tuple: Processed features (X_train, X_test) and target (y_train, y_test).
    """
    # Identifying numerical columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Explicitly defining categorical columns based on data understanding
    cat_cols = ['Gender', 'EducationLevel', 'RecruitmentStrategy', 'HiringDecision']

    # Impute missing values in numerical columns with median
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Impute missing values in categorical columns with mode
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['EducationLevel'] = label_encoder.fit_transform(df['EducationLevel'])
    df['RecruitmentStrategy'] = label_encoder.fit_transform(df['RecruitmentStrategy'])

    # # Plot the distribution of the target variable
    # plt.figure(figsize=(6, 4))
    # sns.countplot(x='HiringDecision', data=df, palette='coolwarm')
    # plt.title('Distribution of Hiring Decision')
    # plt.xlabel('Hiring Decision')
    # plt.ylabel('Count')
    # plt.show()

    # Encode additional categorical variables
    df['Gender'] = df['Gender'].astype('category').cat.codes
    df['EducationLevel'] = df['EducationLevel'].astype('category').cat.codes
    df['RecruitmentStrategy'] = df['RecruitmentStrategy'].astype('category').cat.codes

    # # Visualize the correlation heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    # plt.title('Correlation Heatmap')
    # plt.show()

    # # Visualize pairwise relationships
    # sns.pairplot(df, hue='HiringDecision', diag_kind='kde', palette='coolwarm')
    # plt.show()

    # Split dataset into features and target
    X_features = df.drop(columns=['HiringDecision'])
    y_target = df['HiringDecision']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_target, test_size=0.3, random_state=42)

    # Convert target variable to numpy arrays
    y_train = y_train.to_numpy()

    # Print the split data shapes
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Test target shape: {y_test.shape}")

    # Return the processed data
    return df, X_train, X_test, y_train, y_test




# Define the function for Random Forest evaluation
def random_forest_model(X_train, y_train):
    # Best parameters from hyperparameter tuning
    best_rf_model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        random_state=42
    )
    
    # KFold Cross-validation setup
    cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_per_fold = []

    # Aggregate true labels and predicted probabilities for ROC curve
    y_true_all = []
    y_proba_all = []

    for fold_number, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train), 1):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Fit model on the fold
        best_rf_model.fit(X_train_fold, y_train_fold)
        y_pred = best_rf_model.predict(X_val_fold)
        y_proba = best_rf_model.predict_proba(X_val_fold)[:, 1]

        # Calculate confusion matrix components
        # Calculate confusion matrix components
        tp = tn = fp = fn = 0
        for true, pred in zip(y_val_fold, y_pred):
            if true == 1 and pred == 1:
                tp += 1
            elif true == 0 and pred == 0:
                tn += 1
            elif true == 0 and pred == 1:
                fp += 1
            elif true == 1 and pred == 0:
                fn += 1

        # Calculate metrics
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        TNR = tn / (tn + fp) if (tn + fp) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
        FNR = fn / (fn + tp) if (fn + tp) > 0 else 0
        Precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        F1 = 2 * Precision * TPR / (Precision + TPR) if (Precision + TPR) > 0 else 0
        Accuracy = (tp + tn) / (tp + tn + fp + fn)
        Error_rate = 1 - Accuracy
        BACC = (TPR + TNR) / 2
        TSS = TPR + TNR - 1
        HSS = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) if ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) > 0 else 0
        Brier_score = brier_score_loss(y_val_fold, y_proba)
        AUC = roc_auc_score(y_val_fold, y_proba)

        # Append metrics for this fold
        metrics_per_fold.append([fold_number, tp, tn, fp, fn, TPR, TNR, FPR, FNR, Precision, F1, Accuracy, Error_rate, BACC, TSS, HSS, Brier_score, AUC])

        # Aggregate for ROC curve
        y_true_all.extend(y_val_fold)
        y_proba_all.extend(y_proba)

    # Create DataFrame with metrics
    metrics_df = pd.DataFrame(metrics_per_fold, columns=[
        'Fold', 'TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR',
        'Precision', 'F1_measure', 'Accuracy', 'Error_rate', 'BACC',
        'TSS', 'HSS', 'Brier_score', 'AUC'
    ])

    # Add average metrics row
    avg_metrics = metrics_df.mean(numeric_only=True)
    metrics_df = pd.concat([metrics_df, avg_metrics.to_frame().T], ignore_index=True)

    # Plot ROC curve
    y_true_all = np.array(y_true_all)
    y_proba_all = np.array(y_proba_all)
    fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
    roc_auc = roc_auc_score(y_true_all, y_proba_all)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Random Forest Model (10-Fold CV)')
    plt.legend(loc='lower right')
    plt.show()

    return metrics_df, roc_auc




def svm_model(X_train, y_train):

    
    # Define the best SVM model from pre-tuned parameters
    best_svm = SVC(C=1, kernel='rbf', gamma='scale', probability=True, random_state=42)
    
    # KFold Cross-validation setup
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []
    y_true_all = []
    y_proba_all = []

    # Perform 10-Fold Cross-validation
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Fit the model
        best_svm.fit(X_train_fold, y_train_fold)

        # Predictions and probabilities
        y_pred = best_svm.predict(X_val_fold)
        y_proba = best_svm.predict_proba(X_val_fold)[:, 1]

        # Aggregate labels and probabilities for ROC AUC
        y_true_all.extend(y_val_fold)
        y_proba_all.extend(y_proba)

        # Confusion matrix components
        tp = tn = fp = fn = 0
        for true, pred in zip(y_val_fold, y_pred):
            if true == 1 and pred == 1:
                tp += 1
            elif true == 0 and pred == 0:
                tn += 1
            elif true == 0 and pred == 1:
                fp += 1
            elif true == 1 and pred == 0:
                fn += 1

        # Calculate metrics
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        TNR = tn / (tn + fp) if (tn + fp) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
        FNR = fn / (fn + tp) if (fn + tp) > 0 else 0
        Precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        F1 = 2 * Precision * TPR / (Precision + TPR) if (Precision + TPR) > 0 else 0
        Accuracy = (tp + tn) / (tp + tn + fp + fn)
        Error_rate = 1 - Accuracy
        BACC = (TPR + TNR) / 2
        TSS = TPR + TNR - 1
        HSS = (
            2 * (tp * tn - fp * fn)
            / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
            if ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) > 0
            else 0
        )
        Brier_score = brier_score_loss(y_val_fold, y_proba)
        AUC = roc_auc_score(y_val_fold, y_proba)

        # Append metrics for this fold
        metrics_list.append([fold, tp, tn, fp, fn, TPR, TNR, FPR, FNR, Precision, F1, Accuracy, Error_rate, BACC, TSS, HSS, Brier_score, AUC])

    # Create DataFrame with metrics
    metrics_svm = pd.DataFrame(metrics_list, columns=[
        'Fold', 'TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR',
        'Precision', 'F1_measure', 'Accuracy', 'Error_rate', 'BACC',
        'TSS', 'HSS', 'Brier_score', 'AUC'
    ])

    # Calculate average metrics across folds
    metrics_svm.loc['Average'] = metrics_svm.mean(numeric_only=True)

    # Calculate overall ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
    roc_auc = roc_auc_score(y_true_all, y_proba_all)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for SVM Model (10-Fold CV)')
    plt.legend(loc='lower right')
    plt.show()

    # Return the metrics DataFrame
    return metrics_svm, roc_auc






def decision_tree_model(X_train, y_train):

    # Define the best Decision Tree model with tuned hyperparameters
    best_dt = DecisionTreeClassifier(min_samples_leaf=10, random_state=42)
    
    # KFold Cross-validation setup
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Prepare to store results in a list
    metrics_list = []
    
    # Aggregate true labels and predicted probabilities across all folds
    y_true_all = []
    y_proba_all = []
    
    # Perform 10-Fold Cross-validation
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        # Split the data into training and validation sets
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        # Fit the model on the training data of the current fold
        best_dt.fit(X_train_fold, y_train_fold)
        
        # Predict on the validation data
        y_pred = best_dt.predict(X_val_fold)
        y_proba = best_dt.predict_proba(X_val_fold)[:, 1]
        
        # Store the true labels and probabilities for overall ROC calculation
        y_true_all.extend(y_val_fold)
        y_proba_all.extend(y_proba)
        
        # Calculate confusion matrix components
        tp = tn = fp = fn = 0
        for true, pred in zip(y_val_fold, y_pred):
            if true == 1 and pred == 1:
                tp += 1
            elif true == 0 and pred == 0:
                tn += 1
            elif true == 0 and pred == 1:
                fp += 1
            elif true == 1 and pred == 0:
                fn += 1
        
        # Calculate performance metrics
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        TNR = tn / (tn + fp) if (tn + fp) > 0 else 0
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
        FNR = fn / (fn + tp) if (fn + tp) > 0 else 0
        Precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        F1 = 2 * Precision * TPR / (Precision + TPR) if (Precision + TPR) > 0 else 0
        Accuracy = (tp + tn) / (tp + tn + fp + fn)
        Error_rate = 1 - Accuracy
        BACC = (TPR + TNR) / 2
        TSS = TPR + TNR - 1
        HSS = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) if ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) > 0 else 0
        Brier_score = brier_score_loss(y_val_fold, y_proba)
        AUC = roc_auc_score(y_val_fold, y_proba)
        
        # Append the metrics for the current fold
        metrics_list.append([fold, tp, tn, fp, fn, TPR, TNR, FPR, FNR, Precision, F1, Accuracy, Error_rate, BACC, TSS, HSS, Brier_score, AUC])
    
    # Create a DataFrame from the metrics list
    metrics_dt = pd.DataFrame(metrics_list, columns=[
        'Fold', 'TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR',
        'Precision', 'F1_measure', 'Accuracy', 'Error_rate', 'BACC',
        'TSS', 'HSS', 'Brier_score', 'AUC'
    ])
    metrics_dt.loc['Average'] = metrics_dt.mean(numeric_only=True)
    
    # Convert lists to numpy arrays for ROC calculations
    y_true_all = np.array(y_true_all)
    y_proba_all = np.array(y_proba_all)
    
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_all, y_proba_all)
    
    # Calculate the ROC AUC score
    roc_auc = roc_auc_score(y_true_all, y_proba_all)
    
    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Decision Tree Model (10-Fold CV)')
    plt.legend(loc='lower right')
    plt.show()
    
    return metrics_dt, roc_auc






def lstm_model(df):
    # Encode 'Gender' (binary) and 'EducationLevel' (ordinal)
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    df['EducationLevel'] = label_encoder.fit_transform(df['EducationLevel'])
    df['RecruitmentStrategy'] = label_encoder.fit_transform(df['RecruitmentStrategy'])

    # Separate features and target
    X = df.drop('HiringDecision', axis=1).values  # Use 'HiringDecision' as target variable
    y = df['HiringDecision'].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reshape data for LSTM [samples, time steps, features]
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    # Define LSTM model
    def create_lstm_model(learning_rate=0.01, dropout_rate=0.2):
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Perform 10-fold cross-validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_list = []
    y_true_all = []
    y_proba_all = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Create and train the model
        model = create_lstm_model()
        model.fit(X_train_fold, y_train_fold, epochs=20, batch_size=32, verbose=0)

        # Predict on the validation fold
        y_pred = (model.predict(X_val_fold) > 0.5).astype("int32").flatten()
        y_proba = model.predict(X_val_fold).flatten()

        # Aggregate true labels and predicted probabilities
        y_true_all.extend(y_val_fold)
        y_proba_all.extend(y_proba)

        # Calculate confusion matrix components
        tp = tn = fp = fn = 0
        for true, pred in zip(y_val_fold, y_pred):
            if true == 1 and pred == 1:
                tp += 1
            elif true == 0 and pred == 0:
                tn += 1
            elif true == 0 and pred == 1:
                fp += 1
            elif true == 1 and pred == 0:
                fn += 1

        # Calculate metrics
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
        TNR = tn / (tn + fp) if (tn + fp) > 0 else 0
        Precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        F1 = 2 * Precision * TPR / (Precision + TPR) if (Precision + TPR) > 0 else 0
        Accuracy = (tp + tn) / (tp + tn + fp + fn)
        Brier_score = brier_score_loss(y_val_fold, y_proba)
        AUC = roc_auc_score(y_val_fold, y_proba)

        # Append metrics
        metrics_list.append([fold, tp, tn, fp, fn, TPR, TNR, Precision, F1, Accuracy, Brier_score, AUC])

    # Create DataFrame with fold metrics
    metrics_df = pd.DataFrame(metrics_list, columns=[
        'Fold', 'TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'Precision', 'F1_measure',
        'Accuracy', 'Brier_score', 'AUC'
    ])

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(np.array(y_true_all), np.array(y_proba_all))
    roc_auc = roc_auc_score(np.array(y_true_all), np.array(y_proba_all))

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for LSTM Model (10-Fold CV)')
    plt.legend(loc='lower right')
    plt.show()

    return metrics_df, roc_auc





def merged_metrics(metrics_rf, metrics_svm, metrics_dt, metrics_lstm):

    # Add a new column to each DataFrame for the model name
    metrics_rf['Model'] = 'Random Forest'
    metrics_svm['Model'] = 'SVM'
    metrics_dt['Model'] = 'Decision Tree'
    metrics_lstm['Model'] = 'LSTM'

    # Select the last row from each DataFrame and merge them vertically
    merged_metrics = pd.concat([metrics_rf[-1:], metrics_svm[-1:], metrics_dt[-1:], metrics_lstm[-1:]], ignore_index=True)

    # Move the 'Model' column to the first position
    merged_metrics = merged_metrics[['Model'] + [col for col in merged_metrics.columns if col != 'Model']]

    # Transpose the DataFrame to have models as column names
    merged_metrics = merged_metrics.set_index('Model').T

    # Display the DataFrame with bold model names in the first row
    # Formatting the model names as bold for display (works in Jupyter environments)
    merged_metrics.columns = [f"{col}" for col in merged_metrics.columns]

    # Display the transposed DataFrame
    print(merged_metrics)

    return merged_metrics


def best_model_result(merged_metrics):
    # Assuming merged_metrics is the transposed DataFrame with model metrics
    # Select key metrics for comparison
    key_metrics = ['Accuracy', 'AUC', 'Precision', 'F1_measure', 'BACC', 'HSS']

    # Filter the merged metrics DataFrame to only key metrics
    metrics_for_ranking = merged_metrics.loc[key_metrics]

    # Rank each model for each metric (higher is better, so we rank by descending values)
    ranks = metrics_for_ranking.rank(ascending=False, axis=1)

    # Sum ranks for each model to get a total score (lower score indicates better performance)
    total_scores = ranks.sum()

    # Find the model with the lowest total score
    best_model = total_scores.idxmin()

    # Display the ranking results and the best model
    print("\nRanking of Models by Metrics:\n", ranks)
    print("\nTotal Scores for Each Model:\n", total_scores)
    print(f"\nBest Model Overall: {best_model}")




def main():
    # Path to the dataset
    file_path = 'recruitment_data.csv'
    df=pd.read_csv(file_path)

    # Step 1: Preprocess the dataset
    df_preprocessed, X_train, X_test, y_train, y_test = preprocess_data(df)
    # print(df_preprocessed)



    print("\nData is prepared for model training and testing.")

    # Random Forest Classifier
    metrics_rf, roc_auc_rf  = random_forest_model(X_train, y_train)
    print("\nRandom Forest Classifier Metrics:\n", metrics_rf)
    print(f"Average ROC AUC Score: {roc_auc_rf:.2f}")

    # SVM
    metrics_svm, roc_auc_svm = svm_model(X_train, y_train)
    print("\nSVM Metrics:\n", metrics_svm)
    print(f"Average ROC AUC Score: {roc_auc_svm:.2f}")

    # DecisionTreeClassifier
    metrics_dt, roc_auc_dt = decision_tree_model(X_train, y_train)
    print("\nDecision Tree Classifier Metrics:\n", metrics_dt)
    print(f"Average ROC AUC Score: {roc_auc_dt:.2f}")

    # LSTM
    metrics_lstm, roc_auc_lstm = lstm_model(df_preprocessed)
    print("\nLSTM Model Metrics:\n", metrics_lstm)
    print(f"Average ROC AUC Score: {roc_auc_lstm:.2f}")

    #Merging all the metrics to a single table (Only the average metrics)
    merged_metric = merged_metrics(metrics_rf, metrics_svm, metrics_dt, metrics_lstm)

    #Result 
    print("\n\n")
    print("Merged Metrics : \n")
    best_model_result(merged_metric)


if __name__ == "__main__":
    main()
