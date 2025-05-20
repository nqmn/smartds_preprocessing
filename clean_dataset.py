import numpy as np
import pandas as pd

def clean_dataset(X, y=None, epsilon=1e-10):
    def _convert_to_dataframe(X):
        if isinstance(X, pd.DataFrame):
            return X.copy(), True
        try:
            return pd.DataFrame(X), False
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not convert to DataFrame due to: {e}. Limited cleaning will be applied.")
            variances = np.var(X, axis=0)
            valid_indices = np.where(variances > epsilon)[0]
            X_cleaned = X[:, valid_indices]
            print(f"Removed {X.shape[1] - len(valid_indices)} constant features")
            print(f"Retained {len(valid_indices)} numeric features")
            return X_cleaned, None

    def _remove_non_numeric_columns(df):
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        non_numeric_count = df.shape[1] - len(numeric_cols)
        df = df[numeric_cols]
        if non_numeric_count > 0:
            print(f"Removed {non_numeric_count} non-numeric columns")
        return df, numeric_cols

    def _remove_inf_columns(df):
        inf_mask = np.isinf(df.values)
        if inf_mask.any():
            inf_columns = df.columns[np.any(inf_mask, axis=0)].tolist()
            df = df.drop(columns=inf_columns)
            print(f"Removed {len(inf_columns)} columns containing infinity values")
        return df

    def _remove_constant_columns(df, epsilon):
        if df.empty:
            print("Warning: No valid columns remain after filtering")
            return pd.DataFrame(index=df.index), []

        variances = np.var(df.values, axis=0)
        non_constant_indices = np.where(variances > epsilon)[0]
        retained_df = df.iloc[:, non_constant_indices]
        removed_cols = df.columns.difference(retained_df.columns).tolist()
        if removed_cols:
            print(f"Removed {len(removed_cols)} constant numeric features: {removed_cols}")
        return retained_df, retained_df.columns.tolist()

    def _remove_classwise_constant_columns(df, y):
        features_to_remove = []
        grouped = df.groupby(y)
        for col in df.columns:
            unique_vals = grouped[col].nunique(dropna=False)
            if all(n == 1 for n in unique_vals):
                features_to_remove.append(col)
        if features_to_remove:
            df = df.drop(columns=features_to_remove)
            print(f"Removed {len(features_to_remove)} features constant within each class: {features_to_remove}")
        return df

    # Step 0: Convert to DataFrame if possible
    X_df, was_dataframe = _convert_to_dataframe(X)
    if was_dataframe is None:
        return X_df  # early exit for raw NumPy case with limited cleaning

    initial_columns = X_df.shape[1]

    # Step 1: Remove non-numeric columns
    X_df, numeric_columns = _remove_non_numeric_columns(X_df)

    # Step 2: Remove columns with infinity values
    X_df = _remove_inf_columns(X_df)

    # Step 3: Remove constant columns
    X_df, valid_columns = _remove_constant_columns(X_df, epsilon)

    # Step 4: Remove features constant within each class
    if y is not None and X_df.shape[0] == len(y):
        X_df = _remove_classwise_constant_columns(X_df, y)
        valid_columns = X_df.columns.tolist()  # update after step 4

    # Final cleanup
    total_removed = initial_columns - len(valid_columns)
    print(f"Total features removed: {total_removed}")
    print(f"Retained {len(valid_columns)} clean numeric features")

    X_cleaned = X_df.to_numpy()

    if was_dataframe:
        original_indices = [list(X.columns).index(col) for col in valid_columns]
        return X_cleaned, original_indices
    else:
        numeric_indices = [i for i in range(initial_columns) if i < len(numeric_columns)]
        inf_columns = list(set(numeric_columns) - set(X_df.columns))
        filtered_indices = [i for i in range(len(numeric_columns)) if numeric_columns[i] not in inf_columns]
        valid_indices = [filtered_indices[i] for i in range(len(filtered_indices)) if i < len(X_df.columns)]
        return X_cleaned, valid_indices
