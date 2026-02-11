import pandas as pd
import numpy as np
import logging
import json
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_nan_to_none(obj):
    """
    Recursively convert NaN, inf, and -inf values to None for JSON serialization.
    Handles all numpy and pandas types, nested structures, and edge cases.
    """
    # Handle None
    if obj is None:
        return None
    
    # Handle numpy arrays and pandas Series
    if isinstance(obj, (np.ndarray, pd.Series)):
        return convert_nan_to_none(obj.tolist())
    
    # Handle dictionaries
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            # Convert key if it's not a string
            clean_key = str(key) if not isinstance(key, str) else key
            result[clean_key] = convert_nan_to_none(value)
        return result
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [convert_nan_to_none(item) for item in obj]
    
    # Handle numpy scalar types
    if isinstance(obj, np.generic):
        if isinstance(obj, (np.floating, np.complexfloating)):
            if pd.isna(obj) or math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            # For other numpy types, try to convert
            try:
                if pd.isna(obj):
                    return None
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            except (ValueError, TypeError):
                return str(obj)
    
    # Handle Python float
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    
    # Handle pandas nullable types (Int64, Float64, etc.)
    if hasattr(obj, '__class__') and 'pandas' in str(type(obj)):
        try:
            if pd.isna(obj):
                return None
            # Try to convert to Python native type
            if hasattr(obj, 'item'):
                return convert_nan_to_none(obj.item())
        except (ValueError, TypeError, AttributeError):
            pass
    
    # Check for NaN using pandas
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    
    # Check for NaN using math
    try:
        if isinstance(obj, (int, float)) and (math.isnan(obj) or math.isinf(obj)):
            return None
    except (TypeError, ValueError):
        pass
    
    # Return as-is if it's a basic JSON-serializable type
    if isinstance(obj, (str, int, bool)):
        return obj
    
    # For any other type, try to convert to string as last resort
    try:
        return str(obj)
    except:
        return None

def read_and_validate_file(uploaded_file, sheet_name=None):
    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith('.xlsx'):
            excel_file = pd.ExcelFile(uploaded_file)
            if sheet_name is None:
                sheet_name = excel_file.sheet_names[0]
            df = excel_file.parse(sheet_name)
        else:
            logging.error("Unsupported file format. Please upload a CSV or XLSX file.")
            return None
        
        if df.empty:
            logging.error("The file is empty. Please upload a valid dataset.")
            return None
    
        return df
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return None


def clean_data(df):
    """
    Clean the dataframe, but do NOT fill missing values with mean/median/mode.
    Keep NaN as missing. Do not drop columns with high missing % -- just report it.
    Still convert booleans/yes/no, and datetime columns.
    Removes completely empty rows and columns.
    """
    try:
        if df is None or df.empty:
            logging.error("DataFrame is None or empty in clean_data")
            return None
        
        if len(df.columns) == 0:
            logging.error("DataFrame has no columns in clean_data")
            return None
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Remove completely empty rows (all values are NaN or empty strings)
        initial_rows = len(df)
        # Check for rows where all values are NaN or empty strings
        mask = df.apply(lambda row: not (row.isna().all() or (row.astype(str).str.strip().eq('').all() if len(row) > 0 else True)), axis=1)
        df = df[mask]
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            logging.info(f"Removed {rows_removed} completely empty rows")
        
        # Remove completely empty columns (all values are NaN or empty strings)
        initial_cols = len(df.columns)
        empty_cols = []
        for col in df.columns:
            # Check if all values are NaN
            if df[col].isna().all():
                empty_cols.append(col)
            # Check if all values are empty strings (after converting to string)
            elif df[col].astype(str).str.strip().eq('').all():
                empty_cols.append(col)
        
        if empty_cols:
            df = df.drop(columns=empty_cols)
            logging.info(f"Removed {len(empty_cols)} completely empty columns: {empty_cols}")
        
        # Check if dataframe is now empty after removing empty rows/columns
        if df.empty:
            logging.error("DataFrame is empty after removing empty rows and columns")
            return None
        
        if len(df.columns) == 0:
            logging.error("DataFrame has no columns after removing empty columns")
            return None
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

        logging.info(f"Before cleaning: {len(df)} rows, {len(df.columns)} cols")
        logging.info(f"Numeric cols: {len(numeric_cols)}, Categorical cols: {len(categorical_cols)}, Date cols: {len(date_cols)}")

        # Track columns that have been converted to avoid double-processing
        converted_to_numeric = set()
        
        # Convert some common categorical/bool columns
        for col in categorical_cols:
            try:
                unique_vals = df[col].dropna().astype(str).str.strip().str.lower().unique()
                if len(unique_vals) == 2:
                    if set(unique_vals) == set(["yes", "no"]):
                        mapping = {"yes": 1, "no": 0}
                        df[col] = df[col].astype(str).str.strip().str.lower().map(mapping)
                        converted_to_numeric.add(col)
                        logging.info(f"Converted '{col}' from yes/no to numeric")
                    elif set(unique_vals) == set(["true", "false"]):
                        mapping = {"true": 1, "false": 0}
                        df[col] = df[col].astype(str).str.strip().str.lower().map(mapping)
                        converted_to_numeric.add(col)
                        logging.info(f"Converted '{col}' from true/false to numeric")
            except Exception as e:
                logging.warning(f"Error processing column '{col}' for boolean conversion: {e}")
                continue

        # Try to convert date columns from string
        # Only convert columns that:
        # 1. Haven't been converted to numeric/boolean
        # 2. Have reasonable cardinality (not clearly categorical with few values)
        # 3. Actually contain date-like patterns
        for col in categorical_cols:
            if col in converted_to_numeric:
                continue  # Skip columns we've already converted
            
            try:
                # Skip columns with very few unique values that are clearly categorical
                unique_count = df[col].dropna().nunique()
                if unique_count <= 10:
                    # Check if it's clearly a categorical column (like smoking_status, recovered)
                    sample_values = df[col].dropna().astype(str).str.strip().head(20).tolist()
                    # Skip if values are clearly categorical (yes/no, true/false, or short non-date strings)
                    is_categorical = False
                    for v in sample_values[:min(10, len(sample_values))]:
                        v_lower = str(v).lower()
                        # Check if it's a boolean-like value
                        if v_lower in ['yes', 'no', 'true', 'false', 'y', 'n']:
                            is_categorical = True
                            break
                        # Check if it's a short string without date-like patterns (no digits or date separators)
                        if len(str(v)) < 15 and '/' not in str(v) and '-' not in str(v) and not any(char.isdigit() for char in str(v)[:4]):
                            is_categorical = True
                            break
                    
                    if is_categorical:
                        continue
                
                # Try to convert to datetime
                converted = pd.to_datetime(df[col], errors='coerce')
                valid_count = converted.notnull().sum()
                valid_ratio = valid_count / len(df) if len(df) > 0 else 0
                
                # More strict criteria: need high valid ratio AND actual date values in reasonable range
                if valid_ratio > 0.8:
                    # Verify these are actually reasonable dates (not epoch dates from conversion errors)
                    valid_dates = converted.dropna()
                    if len(valid_dates) > 0:
                        min_date = valid_dates.min()
                        max_date = valid_dates.max()
                        # Only convert if dates are in a reasonable range (1900-2100)
                        if min_date.year >= 1900 and max_date.year <= 2100:
                            df[col] = converted
                            logging.info(f"Converted column '{col}' to datetime")
                        else:
                            logging.info(f"Skipping datetime conversion for '{col}': dates out of reasonable range")
            except Exception as e:
                logging.info(f"Column '{col}' could not be converted to datetime: {e}")

        # Do NOT fillna on any column or drop due to missing
        # Optionally, log columns with high missing
        for col in df.columns:
            missing_percentage = df[col].isnull().mean() * 100
            if missing_percentage > 50:
                logging.info(f"Column '{col}' is highly missing: {missing_percentage:.1f}% (but kept)")

        # Remove duplicates only
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            logging.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Clean column names
        df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        
        logging.info(f"After cleaning: {len(df)} rows, {len(df.columns)} cols")
        
        # Final validation
        if df.empty:
            logging.error("DataFrame is empty after cleaning")
            return None
        
        if len(df.columns) == 0:
            logging.error("DataFrame has no columns after cleaning")
            return None
        
        return df
    except Exception as e:
        logging.error(f"Error during data cleaning: {e}", exc_info=True)
        return None


def enhanced_eda_json(df):
    try:
        # Validate input
        if df is None or df.empty:
            logging.error("DataFrame is None or empty")
            return None
        
        if len(df.columns) == 0:
            logging.error("DataFrame has no columns")
            return None
        
        eda_summary = {}
        eda_summary["num_rows"] = df.shape[0]
        eda_summary["num_columns"] = df.shape[1]
        columns_info = {}
        
        for col in df.columns:
            try:
                col_info = {}
                col_info["dtype"] = str(df[col].dtype)
                missing_count = int(df[col].isnull().sum())
                missing_percent = round(df[col].isnull().mean() * 100, 2)
                col_info["missing_count"] = missing_count
                col_info["missing_percent"] = missing_percent

                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        non_null_values = df[col].dropna()
                        if len(non_null_values) > 0:
                            desc = df[col].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
                            col_info["numeric_stats"] = {
                                "mean": None if pd.isna(desc.get("mean")) else float(desc.get("mean")),
                                "median": None if pd.isna(desc.get("50%")) else float(desc.get("50%")),
                                "min": None if pd.isna(desc.get("min")) else float(desc.get("min")),
                                "max": None if pd.isna(desc.get("max")) else float(desc.get("max")),
                                "std": None if pd.isna(desc.get("std")) else float(desc.get("std")),
                                "25%": None if pd.isna(desc.get("25%")) else float(desc.get("25%")),
                                "75%": None if pd.isna(desc.get("75%")) else float(desc.get("75%"))
                            }
                            skew_val = df[col].skew()
                            kurt_val = df[col].kurt()
                            col_info["skewness"] = None if pd.isna(skew_val) or math.isnan(skew_val) or math.isinf(skew_val) else float(skew_val)
                            col_info["kurtosis"] = None if pd.isna(kurt_val) or math.isnan(kurt_val) or math.isinf(kurt_val) else float(kurt_val)
                            
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            outlier_count = int(df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0])
                            col_info["outlier_count"] = outlier_count
                            col_info["outlier_bounds"] = {
                                "lower_bound": None if pd.isna(lower_bound) or math.isnan(lower_bound) or math.isinf(lower_bound) else float(lower_bound),
                                "upper_bound": None if pd.isna(upper_bound) or math.isnan(upper_bound) or math.isinf(upper_bound) else float(upper_bound)
                            }
                            # Histogram with "NoData" bin
                            counts, bins = np.histogram(non_null_values, bins=10)
                            missing_bin = int(df[col].isnull().sum())
                            col_info["histogram"] = {
                                "bins": [None if pd.isna(b) or math.isnan(b) or math.isinf(b) else float(b) for b in bins.tolist()],
                                "counts": [int(c) for c in counts.tolist()],
                                "nodata_count": missing_bin
                            }
                        else:
                            col_info["numeric_stats"] = {
                                "mean": None,
                                "median": None,
                                "min": None,
                                "max": None,
                                "std": None,
                                "25%": None,
                                "75%": None
                            }
                            col_info["skewness"] = None
                            col_info["kurtosis"] = None
                            col_info["outlier_count"] = 0
                            col_info["outlier_bounds"] = {"lower_bound": None, "upper_bound": None}
                            col_info["histogram"] = {
                                "bins": [],
                                "counts": [],
                                "nodata_count": missing_count
                            }
                    except Exception as e:
                        logging.warning(f"Error processing numeric column '{col}': {e}")
                        # Still add basic info even if stats fail
                        col_info["numeric_stats"] = None
                        col_info["skewness"] = None
                        col_info["kurtosis"] = None
                
                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    try:
                        # Value counts with NoData bucket
                        value_counts = df[col].value_counts(dropna=False)
                        
                        # Get count of actual missing values (NaN)
                        nan_count = df[col].isna().sum()
                        
                        # Remove string "nan" entries (these are from data conversion, not actual missing values)
                        value_counts_clean = value_counts.copy()
                        string_nan_keys = [k for k in value_counts_clean.index if isinstance(k, str) and k.lower().strip() == 'nan']
                        for key in string_nan_keys:
                            value_counts_clean = value_counts_clean.drop(key)
                        
                        # Rename actual NaN to "NoData"
                        if pd.isna(value_counts_clean.index).any():
                            value_counts_clean = value_counts_clean.rename({np.nan: "NoData"})
                        elif nan_count > 0:
                            # If NaN was dropped but we still have missing, add it as "NoData"
                            value_counts_clean["NoData"] = nan_count
                        
                        vc_dict = value_counts_clean.to_dict()
                        # Final check: if there's still a NaN key, rename it to "NoData"
                        if any(pd.isna(key) for key in vc_dict.keys()):
                            for k in list(vc_dict.keys()):
                                if pd.isna(k):
                                    vc_dict["NoData"] = vc_dict.pop(k)
                        
                        # Convert keys to strings to ensure JSON serialization
                        vc_dict_clean = {}
                        for k, v in vc_dict.items():
                            if pd.isna(k):
                                vc_dict_clean["NoData"] = int(v)
                            else:
                                vc_dict_clean[str(k)] = int(v)
                        
                        col_info["top_categories"] = vc_dict_clean
                    except Exception as e:
                        logging.warning(f"Error processing categorical column '{col}': {e}")
                        # Still add basic info even if value counts fail
                        col_info["top_categories"] = {}
                
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        non_null_dates = df[col].dropna()
                        if len(non_null_dates) > 0:
                            min_date = non_null_dates.min()
                            max_date = non_null_dates.max()
                            col_info["min_date"] = None if pd.isna(min_date) else str(min_date)
                            col_info["max_date"] = None if pd.isna(max_date) else str(max_date)
                            col_series = pd.to_datetime(df[col], errors='coerce')
                            monthly_counts = col_series.dt.to_period('M').value_counts().sort_index().to_dict()
                            if len(monthly_counts) <= 20:
                                col_info["monthly_distribution"] = {str(k): int(v) for k, v in monthly_counts.items()}
                        else:
                            col_info["min_date"] = None
                            col_info["max_date"] = None
                    except Exception as e:
                        logging.info(f"Error generating monthly distribution for column '{col}': {e}")
                        col_info["min_date"] = None
                        col_info["max_date"] = None
                else:
                    # Handle other dtypes (like bool, etc.)
                    logging.info(f"Column '{col}' has unhandled dtype: {df[col].dtype}")
                    # Still add basic info
                    col_info["top_categories"] = {}

                columns_info[col] = col_info
            except Exception as col_error:
                logging.error(f"Error processing column '{col}': {col_error}")
                # Add minimal info for this column
                columns_info[col] = {
                    "dtype": str(df[col].dtype),
                    "missing_count": int(df[col].isnull().sum()),
                    "missing_percent": round(df[col].isnull().mean() * 100, 2),
                    "error": str(col_error)
                }
        
        eda_summary["columns"] = columns_info
        # Convert missing percentages to dict and ensure no NaN values
        missing_percentages = (df.isnull().mean() * 100).round(2)
        eda_summary["missing_data_overall"] = {
            col: None if pd.isna(val) or math.isnan(val) or math.isinf(val) else float(val)
            for col, val in missing_percentages.items()
        }
        duplicate_count = int(df.duplicated().sum())
        eda_summary["duplicate_rows"] = duplicate_count
        eda_summary["duplicate_percentage"] = round(duplicate_count / df.shape[0] * 100, 2) if df.shape[0] > 0 else 0
        
        # Only add correlations if there are numeric columns
        numeric_df = df.select_dtypes(include=["number"])
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            try:
                corr_matrix = numeric_df.corr()
                # Convert correlation matrix to dict and replace NaN with None
                corr_dict = {}
                for col1 in corr_matrix.columns:
                    corr_dict[col1] = {}
                    for col2 in corr_matrix.columns:
                        val = corr_matrix.loc[col1, col2]
                        corr_dict[col1][col2] = None if pd.isna(val) or math.isnan(val) or math.isinf(val) else float(val)
                eda_summary["correlations"] = corr_dict
            except Exception as e:
                logging.warning(f"Error computing correlations: {e}")
                eda_summary["correlations"] = {}
        else:
            eda_summary["correlations"] = {}
        
        # Convert all NaN values to None for JSON serialization
        eda_summary = convert_nan_to_none(eda_summary)
        
        return eda_summary
    except Exception as e:
        logging.error(f"Error during EDA summary: {e}", exc_info=True)
        return None

