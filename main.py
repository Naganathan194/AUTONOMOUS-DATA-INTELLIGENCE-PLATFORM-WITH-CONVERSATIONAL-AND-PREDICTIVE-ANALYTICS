from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import io
import re
import math
from typing import Optional, List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import uuid
import logging

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report,
                              mean_squared_error, r2_score, mean_absolute_error)
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, RandomForestRegressor,
                               GradientBoostingRegressor, ExtraTreesRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
import zipfile
from PIL import Image

# Import your existing modules
from clean_and_EDA_generate import enhanced_eda_json, clean_data, read_and_validate_file
from generate_report import generate_eda_report_ppt
from utils import get_gemini_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DataSet Querying LLM API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
datasets = {}
eda_results = {}

MAX_EDA_SAMPLE_ROWS = 100000
MAX_PRED_SAMPLE_ROWS = 200000
MAX_VIZ_COLUMNS = 30
MAX_PRED_FEATURES = 80

class QueryRequest(BaseModel):
    dataset_id: str
    query: str
    page: Optional[int] = 1
    page_size: Optional[int] = 50

class ChatRequest(BaseModel):
    dataset_id: str
    message: str
    history: Optional[List[dict]] = []

class ExploreRequest(BaseModel):
    dataset_id: str
    filters: Optional[Dict[str, Any]] = {}
    sort_by: Optional[str] = None
    sort_order: Optional[str] = "asc"
    page: Optional[int] = 1
    page_size: Optional[int] = 50

class PredictRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None

def detect_primary_keys(df: pd.DataFrame) -> List[str]:
    """
    Detect all primary key/identifier columns in the dataset.
    Returns a list of column names that are likely primary keys.
    """
    if df is None or df.empty:
        return []
    
    primary_keys = []
    total_rows = len(df)
    
    if total_rows == 0:
        return []
    
    # Common identifier patterns in column names
    identifier_patterns = [
        'id', 'identifier', 'key', 'pk', 'primary_key', 'uuid', 'guid',
        'code', 'number', 'num', 'no', 'ref', 'reference', 'index',
        'email', 'mail', 'username', 'user_name', 'login'
    ]
    
    for col in df.columns:
        try:
            col_lower = str(col).lower()
            unique_count = df[col].nunique(dropna=True)
            unique_ratio = unique_count / total_rows if total_rows > 0 else 0
            null_ratio = df[col].isna().mean()
            
            # Criteria for primary key detection:
            # 1. Very high uniqueness (>95%)
            # 2. Low null ratio (<5%)
            # 3. Column name suggests identifier
            
            is_highly_unique = unique_ratio > 0.95
            has_low_nulls = null_ratio < 0.05
            name_suggests_id = any(pattern in col_lower for pattern in identifier_patterns)
            
            # Primary key if:
            # - Very high uniqueness (>95%) AND low nulls (<5%)
            # OR
            # - High uniqueness (>90%) AND name suggests ID AND low nulls
            if is_highly_unique and has_low_nulls:
                primary_keys.append(col)
                logger.info(f"Detected primary key: '{col}' (uniqueness: {unique_ratio:.2%}, nulls: {null_ratio:.2%})")
            elif unique_ratio > 0.90 and name_suggests_id and has_low_nulls:
                primary_keys.append(col)
                logger.info(f"Detected primary key: '{col}' (uniqueness: {unique_ratio:.2%}, name suggests ID)")
            
        except Exception as e:
            logger.warning(f"Error checking column '{col}' for primary key: {str(e)}")
            continue
    
    if primary_keys:
        logger.info(f"Detected {len(primary_keys)} primary key column(s): {primary_keys}")
    else:
        logger.info("No primary keys detected in dataset")
    
    return primary_keys

def sample_dataframe(df: pd.DataFrame, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state)

def downsample_ordered(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if len(df) <= max_rows:
        return df
    step = max(1, len(df) // max_rows)
    return df.iloc[::step].copy()

def compute_column_groups(df: pd.DataFrame, primary_keys: List[str]) -> Dict[str, List[str]]:
    groups = {
        "identifiers": [],
        "datetime": [],
        "numeric": [],
        "categorical": [],
        "text": []
    }
    if df is None or df.empty:
        return groups

    for col in df.columns:
        if col in primary_keys:
            groups["identifiers"].append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            groups["datetime"].append(col)
            continue

        if is_numeric_column(df, col):
            groups["numeric"].append(col)
            continue

        if is_categorical_column(df, col, primary_keys):
            groups["categorical"].append(col)
        else:
            sample = df[col].dropna().astype(str).head(100)
            avg_len = sample.map(len).mean() if len(sample) > 0 else 0
            if avg_len >= 20:
                groups["text"].append(col)
            else:
                groups["categorical"].append(col)

    return groups

def select_visual_columns(df: pd.DataFrame, primary_keys: List[str], max_columns: int) -> Dict[str, List[str]]:
    numeric_scores = []
    categorical_scores = []

    for col in df.columns:
        if col in primary_keys:
            continue

        missing_ratio = df[col].isna().mean()

        if is_numeric_column(df, col):
            series = pd.to_numeric(df[col], errors='coerce')
            if series.notna().sum() < 3:
                continue
            variance = float(series.var(skipna=True)) if series.notna().sum() > 1 else 0.0
            score = variance * (1.0 - missing_ratio)
            numeric_scores.append((col, score))
        elif is_categorical_column(df, col, primary_keys):
            unique_count = df[col].nunique(dropna=True)
            if unique_count < 2:
                continue
            penalty = math.log(unique_count + 1)
            score = (1.0 - missing_ratio) / max(1.0, penalty)
            categorical_scores.append((col, score))

    numeric_scores.sort(key=lambda x: x[1], reverse=True)
    categorical_scores.sort(key=lambda x: x[1], reverse=True)

    return {
        "numeric": [c for c, _ in numeric_scores[:max_columns]],
        "categorical": [c for c, _ in categorical_scores[:max_columns]]
    }

def detect_candidate_targets(df: pd.DataFrame, primary_keys: List[str]) -> List[str]:
    candidates = []
    for col in df.columns:
        if col in primary_keys:
            continue
        if is_categorical_column(df, col, primary_keys):
            unique_count = df[col].nunique(dropna=True)
            if 2 <= unique_count <= 20:
                candidates.append(col)
    return candidates

def summarize_imbalance(series: pd.Series) -> Dict[str, Any]:
    counts = series.value_counts(dropna=True)
    total = int(counts.sum())
    if total == 0:
        return {"total": 0, "imbalance_ratio": None, "majority_share": None, "minority_share": None}
    majority = float(counts.max() / total)
    minority = float(counts.min() / total) if len(counts) > 1 else 0.0
    imbalance_ratio = float(counts.max() / max(1, counts.min())) if len(counts) > 1 else float('inf')
    return {
        "total": total,
        "classes": {str(k): int(v) for k, v in counts.head(10).to_dict().items()},
        "imbalance_ratio": round(imbalance_ratio, 3),
        "majority_share": round(majority, 3),
        "minority_share": round(minority, 3)
    }

def build_profile_summary(df: pd.DataFrame, df_sample: pd.DataFrame, eda: dict, primary_keys: List[str]) -> Dict[str, Any]:
    columns = eda.get("columns", {}) if eda else {}
    missing_rank = []
    for col, info in columns.items():
        missing_rank.append((col, info.get("missing_percent", 0)))
    missing_rank.sort(key=lambda x: x[1], reverse=True)

    column_groups = compute_column_groups(df_sample, primary_keys)
    candidates = detect_candidate_targets(df_sample, primary_keys)

    imbalance = {}
    for col in candidates[:5]:
        imbalance[col] = summarize_imbalance(df_sample[col].dropna())

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "sampled_rows": len(df_sample),
        "sample_ratio": round(len(df_sample) / max(1, len(df)), 4),
        "primary_keys": primary_keys,
        "top_missing_columns": [
            {"column": c, "missing_percent": round(p, 2)}
            for c, p in missing_rank[:8]
        ],
        "column_groups": {k: v[:30] for k, v in column_groups.items()},
        "candidate_targets": candidates[:10],
        "imbalance_hints": imbalance
    }

def auto_select_target(df: pd.DataFrame, primary_keys: List[str]) -> Optional[str]:
    candidates = detect_candidate_targets(df, primary_keys)
    if not candidates:
        return None
    scored = []
    for col in candidates:
        series = df[col].dropna()
        if series.empty:
            continue
        imbalance = summarize_imbalance(series)
        majority = imbalance.get("majority_share", 1.0)
        unique_count = series.nunique()
        score = (1.0 - majority) + (unique_count / 20.0)
        scored.append((col, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored else candidates[0]

def select_prediction_features(df: pd.DataFrame, target_col: str, primary_keys: List[str]) -> List[str]:
    numeric_scores = []
    categorical_scores = []

    for col in df.columns:
        if col == target_col or col in primary_keys:
            continue
        missing_ratio = df[col].isna().mean()

        if is_numeric_column(df, col):
            series = pd.to_numeric(df[col], errors='coerce')
            if series.notna().sum() < 10:
                continue
            variance = float(series.var(skipna=True)) if series.notna().sum() > 1 else 0.0
            score = variance * (1.0 - missing_ratio)
            numeric_scores.append((col, score))
        elif is_categorical_column(df, col, primary_keys):
            unique_count = df[col].nunique(dropna=True)
            if unique_count < 2 or unique_count > 200:
                continue
            penalty = math.log(unique_count + 1)
            score = (1.0 - missing_ratio) / max(1.0, penalty)
            categorical_scores.append((col, score))

    numeric_scores.sort(key=lambda x: x[1], reverse=True)
    categorical_scores.sort(key=lambda x: x[1], reverse=True)

    numeric_selected = [c for c, _ in numeric_scores[:MAX_PRED_FEATURES // 2]]
    categorical_selected = [c for c, _ in categorical_scores[:MAX_PRED_FEATURES // 2]]

    selected = numeric_selected + categorical_selected
    return selected[:MAX_PRED_FEATURES]

def detect_entity_id(df: pd.DataFrame) -> Optional[str]:
    """Detect identifier column for deduplication (returns first primary key)"""
    primary_keys = detect_primary_keys(df)
    return primary_keys[0] if primary_keys else None

def get_clean_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Get clean series with per-entity deduplication if applicable"""
    try:
        entity_id = detect_entity_id(df)
        if entity_id and entity_id in df.columns and entity_id != col:
            subset = df[[entity_id, col]].copy()
            return subset.groupby(entity_id)[col].apply(
                lambda s: s.dropna().iloc[-1] if not s.dropna().empty else np.nan
            )
        return df[col].copy()
    except Exception as e:
        logger.warning(f"Error in get_clean_series for {col}: {str(e)}")
        return df[col].copy()

def is_numeric_column(df: pd.DataFrame, col: str) -> bool:
    """Determine if column should be treated as numeric"""
    try:
        # First check if already numeric dtype
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_count = df[col].nunique()
            total_count = len(df)
            unique_ratio = unique_count / total_count if total_count > 0 else 0
            
            # Numeric if: more than 10 unique values OR unique ratio > 10%
            return unique_count > 10 or unique_ratio > 0.1
        
        # Try converting to numeric
        test_series = pd.to_numeric(df[col], errors='coerce')
        non_null_converted = test_series.dropna()
        
        # If more than 50% can be converted and has enough unique values
        if len(non_null_converted) > len(df) * 0.5:
            unique_count = non_null_converted.nunique()
            unique_ratio = unique_count / len(non_null_converted) if len(non_null_converted) > 0 else 0
            return unique_count > 10 or unique_ratio > 0.1
        
        return False
    except Exception as e:
        logger.warning(f"Error checking numeric for {col}: {str(e)}")
        return False

def is_sensible_numeric_column(df: pd.DataFrame, col: str, eda_info: dict = None, primary_keys: List[str] = None) -> bool:
    """
    Use LLM to determine if a numeric column is sensible for statistical analysis.
    Returns False for ID columns, mobile numbers, and other non-analyzable numeric columns.
    """
    try:
        # Check if column is in primary keys list (if provided)
        if primary_keys and col in primary_keys:
            logger.info(f"Column '{col}' is a detected primary key - skipping numeric analysis")
            return False
        
        # Quick heuristic checks first (faster than LLM)
        col_lower = col.lower()
        
        # Common ID/identifier patterns
        id_patterns = ['id', 'identifier', 'key', 'code', 'number', 'num', 'no', 'ref', 'reference']
        mobile_patterns = ['mobile', 'phone', 'contact', 'tel', 'cell']
        
        # Check column name patterns
        if any(pattern in col_lower for pattern in id_patterns):
            # Check if it's likely an ID (high uniqueness, sequential, or all unique)
            unique_count = df[col].nunique()
            total_count = len(df)
            unique_ratio = unique_count / total_count if total_count > 0 else 0
            
            # If almost all values are unique, it's likely an ID
            if unique_ratio > 0.95:
                logger.info(f"Column '{col}' identified as ID (unique ratio: {unique_ratio:.2%})")
                return False
        
        # Check for mobile number patterns
        if any(pattern in col_lower for pattern in mobile_patterns):
            # Check if values look like phone numbers (10+ digits, mostly unique)
            sample_values = df[col].dropna().head(20)
            if len(sample_values) > 0:
                # Check if values are long integers (phone numbers are typically 10+ digits)
                numeric_values = pd.to_numeric(sample_values, errors='coerce').dropna()
                if len(numeric_values) > 0:
                    min_val = numeric_values.min()
                    max_val = numeric_values.max()
                    # Phone numbers are typically 10-15 digits
                    if min_val >= 1000000000 and max_val < 1e15:
                        unique_count = df[col].nunique()
                        if unique_count / len(df) > 0.8:  # Mostly unique
                            logger.info(f"Column '{col}' identified as mobile/phone number")
                            return False
        
        # If heuristics don't rule it out, use LLM for final decision
        if eda_info and col in eda_info.get("columns", {}):
            col_info = eda_info["columns"][col]
            
            # Prepare sample data for LLM
            sample_data = df[col].dropna().head(10).tolist()
            unique_count = df[col].nunique()
            total_count = len(df)
            
            prompt = f"""Analyze this numeric column from a dataset and determine if it's sensible for statistical analysis and visualization.

Column Name: {col}
Data Type: {col_info.get('dtype', 'unknown')}
Total Rows: {total_count}
Unique Values: {unique_count}
Sample Values: {sample_data[:10]}

Consider these guidelines:
- ID columns (like Candidate Id, User ID, etc.) should return FALSE
- Mobile/Phone numbers should return FALSE
- Reference numbers, codes, or identifiers should return FALSE
- Measurements, counts, scores, ratings, prices, ages, etc. should return TRUE
- Columns where statistical analysis (mean, median, distribution) makes sense should return TRUE

Respond with ONLY "TRUE" or "FALSE" (all caps, no other text)."""

            try:
                response = get_gemini_response(prompt, "lite")
                result = response.strip().upper()
                
                if "FALSE" in result:
                    logger.info(f"LLM determined column '{col}' is NOT sensible for analysis")
                    return False
                elif "TRUE" in result:
                    logger.info(f"LLM determined column '{col}' IS sensible for analysis")
                    return True
                else:
                    # If LLM response is unclear, default to True (analyze it)
                    logger.warning(f"Unclear LLM response for '{col}': {response}. Defaulting to True.")
                    return True
            except Exception as llm_error:
                logger.warning(f"LLM check failed for '{col}': {llm_error}. Defaulting to True.")
                return True
        
        # Default to True if we can't determine
        return True
        
    except Exception as e:
        logger.warning(f"Error checking if '{col}' is sensible numeric: {str(e)}")
        return True  # Default to analyzing if check fails

def is_categorical_column(df: pd.DataFrame, col: str, primary_keys: List[str] = None) -> bool:
    """Determine if column should be treated as categorical.
    Excludes primary keys and identifiers (columns with very high uniqueness)."""
    try:
        # Check if column is in primary keys list (if provided)
        if primary_keys and col in primary_keys:
            logger.info(f"Column '{col}' is a detected primary key - skipping categorical analysis")
            return False
        
        # Check uniqueness first - if all or almost all values are unique, it's likely a primary key
        unique_count = df[col].nunique()
        total_count = len(df)
        unique_ratio = unique_count / total_count if total_count > 0 else 0
        
        # If more than 95% of values are unique, it's likely a primary key/identifier - skip it
        if unique_ratio > 0.95:
            logger.info(f"Column '{col}' has high uniqueness ({unique_ratio:.2%}), treating as identifier/primary key - skipping categorical analysis")
            return False
        
        # Object/string types
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            # Additional check: if it's an email-like column (contains @) and high uniqueness, skip it
            col_lower = col.lower()
            if ('email' in col_lower or 'mail' in col_lower) and unique_ratio > 0.8:
                logger.info(f"Column '{col}' appears to be email with high uniqueness ({unique_ratio:.2%}), skipping categorical analysis")
                return False
            
            # Additional check: if column name suggests it's an identifier
            identifier_keywords = ['id', 'identifier', 'key', 'code', 'name', 'email', 'mail']
            if any(keyword in col_lower for keyword in identifier_keywords) and unique_ratio > 0.9:
                logger.info(f"Column '{col}' appears to be identifier with high uniqueness ({unique_ratio:.2%}), skipping categorical analysis")
                return False
            
            return True
        
        # Low cardinality numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            # Already checked uniqueness above, so if we get here and it's numeric with low cardinality, it's categorical
            return unique_count <= 20 and unique_ratio < 0.1
        
        return False
    except Exception as e:
        logger.warning(f"Error checking categorical for {col}: {str(e)}")
        return False

def convert_to_json_serializable(obj):
    """Convert pandas Timestamps, NaN values, and other non-JSON-serializable objects to JSON-compatible types"""
    # Handle None
    if obj is None:
        return None
    
    # Handle arrays first before checking for NaN
    if isinstance(obj, np.ndarray):
        return [convert_to_json_serializable(item) for item in obj.tolist()]
    elif isinstance(obj, pd.Series):
        return [convert_to_json_serializable(item) for item in obj.tolist()]
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(key): convert_to_json_serializable(value) for key, value in obj.items()}
    
    # Handle specific pandas/numpy types
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Timedelta):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    
    # Handle numpy scalar types
    if isinstance(obj, np.generic):
        if isinstance(obj, (np.floating, np.complexfloating)):
            try:
                if pd.isna(obj) or math.isnan(obj) or math.isinf(obj):
                    return None
                return float(obj)
            except (ValueError, TypeError, OverflowError):
                return None
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            try:
                if pd.isna(obj):
                    return None
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            except (ValueError, TypeError, OverflowError):
                return str(obj)
    
    # Handle Python float - check for NaN/inf
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    
    # Check for NaN values using pandas (for any remaining types)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    
    # Check for NaN using math (for numeric types)
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

def convert_plotly_figure_to_dict(fig):
    """Convert Plotly figure to dict, ensuring proper JSON serialization"""
    try:
        # Use to_json() and parse it for better compatibility with frontend
        fig_json = fig.to_json()
        fig_dict = json.loads(fig_json)
        # Ensure data is a list and layout is a dict
        if isinstance(fig_dict, dict):
            if 'data' not in fig_dict:
                logger.warning("Figure data missing, setting empty list")
                fig_dict['data'] = []
            elif not isinstance(fig_dict['data'], list):
                logger.warning("Figure data is not a list, converting")
                fig_dict['data'] = [fig_dict['data']] if fig_dict['data'] else []
            else:
                # Ensure all data items have proper structure
                for i, trace in enumerate(fig_dict['data']):
                    if isinstance(trace, dict):
                        # Convert numpy arrays and pandas Series to lists
                        for key, value in trace.items():
                            if hasattr(value, 'tolist'):
                                trace[key] = value.tolist()
                            elif hasattr(value, '__iter__') and not isinstance(value, (str, dict, list)):
                                try:
                                    trace[key] = list(value)
                                except:
                                    pass
            
            if 'layout' not in fig_dict:
                logger.warning("Figure layout missing, setting empty dict")
                fig_dict['layout'] = {}
            elif not isinstance(fig_dict['layout'], dict):
                logger.warning("Figure layout is not a dict, converting")
                fig_dict['layout'] = dict(fig_dict['layout']) if fig_dict['layout'] else {}
            else:
                # Ensure layout values are JSON serializable
                def clean_layout(obj):
                    if isinstance(obj, dict):
                        return {k: clean_layout(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [clean_layout(item) for item in obj]
                    elif hasattr(obj, 'tolist'):
                        return obj.tolist()
                    elif isinstance(obj, (int, float, str, bool, type(None))):
                        return obj
                    else:
                        return str(obj)
                fig_dict['layout'] = clean_layout(fig_dict['layout'])
        return fig_dict
    except Exception as e:
        logger.error(f"Error converting Plotly figure to dict: {str(e)}", exc_info=True)
        # Return minimal valid figure structure
        return {
            "data": [],
            "layout": {"title": {"text": "Error rendering figure"}}
        }

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and process a dataset"""
    try:
        contents = await file.read()
        
        # Read file
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(contents))
            elif file.filename.endswith('.xlsx'):
                df = pd.read_excel(io.BytesIO(contents))
            else:
                raise HTTPException(400, "Unsupported file format. Use CSV or XLSX.")
        except Exception as read_error:
            logger.error(f"Error reading file {file.filename}: {str(read_error)}")
            raise HTTPException(400, f"Failed to read file: {str(read_error)}")
        
        if df.empty:
            raise HTTPException(400, "Uploaded file is empty")
        
        if len(df.columns) == 0:
            raise HTTPException(400, "Uploaded file has no columns")
        
        # Remove completely empty rows before processing
        initial_rows = len(df)
        # Check for rows where all values are NaN or empty strings
        mask = df.apply(lambda row: not (row.isna().all() or (row.astype(str).str.strip().eq('').all() if len(row) > 0 else True)), axis=1)
        df = df[mask]
        empty_rows_removed = initial_rows - len(df)
        if empty_rows_removed > 0:
            logger.info(f"Removed {empty_rows_removed} completely empty rows before cleaning")
        
        # Remove completely empty columns before processing
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
            logger.info(f"Removed {len(empty_cols)} completely empty columns before cleaning: {empty_cols}")
        
        # Check if dataframe is now empty after removing empty rows/columns
        if df.empty:
            raise HTTPException(400, "Dataset is empty after removing empty rows and columns")
        
        if len(df.columns) == 0:
            raise HTTPException(400, "Dataset has no columns after removing empty columns")
        
        logger.info(f"Read dataset: {file.filename} ({len(df)} rows, {len(df.columns)} cols)")
        logger.info(f"Column types: {df.dtypes.to_dict()}")
        
        # Clean data
        logger.info(f"Cleaning dataset: {file.filename}")
        try:
            df = clean_data(df)
        except Exception as clean_error:
            logger.error(f"Error cleaning data: {str(clean_error)}", exc_info=True)
            raise HTTPException(400, f"Data cleaning failed: {str(clean_error)}")
        
        if df is None:
            raise HTTPException(400, "Data cleaning returned None")
        
        if df.empty:
            raise HTTPException(400, "Data cleaning resulted in empty dataset")
        
        if len(df.columns) == 0:
            raise HTTPException(400, "Data cleaning removed all columns")
        
        logger.info(f"After cleaning: {len(df)} rows, {len(df.columns)} cols")
        
        # Sample for faster EDA and visualizations
        df_sample = sample_dataframe(df, MAX_EDA_SAMPLE_ROWS)
        is_sampled = len(df_sample) < len(df)

        # Generate EDA
        logger.info(f"Generating EDA for: {file.filename}")
        try:
            eda = enhanced_eda_json(df_sample)
        except Exception as eda_error:
            logger.error(f"Error generating EDA: {str(eda_error)}", exc_info=True)
            raise HTTPException(500, f"EDA generation failed: {str(eda_error)}")
        
        if eda is None:
            raise HTTPException(500, "EDA generation returned None")
        
        if "columns" not in eda:
            raise HTTPException(500, "EDA generation failed: missing 'columns' key")
        
        if not eda.get("columns") or len(eda["columns"]) == 0:
            raise HTTPException(500, "EDA generation failed: no columns in EDA result")
        
        # Ensure EDA is JSON-serializable by applying conversion
        eda = convert_to_json_serializable(eda)
        
        # Generate unique ID
        dataset_id = str(uuid.uuid4())
        
        # Detect primary keys
        primary_keys = detect_primary_keys(df)

        # Enrich EDA with sampling metadata
        eda["sampled_rows"] = len(df_sample)
        eda["sample_ratio"] = round(len(df_sample) / max(1, len(df)), 4)

        # Build profile summary
        profile = build_profile_summary(df, df_sample, eda, primary_keys)
        
        # Store in memory
        datasets[dataset_id] = {
            "df": df,
            "df_sample": df_sample,
            "filename": file.filename,
            "uploaded_at": datetime.now().isoformat(),
            "primary_keys": primary_keys,
            "is_sampled": is_sampled,
            "profile": profile,
            "analysis_cache": {}
        }
        eda_results[dataset_id] = eda
        
        logger.info(f"Successfully uploaded dataset: {file.filename} ({len(df)} rows, {len(df.columns)} cols)")
        logger.info(f"Column types after processing: {df.dtypes.to_dict()}")
        
        # Build response and ensure it's JSON-serializable
        response_data = {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "primary_keys": primary_keys,
            "eda": eda,
            "profile": profile,
            "is_sampled": is_sampled
        }
        
        # Final check: ensure response is JSON-serializable
        response_data = convert_to_json_serializable(response_data)
        
        return response_data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error processing file: {str(e)}")

@app.get("/api/datasets")
async def list_datasets():
    """List all uploaded datasets"""
    return {
        "datasets": [
            {
                "id": ds_id,
                "filename": info["filename"],
                "rows": len(info["df"]),
                "columns": len(info["df"].columns),
                "uploaded_at": info["uploaded_at"]
            }
            for ds_id, info in datasets.items()
        ]
    }

@app.delete("/api/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    
    del datasets[dataset_id]
    if dataset_id in eda_results:
        del eda_results[dataset_id]
    
    return {"success": True, "message": "Dataset deleted"}

@app.get("/api/dataset/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get dataset information and EDA results"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    
    df = datasets[dataset_id]["df"]
    eda = eda_results[dataset_id]
    primary_keys = datasets[dataset_id].get("primary_keys", [])
    
    # Enhance EDA with is_unique flag for each column
    enhanced_eda = eda.copy()
    if "columns" in enhanced_eda:
        for col_name in enhanced_eda["columns"]:
            if col_name in primary_keys:
                enhanced_eda["columns"][col_name]["is_unique"] = True
            else:
                # Check uniqueness ratio
                unique_count = df[col_name].nunique()
                total_count = len(df)
                unique_ratio = unique_count / total_count if total_count > 0 else 0
                enhanced_eda["columns"][col_name]["is_unique"] = unique_ratio > 0.95
    
    return {
        "dataset_id": dataset_id,
        "filename": datasets[dataset_id]["filename"],
        "rows": len(df),
        "columns": len(df.columns),
        "primary_keys": primary_keys,
        "profile": datasets[dataset_id].get("profile"),
        "is_sampled": datasets[dataset_id].get("is_sampled", False),
        "eda": enhanced_eda,
        "preview": df.head(10).fillna("").to_dict('records')
    }

@app.get("/api/analyze/{dataset_id}/numerical")
async def get_numerical_analysis(dataset_id: str):
    """Get numerical analysis with robust statistics, outlier computation, information-rich Plotly charts"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    cache = datasets[dataset_id].setdefault("analysis_cache", {})
    if "numerical" in cache:
        return cache["numerical"]
    try:
        df = datasets[dataset_id]["df"]
        df_visual = datasets[dataset_id].get("df_sample", df)
        eda = eda_results.get(dataset_id, {})
        primary_keys = datasets[dataset_id].get("primary_keys", [])
        
        # First, get all numeric columns
        all_numeric_cols = [col for col in df_visual.columns if is_numeric_column(df_visual, col)]
        logger.info(f"Numerical: {len(all_numeric_cols)} numeric columns found: {all_numeric_cols}")
        
        # Filter out non-sensible numeric columns (IDs, mobile numbers, etc.)
        sensible_numeric_cols = []
        skipped_cols = []
        
        for col in all_numeric_cols:
            if is_sensible_numeric_column(df_visual, col, eda, primary_keys):
                sensible_numeric_cols.append(col)
            else:
                skipped_cols.append(col)
                logger.info(f"Skipping non-sensible numeric column: {col}")
        
        logger.info(f"After filtering: {len(sensible_numeric_cols)} sensible numeric columns: {sensible_numeric_cols}")
        if skipped_cols:
            logger.info(f"Skipped {len(skipped_cols)} non-sensible columns: {skipped_cols}")
        
        if not sensible_numeric_cols:
            message = "No sensible numeric columns found in dataset"
            if skipped_cols:
                message += f" (skipped {len(skipped_cols)} ID/identifier columns: {', '.join(skipped_cols[:5])})"
            return {
                "type": "numerical", 
                "columns": [], 
                "count": 0, 
                "visualizations": [], 
                "statistics": {}, 
                "skipped_columns": skipped_cols,
                "message": message
            }
        
        selected = select_visual_columns(df_visual, primary_keys, MAX_VIZ_COLUMNS)
        selected_numeric = [c for c in sensible_numeric_cols if c in selected["numeric"]]

        visualizations = []
        statistics = {}
        for col in selected_numeric[:MAX_VIZ_COLUMNS]:
            try:
                series = get_clean_series(df_visual, col)
                series_numeric = pd.to_numeric(series, errors='coerce')
                series_numeric_clean = series_numeric.dropna()
                if len(series_numeric_clean) == 0:
                    logger.warning(f"Column {col} has no valid numeric values after conversion")
                    continue
                stats = {
                    "count": int(len(series_numeric_clean)),
                    "missing_count": int(series_numeric.isna().sum()),
                    "mean": float(series_numeric_clean.mean()),
                    "median": float(series_numeric_clean.median()),
                    "std": float(series_numeric_clean.std()) if len(series_numeric_clean) > 1 else 0.0,
                    "min": float(series_numeric_clean.min()),
                    "max": float(series_numeric_clean.max()),
                    "q25": float(series_numeric_clean.quantile(0.25)),
                    "q75": float(series_numeric_clean.quantile(0.75)),
                    "skewness": float(series_numeric_clean.skew()),
                    "kurtosis": float(series_numeric_clean.kurtosis()),
                }
                q1, q3 = stats["q25"], stats["q75"]
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = series_numeric_clean[(series_numeric_clean < lower) | (series_numeric_clean > upper)]
                stats["outlier_count"] = int(len(outliers))
                stats["outlier_percentage"] = round(len(outliers) / len(series_numeric_clean) * 100, 2) if len(series_numeric_clean) > 0 else 0
                statistics[col] = stats
                # Plot: Histogram/bar (Plotly) - ensure data is converted to list
                is_integer = series_numeric_clean.mod(1).eq(0).all() if len(series_numeric_clean) > 0 else False
                fig = go.Figure()
                if is_integer and series_numeric_clean.nunique() <= 60:
                    counts = series_numeric_clean.astype(int).value_counts().sort_index()
                    x_vals = counts.index.astype(str).tolist()
                    y_vals = counts.values.astype(int).tolist()  # Ensure int conversion
                    if stats["missing_count"] > 0:
                        x_vals.append("No Value")
                        y_vals.append(int(stats["missing_count"]))
                    fig.add_trace(go.Bar(
                        x=x_vals, 
                        y=y_vals, 
                        marker_color='rgb(102,126,234)', 
                        hovertemplate='%{x}: %{y}<extra></extra>'
                    ))
                else:
                    # Convert to list for histogram
                    hist_data = series_numeric_clean.tolist()
                    bins = min(50, max(10, int(np.sqrt(len(hist_data)))))
                    fig.add_trace(go.Histogram(
                        x=hist_data, 
                        nbinsx=bins, 
                        marker_color='rgb(102,126,234)', 
                        hovertemplate='Value: %{x}<br>Count: %{y}<extra></extra>'
                    ))
                # Add mean/median lines
                fig.add_vline(x=stats["mean"], line_color="#fbc531", line_dash="dash", annotation_text="Mean", annotation_position="top")
                fig.add_vline(x=stats["median"], line_color="#e17055", line_dash="dash", annotation_text="Median", annotation_position="top")
                fig.update_layout(
                    title=f"{col} Distribution",
                    xaxis_title=col,
                    yaxis_title="Count",
                    template="plotly_dark",
                    margin=dict(l=30, r=20, t=50, b=30),
                    height=450
                )
                # Ensure figure is properly converted and has data
                fig_dict = convert_plotly_figure_to_dict(fig)
                # Verify data is not empty
                if fig_dict.get('data') and len(fig_dict['data']) > 0:
                    # Double-check that data traces have actual values
                    has_data = False
                    for trace in fig_dict['data']:
                        if isinstance(trace, dict):
                            # Check if trace has x or y values
                            if ('x' in trace and trace['x'] and len(trace['x']) > 0) or \
                               ('y' in trace and trace['y'] and len(trace['y']) > 0):
                                has_data = True
                                break
                    
                    if has_data:
                        visualizations.append({"type": "histogram", "column": col, "figure": fig_dict})
                    else:
                        logger.warning(f"Skipping histogram for {col} - trace has no data values")
                else:
                    logger.warning(f"Skipping histogram for {col} - empty figure data")
                
                # Also add a box plot for outlier visualization
                fig_box = go.Figure()
                # Convert pandas Series to list for Plotly
                box_data = series_numeric_clean.tolist()
                fig_box.add_trace(go.Box(
                    y=box_data,
                    name=col,
                    marker_color='rgb(102,126,234)',
                    boxmean='sd',
                    hovertemplate='<b>%{y}</b><extra></extra>'
                ))
                fig_box.update_layout(
                    title=f"{col} Box Plot (Outliers)",
                    yaxis_title=col,
                    template="plotly_dark",
                    margin=dict(l=30, r=20, t=50, b=30),
                    height=400,
                    showlegend=False
                )
                # Ensure box plot figure is properly converted and has data
                box_fig_dict = convert_plotly_figure_to_dict(fig_box)
                if box_fig_dict.get('data') and len(box_fig_dict['data']) > 0:
                    has_box_data = False
                    for trace in box_fig_dict['data']:
                        if isinstance(trace, dict) and ('y' in trace and trace['y'] and len(trace['y']) > 0):
                            has_box_data = True
                            break
                    
                    if has_box_data:
                        visualizations.append({"type": "box", "column": col, "figure": box_fig_dict})
                    else:
                        logger.warning(f"Skipping box plot for {col} - no data values")
                else:
                    logger.warning(f"Skipping box plot for {col} - empty figure data")
                
            except Exception as e:
                logger.warning(f"Numerical analysis skipped column {col}: {str(e)}")
        
        result = {
            "type": "numerical", 
            "columns": sensible_numeric_cols, 
            "count": len(sensible_numeric_cols), 
            "visualizations": visualizations, 
            "statistics": statistics,
            "skipped_columns": skipped_cols if skipped_cols else None,
            "total_numeric_columns": len(all_numeric_cols),
            "selected_columns": selected_numeric
        }
        cache["numerical"] = result
        return result
    except Exception as e:
        logger.error(f"Numerical analysis error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error generating numerical analysis: {str(e)}")

@app.get("/api/analyze/{dataset_id}/categorical")
async def get_categorical_analysis(dataset_id: str):
    """Get categorical analysis with visualizations"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    cache = datasets[dataset_id].setdefault("analysis_cache", {})
    if "categorical" in cache:
        return cache["categorical"]
    
    try:
        df = datasets[dataset_id]["df"]
        df_visual = datasets[dataset_id].get("df_sample", df)
        primary_keys = datasets[dataset_id].get("primary_keys", [])
        
        # Get all potential categorical columns and track skipped identifiers
        all_potential_categorical = []
        skipped_identifiers = []
        
        for col in df_visual.columns:
            unique_count = df_visual[col].nunique()
            total_count = len(df_visual)
            unique_ratio = unique_count / total_count if total_count > 0 else 0
            
            # Check if it's categorical (this function now filters out high-uniqueness columns and primary keys)
            if is_categorical_column(df_visual, col, primary_keys):
                all_potential_categorical.append(col)
            elif col in primary_keys or unique_ratio > 0.95:
                # Track skipped identifier columns
                reason = "Detected primary key" if col in primary_keys else "High uniqueness - likely identifier/primary key"
                skipped_identifiers.append({
                    "column": col,
                    "uniqueness": round(unique_ratio * 100, 2),
                    "reason": reason
                })
        
        categorical_cols = all_potential_categorical
        
        logger.info(f"Found {len(categorical_cols)} categorical columns: {categorical_cols}")
        if skipped_identifiers:
            logger.info(f"Skipped {len(skipped_identifiers)} identifier columns: {[s['column'] for s in skipped_identifiers]}")
        
        if not categorical_cols:
            message = "No categorical columns found in dataset"
            if skipped_identifiers:
                skipped_names = [s['column'] for s in skipped_identifiers[:5]]
                message += f" (skipped {len(skipped_identifiers)} identifier/primary key columns: {', '.join(skipped_names)})"
            
            return {
                "type": "categorical",
                "columns": [],
                "count": 0,
                "visualizations": [],
                "skipped_identifiers": skipped_identifiers if skipped_identifiers else None,
                "message": message
            }
        
        selected = select_visual_columns(df_visual, primary_keys, MAX_VIZ_COLUMNS)
        selected_categorical = [c for c in categorical_cols if c in selected["categorical"]]

        visualizations = []
        
        for col in selected_categorical[:MAX_VIZ_COLUMNS]:
            try:
                series = get_clean_series(df_visual, col)
                
                # Count missing
                missing_count = int(series.isna().sum() + series.astype(str).str.strip().eq("").sum())
                
                # Get value counts
                series_clean = series.astype(str).replace("", np.nan).dropna()
                if len(series_clean) == 0:
                    logger.warning(f"Column {col} has no valid values")
                    continue
                
                value_counts = series_clean.value_counts()
                # Filter out "nan" strings (these are actually missing values converted to strings)
                value_counts = value_counts[value_counts.index.astype(str).str.lower() != 'nan']
                total_unique = len(value_counts)
                top_30 = value_counts.head(30)
                
                # Bar chart
                x_vals = top_30.index.tolist()
                y_vals = top_30.values.tolist()
                
                if missing_count > 0:
                    x_vals.append("No Value")
                    y_vals.append(missing_count)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=x_vals,
                    y=y_vals,
                    marker_color='rgb(102, 126, 234)',
                    text=[f'{v:,}' for v in y_vals],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>'
                ))
                
                title = f"{col} Value Counts"
                if total_unique > 30:
                    title += f" (Top 30 of {total_unique} unique values)"
                
                fig.update_layout(
                    title=title,
                    template="plotly_dark",
                    xaxis_title=col,
                    yaxis_title="Count",
                    showlegend=False,
                    xaxis=dict(tickangle=-45 if len(x_vals) > 5 else 0),
                    height=500
                )
                
                visualizations.append({
                    "column": col,
                    "type": "bar",
                    "figure": convert_plotly_figure_to_dict(fig)
                })
                
                # Pie chart for small categories
                # Use x_vals and y_vals which already include "No Value" if present
                if len(x_vals) <= 15 and len(x_vals) > 1:
                    fig_pie = go.Figure()
                    fig_pie.add_trace(go.Pie(
                        labels=x_vals,
                        values=y_vals,
                        hole=0.4,
                        textinfo='label+percent',
                        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>'
                    ))
                    
                    fig_pie.update_layout(
                        title=f"{col} Distribution",
                        template="plotly_dark",
                        height=450
                    )
                    
                    visualizations.append({
                        "column": col,
                        "type": "pie",
                        "figure": convert_plotly_figure_to_dict(fig_pie)
                    })
                
                logger.info(f"Successfully processed categorical column: {col}")
                
            except Exception as col_error:
                logger.error(f"Error processing column {col}: {str(col_error)}", exc_info=True)
                continue
        
        result = {
            "type": "categorical",
            "columns": categorical_cols,
            "count": len(categorical_cols),
            "visualizations": visualizations,
            "skipped_identifiers": skipped_identifiers if skipped_identifiers else None,
            "selected_columns": selected_categorical
        }
        cache["categorical"] = result
        return result
    
    except Exception as e:
        logger.error(f"Categorical analysis error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error generating categorical analysis: {str(e)}")

@app.get("/api/analyze/{dataset_id}/correlations")
async def get_correlation_analysis(dataset_id: str):
    """Get correlation analysis with heatmap. Handles NaNs and zero-variance columns. Only shows strong/high correlations."""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    cache = datasets[dataset_id].setdefault("analysis_cache", {})
    if "correlations" in cache:
        return cache["correlations"]
    try:
        df = datasets[dataset_id]["df"]
        df_visual = datasets[dataset_id].get("df_sample", df)
        eda = eda_results.get(dataset_id, {})
        primary_keys = datasets[dataset_id].get("primary_keys", [])
        
        # Get all numeric columns and filter out non-sensible ones
        all_numeric_cols = [col for col in df_visual.columns if is_numeric_column(df_visual, col)]
        numeric_cols = [col for col in all_numeric_cols if is_sensible_numeric_column(df_visual, col, eda, primary_keys)]
        logger.info(f"Correlation: {len(numeric_cols)} sensible numeric columns found (out of {len(all_numeric_cols)} total): {numeric_cols}")
        if len(numeric_cols) < 2:
            return {"type": "correlations", "error": "Not enough numeric columns (at least 2 required)", "numeric_columns_found": len(numeric_cols), "columns": numeric_cols, "strong_correlations": [], "visualizations": []}
        numeric_df = df_visual[numeric_cols].apply(pd.to_numeric, errors='coerce')
        # Drop constant and mostly NaN columns
        usable_cols = [col for col in numeric_cols if numeric_df[col].nunique(dropna=True) > 1 and numeric_df[col].notna().sum() > 2]
        numeric_df = numeric_df[usable_cols]
        if numeric_df.shape[1] < 2:
            return {"type": "correlations", "error": "Not enough valid numeric columns after cleaning", "numeric_columns_found": numeric_df.shape[1], "strong_correlations": [], "visualizations": []}
        corr_matrix = numeric_df.corr().fillna(0)
        # Plotly heatmap - convert numpy array to list properly
        z_values = corr_matrix.values
        if hasattr(z_values, 'tolist'):
            z_values = z_values.tolist()
        else:
            z_values = [[float(z_values[i, j]) for j in range(len(corr_matrix.columns))] for i in range(len(corr_matrix.columns))]
        
        text_values = [[float(x) for x in row] for row in z_values]  # Ensure float conversion
        
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=z_values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=text_values,
            texttemplate='%{text:.2f}',
            textfont={"size": min(12, max(8, 400 // len(corr_matrix.columns)))},
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
            colorbar=dict(title="Correlation")
        ))
        size = max(600, min(1000, len(corr_matrix.columns) * 80))
        fig.update_layout(
            title=f"Correlation Heatmap ({len(corr_matrix.columns)} variables)",
            template="plotly_dark",
            width=size,
            height=size,
            xaxis=dict(side="bottom", tickangle=-45),
            yaxis=dict(autorange="reversed")
        )
        # Find strong correlations (>|0.5|, omit self and trivial pairs)
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if not pd.isna(corr_val) and abs(corr_val) >= 0.5:
                    strong_corr.append({"col1": corr_matrix.columns[i], "col2": corr_matrix.columns[j], "correlation": round(float(corr_val), 3), "strength": "Strong Positive" if corr_val > 0 else "Strong Negative"})
        strong_corr.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        logger.info(f"Correlation: {len(strong_corr)} strong pairs")
        result = {"type": "correlations", "columns": corr_matrix.columns.tolist(), "strong_correlations": strong_corr, "visualizations": [{"type": "heatmap", "figure": convert_plotly_figure_to_dict(fig)}]}
        cache["correlations"] = result
        return result
    except Exception as e:
        logger.error(f"Correlation analysis error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error generating correlation analysis: {str(e)}")

@app.get("/api/analyze/{dataset_id}/outliers")
async def get_outliers_analysis(dataset_id: str):
    """Get comprehensive outliers analysis with IQR, Z-score, and visualization"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    cache = datasets[dataset_id].setdefault("analysis_cache", {})
    if "outliers" in cache:
        return cache["outliers"]
    try:
        df = datasets[dataset_id]["df"]
        df_visual = datasets[dataset_id].get("df_sample", df)
        eda = eda_results.get(dataset_id, {})
        primary_keys = datasets[dataset_id].get("primary_keys", [])
        
        # Get all numeric columns and filter out non-sensible ones (including primary keys)
        all_numeric_cols = [col for col in df_visual.columns if is_numeric_column(df_visual, col)]
        numeric_cols = [col for col in all_numeric_cols if is_sensible_numeric_column(df_visual, col, eda, primary_keys)]
        logger.info(f"Outliers: {len(numeric_cols)} sensible numeric columns found (out of {len(all_numeric_cols)} total): {numeric_cols}")
        
        if not numeric_cols:
            return {
                "type": "outliers",
                "columns": [],
                "count": 0,
                "visualizations": [],
                "statistics": {},
                "message": "No numeric columns found for outlier analysis"
            }
        
        visualizations = []
        statistics = {}
        outlier_details = {}
        
        for col in numeric_cols[:10]:
            try:
                series = get_clean_series(df_visual, col)
                series_numeric = pd.to_numeric(series, errors='coerce')
                series_numeric_clean = series_numeric.dropna()
                
                if len(series_numeric_clean) < 4:
                    continue
                
                # IQR Method
                q1 = series_numeric_clean.quantile(0.25)
                q3 = series_numeric_clean.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                iqr_outliers = series_numeric_clean[(series_numeric_clean < lower_bound) | (series_numeric_clean > upper_bound)]
                
                # Z-score Method
                mean_val = series_numeric_clean.mean()
                std_val = series_numeric_clean.std()
                if std_val > 0:
                    z_scores = np.abs((series_numeric_clean - mean_val) / std_val)
                    z_outliers = series_numeric_clean[z_scores > 3]
                else:
                    z_outliers = pd.Series(dtype=float)
                
                stats = {
                    "total_values": int(len(series_numeric_clean)),
                    "iqr_outliers_count": int(len(iqr_outliers)),
                    "iqr_outliers_percentage": round(len(iqr_outliers) / len(series_numeric_clean) * 100, 2),
                    "zscore_outliers_count": int(len(z_outliers)),
                    "zscore_outliers_percentage": round(len(z_outliers) / len(series_numeric_clean) * 100, 2),
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "min": float(series_numeric_clean.min()),
                    "max": float(series_numeric_clean.max()),
                    "z_score_threshold": 3.0
                }
                statistics[col] = stats
                
                # Store outlier values
                outlier_details[col] = {
                    "iqr_outliers": iqr_outliers.tolist()[:20],  # Top 20
                    "zscore_outliers": z_outliers.tolist()[:20]
                }
                
                # Box plot with outliers highlighted
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=series_numeric_clean,
                    name=col,
                    boxmean='sd',
                    marker_color='rgb(102,126,234)',
                    hovertemplate='<b>%{y}</b><extra></extra>'
                ))
                
                # Highlight outliers
                if len(iqr_outliers) > 0:
                    fig.add_trace(go.Scatter(
                        y=iqr_outliers,
                        x=[col] * len(iqr_outliers),
                        mode='markers',
                        marker=dict(color='red', size=8, symbol='x'),
                        name='IQR Outliers',
                        hovertemplate='Outlier: %{y}<extra></extra>'
                    ))
                
                fig.update_layout(
                    title=f"{col} - Outlier Detection",
                    yaxis_title=col,
                    template="plotly_dark",
                    margin=dict(l=30, r=20, t=50, b=30),
                    height=400
                )
                visualizations.append({"type": "box", "column": col, "figure": convert_plotly_figure_to_dict(fig)})
                
            except Exception as e:
                logger.warning(f"Outlier analysis skipped column {col}: {str(e)}")
                continue
        
        result = {
            "type": "outliers",
            "columns": numeric_cols,
            "count": len(numeric_cols),
            "visualizations": visualizations,
            "statistics": statistics,
            "outlier_details": outlier_details
        }
        cache["outliers"] = result
        return result
    
    except Exception as e:
        logger.error(f"Outliers analysis error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error generating outliers analysis: {str(e)}")

@app.get("/api/analyze/{dataset_id}/timeseries")
async def get_timeseries_analysis(dataset_id: str):
    """Get time series analysis with trend detection, seasonality, and forecasting"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    cache = datasets[dataset_id].setdefault("analysis_cache", {})
    if "timeseries" in cache:
        return cache["timeseries"]
    try:
        df = datasets[dataset_id]["df"]
        df_visual = downsample_ordered(df, MAX_EDA_SAMPLE_ROWS)
        
        # Try to detect date/time columns - be strict: only accept actual datetime columns
        date_cols = []
        for col in df_visual.columns:
            # Skip if column is already numeric (not a date)
            if pd.api.types.is_numeric_dtype(df_visual[col]):
                continue
            
            # Try strict datetime conversion
            try:
                # Attempt to convert entire column to datetime
                test_series = pd.to_datetime(df_visual[col], errors='raise', format='mixed')
                
                # Check if conversion was successful and has valid datetime values
                valid_count = test_series.notna().sum()
                if valid_count > len(df_visual) * 0.8:  # At least 80% valid datetime values
                    # Check if it's actually a datetime type or datetime64
                    if pd.api.types.is_datetime64_any_dtype(test_series) or isinstance(test_series.dtype, pd.DatetimeTZDtype):
                        date_cols.append(col)
                        logger.info(f"Detected datetime column: {col}")
                    # Also check if the values are actually dates (not just strings that look like dates)
                    elif valid_count == len(df_visual):
                        # Double check by verifying it's not just sequential numbers
                        try:
                            # If we can convert to datetime without errors, it's likely a date column
                            sample_dates = test_series.dropna().head(10)
                            if len(sample_dates) > 0:
                                # Check if dates are in reasonable range
                                min_date = sample_dates.min()
                                max_date = sample_dates.max()
                                if min_date.year >= 1900 and max_date.year <= 2100:
                                    date_cols.append(col)
                                    logger.info(f"Detected date column by validation: {col}")
                        except:
                            pass
            except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
                # Column cannot be converted to datetime - skip it
                continue
        
        eda = eda_results.get(dataset_id, {})
        primary_keys = datasets[dataset_id].get("primary_keys", [])
        
        # Get all numeric columns and filter out non-sensible ones (including primary keys)
        all_numeric_cols = [col for col in df_visual.columns if is_numeric_column(df_visual, col)]
        numeric_cols = [col for col in all_numeric_cols if is_sensible_numeric_column(df_visual, col, eda, primary_keys)]
        
        if not date_cols:
            result = {
                "type": "timeseries",
                "error": "No time series detected. Dataset does not contain a valid date/time column or timestamp.",
                "date_columns_found": 0,
                "numeric_columns_found": len(numeric_cols),
                "visualizations": [],
                "message": "No time series detected"
            }
            cache["timeseries"] = result
            return result
        
        if not numeric_cols:
            result = {
                "type": "timeseries",
                "error": "No numeric columns found for time series analysis",
                "date_columns_found": len(date_cols),
                "numeric_columns_found": 0,
                "visualizations": [],
                "message": "No time series detected"
            }
            cache["timeseries"] = result
            return result
        
        visualizations = []
        analyses = {}
        
        # Analyze each numeric column with each date column
        valid_combinations = 0
        for date_col in date_cols[:2]:  # Limit to 2 date columns
            for num_col in numeric_cols[:5]:  # Limit to 5 numeric columns
                try:
                    # Prepare time series data
                    ts_df = df_visual[[date_col, num_col]].copy()
                    ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors='coerce')
                    ts_df[num_col] = pd.to_numeric(ts_df[num_col], errors='coerce')
                    ts_df = ts_df.dropna()
                    
                    # Check if we have valid datetime and numeric values
                    if len(ts_df) < 10:
                        continue
                    
                    # Verify dates are sequential/meaningful (not all same date)
                    unique_dates = ts_df[date_col].nunique()
                    if unique_dates < 3:
                        logger.warning(f"Skipping {date_col} x {num_col}: too few unique dates ({unique_dates})")
                        continue
                    
                    ts_df = ts_df.sort_values(date_col)
                    valid_combinations += 1
                    
                    # Time series plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ts_df[date_col],
                        y=ts_df[num_col],
                        mode='lines+markers',
                        name=num_col,
                        line=dict(color='rgb(102,126,234)', width=2),
                        marker=dict(size=4),
                        hovertemplate='Date: %{x}<br>Value: %{y}<extra></extra>'
                    ))
                    
                    # Add trend line (simple moving average)
                    if len(ts_df) > 7:
                        window = min(7, len(ts_df) // 3)
                        ts_df['trend'] = ts_df[num_col].rolling(window=window, center=True).mean()
                        fig.add_trace(go.Scatter(
                            x=ts_df[date_col],
                            y=ts_df['trend'],
                            mode='lines',
                            name='Trend (MA)',
                            line=dict(color='rgb(255,193,7)', width=2, dash='dash'),
                            hovertemplate='Trend: %{y}<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        title=f"{num_col} over Time ({date_col})",
                        xaxis_title=date_col,
                        yaxis_title=num_col,
                        template="plotly_dark",
                        hovermode='x unified',
                        height=500
                    )
                    
                    unique_id = f"{date_col}_{num_col}"
                    visualizations.append({
                        "type": "timeseries",
                        "date_column": date_col,
                        "value_column": num_col,
                        "figure": convert_plotly_figure_to_dict(fig),
                        "unique_id": unique_id
                    })
                    
                    # Basic statistics
                    analyses[unique_id] = {
                        "date_column": date_col,
                        "value_column": num_col,
                        "data_points": len(ts_df),
                        "start_date": str(ts_df[date_col].min()),
                        "end_date": str(ts_df[date_col].max()),
                        "mean": float(ts_df[num_col].mean()),
                        "std": float(ts_df[num_col].std()),
                        "trend_direction": "increasing" if ts_df[num_col].iloc[-1] > ts_df[num_col].iloc[0] else "decreasing"
                    }
                    
                except Exception as e:
                    logger.warning(f"Time series analysis skipped {date_col} x {num_col}: {str(e)}")
                    continue
        
        # If no valid time series combinations found
        if valid_combinations == 0 or len(visualizations) == 0:
            result = {
                "type": "timeseries",
                "error": "No time series detected. Dataset does not contain valid sequential date/time data.",
                "date_columns_found": len(date_cols),
                "numeric_columns_found": len(numeric_cols),
                "visualizations": [],
                "message": "No time series detected"
            }
            cache["timeseries"] = result
            return result
        
        result = {
            "type": "timeseries",
            "date_columns": date_cols,
            "numeric_columns": numeric_cols,
            "visualizations": visualizations,
            "analyses": analyses
        }
        cache["timeseries"] = result
        return result
    
    except Exception as e:
        logger.error(f"Time series analysis error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error generating time series analysis: {str(e)}")

@app.get("/api/analyze/{dataset_id}/contour")
async def get_contour_analysis(dataset_id: str):
    """Get contour box plots for numeric column pairs"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    cache = datasets[dataset_id].setdefault("analysis_cache", {})
    if "contour" in cache:
        return cache["contour"]
    try:
        df = datasets[dataset_id]["df"]
        df_visual = datasets[dataset_id].get("df_sample", df)
        eda = eda_results.get(dataset_id, {})
        primary_keys = datasets[dataset_id].get("primary_keys", [])
        
        # Get all numeric columns and filter out non-sensible ones (including primary keys)
        all_numeric_cols = [col for col in df_visual.columns if is_numeric_column(df_visual, col)]
        numeric_cols = [col for col in all_numeric_cols if is_sensible_numeric_column(df_visual, col, eda, primary_keys)]
        
        if len(numeric_cols) < 2:
            result = {
                "type": "contour",
                "error": "Need at least 2 numeric columns for contour plots",
                "numeric_columns_found": len(numeric_cols),
                "visualizations": []
            }
            cache["contour"] = result
            return result
        
        visualizations = []
        
        # Create contour plots for pairs of numeric columns
        for i, col1 in enumerate(numeric_cols[:5]):
            for col2 in numeric_cols[i+1:6]:  # Limit pairs
                try:
                    data_df = df_visual[[col1, col2]].copy()
                    data_df[col1] = pd.to_numeric(data_df[col1], errors='coerce')
                    data_df[col2] = pd.to_numeric(data_df[col2], errors='coerce')
                    data_df = data_df.dropna()
                    
                    if len(data_df) < 10:
                        continue
                    
                    # Contour plot (density)
                    fig = go.Figure()
                    
                    # Create 2D histogram for contour
                    hist, xedges, yedges = np.histogram2d(
                        data_df[col1].values,
                        data_df[col2].values,
                        bins=20
                    )
                    
                    fig.add_trace(go.Contour(
                        z=hist.T,
                        x=xedges[:-1],
                        y=yedges[:-1],
                        colorscale='Viridis',
                        contours=dict(showlabels=True),
                        hovertemplate=f'{col1}: %{{x}}<br>{col2}: %{{y}}<br>Density: %{{z}}<extra></extra>'
                    ))
                    
                    # Add scatter overlay
                    fig.add_trace(go.Scatter(
                        x=data_df[col1],
                        y=data_df[col2],
                        mode='markers',
                        marker=dict(color='rgba(255,255,255,0.3)', size=3),
                        name='Data Points',
                        hovertemplate=f'{col1}: %{{x}}<br>{col2}: %{{y}}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=f"Contour Plot: {col1} vs {col2}",
                        xaxis_title=col1,
                        yaxis_title=col2,
                        template="plotly_dark",
                        height=500
                    )
                    
                    visualizations.append({
                        "type": "contour",
                        "column1": col1,
                        "column2": col2,
                        "figure": convert_plotly_figure_to_dict(fig)
                    })
                    
                except Exception as e:
                    logger.warning(f"Contour plot skipped {col1} x {col2}: {str(e)}")
                    continue
        
        result = {
            "type": "contour",
            "columns": numeric_cols,
            "visualizations": visualizations
        }
        cache["contour"] = result
        return result
    
    except Exception as e:
        logger.error(f"Contour analysis error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error generating contour analysis: {str(e)}")

@app.post("/api/explore")
async def explore_dataset(request: ExploreRequest):
    """Enhanced data exploration with filtering, sorting, and pagination"""
    if request.dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    
    try:
        df = datasets[request.dataset_id]["df"].copy()
        primary_keys = datasets[request.dataset_id].get("primary_keys", [])
        
        # Apply filters
        if request.filters:
            for col, filter_value in request.filters.items():
                if col not in df.columns:
                    continue
                
                if isinstance(filter_value, dict):
                    # Range filter for numeric columns
                    if 'min' in filter_value and pd.notna(filter_value['min']):
                        df = df[pd.to_numeric(df[col], errors='coerce') >= float(filter_value['min'])]
                    if 'max' in filter_value and pd.notna(filter_value['max']):
                        df = df[pd.to_numeric(df[col], errors='coerce') <= float(filter_value['max'])]
                elif isinstance(filter_value, list):
                    # Multiple value filter for categorical
                    if filter_value:
                        df = df[df[col].isin(filter_value)]
                else:
                    # Single value filter
                    df = df[df[col] == filter_value]
        
        # Apply sorting
        if request.sort_by and request.sort_by in df.columns:
            ascending = request.sort_order.lower() == 'asc'
            df = df.sort_values(by=request.sort_by, ascending=ascending)
        
        # Get total count after filtering
        total_rows = len(df)
        
        # Apply pagination
        page = max(1, request.page)
        page_size = min(500, max(10, request.page_size))
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        df_page = df.iloc[start_idx:end_idx]
        
        # Get column info
        column_info = []
        for col in df.columns:
            col_data = {
                "name": col,
                "dtype": str(df[col].dtype),
                "is_numeric": is_numeric_column(df, col),
                "is_categorical": is_categorical_column(df, col, primary_keys),
                "unique_count": int(df[col].nunique()),
                "null_count": int(df[col].isna().sum())
            }
            
            # Add value range for numeric columns
            if col_data["is_numeric"]:
                numeric_series = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(numeric_series) > 0:
                    col_data["min"] = float(numeric_series.min())
                    col_data["max"] = float(numeric_series.max())
            
            # Add top values for categorical columns
            if col_data["is_categorical"] and col_data["unique_count"] <= 50:
                top_values = df[col].value_counts().head(20).to_dict()
                col_data["top_values"] = {str(k): int(v) for k, v in top_values.items()}
            
            column_info.append(col_data)
        
        return {
            "success": True,
            "data": df_page.fillna("").to_dict('records'),
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_rows": total_rows,
                "total_pages": (total_rows + page_size - 1) // page_size,
                "has_next": end_idx < total_rows,
                "has_prev": page > 1
            },
            "columns": column_info,
            "filters_applied": len(request.filters) if request.filters else 0,
            "sort_applied": request.sort_by is not None
        }
    
    except Exception as e:
        logger.error(f"Explore error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Data exploration failed: {str(e)}")

@app.get("/api/insights/{dataset_id}")
async def generate_insights(dataset_id: str):
    """Generate AI insights"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    
    try:
        df = datasets[dataset_id]["df"]
        eda = eda_results[dataset_id]
        filename = datasets[dataset_id]["filename"]
        
        prompt = f"""Analyze this dataset and provide insights in a structured format.

Dataset: {filename}
Rows: {len(df):,}, Columns: {len(df.columns)}

EDA Summary:
{json.dumps(eda, indent=2)}

Provide insights in this EXACT format:

SECTION: Data Overview
- Key finding 1
- Key finding 2

SECTION: Data Quality
- Quality insight 1
- Quality insight 2

SECTION: Key Patterns
- Pattern 1
- Pattern 2

SECTION: Notable Findings
- Finding 1
- Finding 2

SECTION: Recommendations
- Recommendation 1
- Recommendation 2

Keep each point concise and actionable."""
        
        raw_insights = get_gemini_response(prompt, "flash")
        sections = parse_insights_into_sections(raw_insights)
        
        return {
            "insights": sections,
            "raw": raw_insights
        }
    
    except Exception as e:
        logger.error(f"Insights error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Insight generation failed: {str(e)}")

def detect_task_type(series: pd.Series) -> str:
    """Detect if the target column is regression or classification."""
    if pd.api.types.is_float_dtype(series):
        unique_count = series.nunique()
        if unique_count > 20:
            return "regression"
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_numeric_dtype(series):
        unique_count = series.nunique()
        total = len(series)
        if unique_count > 30 and (unique_count / total) > 0.1:
            return "regression"
    return "classification"


def compute_data_health_score(df: pd.DataFrame, primary_keys: list) -> dict:
    """Compute a composite data health score (0-100) for a dataframe."""
    try:
        scores = {}
        total_cells = df.shape[0] * df.shape[1]
        if total_cells == 0:
            return {"score": 0, "breakdown": {}}

        # 1. Completeness (no missing values → 100)
        missing_ratio = df.isna().sum().sum() / total_cells
        completeness = max(0.0, 1.0 - missing_ratio) * 100
        scores["completeness"] = round(completeness, 1)

        # 2. Uniqueness (no duplicate rows → 100)
        dup_ratio = df.duplicated().sum() / max(1, len(df))
        uniqueness = max(0.0, 1.0 - dup_ratio) * 100
        scores["uniqueness"] = round(uniqueness, 1)

        # 3. Consistency (numeric skewness penalty)
        skew_penalties = []
        for col in df.columns:
            if col in primary_keys:
                continue
            try:
                if is_numeric_column(df, col):
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(s) > 5:
                        sk = abs(float(s.skew()))
                        # Low skew = good, high skew = penalty
                        penalty = min(1.0, sk / 10.0)
                        skew_penalties.append(1.0 - penalty)
            except Exception:
                pass
        consistency = (np.mean(skew_penalties) * 100) if skew_penalties else 100.0
        scores["consistency"] = round(float(consistency), 1)

        # 4. Validity (columns with >50% missing are penalised)
        high_missing_cols = sum(1 for col in df.columns if df[col].isna().mean() > 0.5)
        validity = max(0.0, 1.0 - high_missing_cols / max(1, len(df.columns))) * 100
        scores["validity"] = round(validity, 1)

        # Overall weighted score
        overall = (
            completeness * 0.40 +
            uniqueness * 0.25 +
            consistency * 0.20 +
            validity * 0.15
        )
        scores["overall"] = round(overall, 1)

        # Grade
        if overall >= 90:
            grade = "Excellent"
        elif overall >= 75:
            grade = "Good"
        elif overall >= 60:
            grade = "Fair"
        else:
            grade = "Poor"
        scores["grade"] = grade

        return scores
    except Exception as e:
        logger.warning(f"Health score computation failed: {e}")
        return {"score": 0, "breakdown": {}, "grade": "Unknown"}


def _build_preprocessor(numeric_features, categorical_features):
    """Build sklearn ColumnTransformer for mixed feature sets."""
    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
        ]), categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _get_feature_importances(pipeline, feature_names, top_n=25):
    """Extract feature importances from the best model pipeline."""
    try:
        model = pipeline.named_steps["model"]
        # Tree-based models have feature_importances_
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        # Linear models have coef_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importance = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            return []

        top_idx = np.argsort(importance)[-top_n:][::-1]
        results = []
        for idx in top_idx:
            if idx < len(feature_names) and float(importance[idx]) > 0:
                results.append({
                    "feature": str(feature_names[idx]),
                    "importance": round(float(importance[idx]), 6)
                })
        return results
    except Exception as e:
        logger.warning(f"Feature importance extraction failed: {e}")
        return []


@app.post("/api/predictive")
async def run_predictive_analysis(request: PredictRequest):
    """AutoML predictive analysis — runs multiple models, compares, and picks the best."""
    if request.dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")

    try:
        df_full = datasets[request.dataset_id]["df"]
        primary_keys = datasets[request.dataset_id].get("primary_keys", [])

        df_sample = sample_dataframe(df_full, MAX_PRED_SAMPLE_ROWS)
        target_column = request.target_column or auto_select_target(df_sample, primary_keys)

        if not target_column or target_column not in df_full.columns:
            raise HTTPException(400, "No suitable target column found. Please specify a target column.")

        data = df_sample.copy().dropna(subset=[target_column])
        if data.empty:
            raise HTTPException(400, "No rows remain after dropping missing target values.")

        # ── Detect task type ──────────────────────────────────────────────────────
        task = detect_task_type(data[target_column])

        selected_features = select_prediction_features(data, target_column, primary_keys)
        if not selected_features:
            raise HTTPException(400, "No suitable features found for prediction.")

        data = data[selected_features + [target_column]].copy()
        X = data[selected_features]

        numeric_features = [c for c in selected_features if is_numeric_column(data, c)]
        categorical_features = [c for c in selected_features if c not in numeric_features]
        preprocessor = _build_preprocessor(numeric_features, categorical_features)

        warnings_list = []
        model_results = []
        best_pipeline = None
        best_score = -np.inf
        best_model_name = ""
        le = None
        classes_list = []
        conf_matrix = []
        clf_report = {}

        # ── Classification ────────────────────────────────────────────────────────
        if task == "classification":
            y = data[target_column].astype(str)
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            classes_list = le.classes_.tolist()
            unique_cls = len(classes_list)

            if unique_cls < 2:
                raise HTTPException(400, "Target column must have at least 2 classes.")
            if unique_cls > 50:
                raise HTTPException(400, "Too many classes (>50). Please choose a different target.")

            class_counts = np.bincount(y_enc)
            can_stratify = class_counts.min() >= 2
            cv_folds = min(5, class_counts.min()) if can_stratify else 3

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_enc, test_size=0.2, random_state=42,
                stratify=y_enc if can_stratify else None
            )

            imbalance = summarize_imbalance(y)
            if imbalance.get("majority_share", 0) >= 0.8:
                warnings_list.append("Class imbalance detected — consider SMOTE or class_weight='balanced'.")
            if len(y) < 200:
                warnings_list.append("Small training sample — metrics may be unstable.")
            if not can_stratify:
                warnings_list.append("Stratified split not possible due to rare classes.")

            candidate_models = [
                ("Logistic Regression",
                 LogisticRegression(max_iter=500, n_jobs=-1, solver="saga",
                                    class_weight="balanced")),
                ("Decision Tree",
                 DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42)),
                ("Random Forest",
                 RandomForestClassifier(n_estimators=150, n_jobs=-1, class_weight="balanced",
                                        max_features="sqrt", random_state=42)),
                ("Gradient Boosting",
                 GradientBoostingClassifier(n_estimators=80, learning_rate=0.1,
                                            max_depth=4, random_state=42)),
                ("Extra Trees",
                 ExtraTreesClassifier(n_estimators=150, n_jobs=-1, class_weight="balanced",
                                      random_state=42)),
                ("K-Nearest Neighbors",
                 KNeighborsClassifier(n_neighbors=min(5, len(X_tr) // 10 or 1), n_jobs=-1)),
            ]

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if can_stratify \
                 else KFold(n_splits=cv_folds, shuffle=True, random_state=42)

            for name, clf_model in candidate_models:
                try:
                    pipe = Pipeline([("preprocessor", preprocessor), ("model", clf_model)])
                    # Cross-validation on training split
                    cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv,
                                                scoring="f1_macro", n_jobs=-1)
                    pipe.fit(X_tr, y_tr)
                    y_pred = pipe.predict(X_te)
                    acc = float(accuracy_score(y_te, y_pred))
                    f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
                    cv_mean = float(cv_scores.mean())
                    cv_std = float(cv_scores.std())
                    model_results.append({
                        "model": name,
                        "accuracy": round(acc, 4),
                        "f1_macro": round(f1, 4),
                        "cv_f1_mean": round(cv_mean, 4),
                        "cv_f1_std": round(cv_std, 4),
                        "stability": round(max(0.0, 1.0 - cv_std) * 100, 1),
                        "is_best": False
                    })
                    # Select best by CV score (more stable than test score)
                    if cv_mean > best_score:
                        best_score = cv_mean
                        best_pipeline = pipe
                        best_model_name = name
                except Exception as me:
                    logger.warning(f"Model {name} failed: {me}")
                    model_results.append({"model": name, "accuracy": None, "f1_macro": None,
                                          "cv_f1_mean": None, "cv_f1_std": None,
                                          "stability": None, "is_best": False, "error": str(me)})

            # Mark best
            for r in model_results:
                if r["model"] == best_model_name:
                    r["is_best"] = True

            if best_pipeline:
                y_pred_best = best_pipeline.predict(X_te)
                clf_report = classification_report(y_te, y_pred_best,
                                                   output_dict=True, zero_division=0)
                conf_matrix = confusion_matrix(y_te, y_pred_best).tolist() \
                    if len(classes_list) <= 25 else []

            # ── Feature importance chart ──────────────────────────────────────────
            feature_importance_chart = {}
            top_features = []
            if best_pipeline:
                try:
                    feat_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
                    top_features = _get_feature_importances(best_pipeline, feat_names)
                    if top_features:
                        fi_fig = go.Figure(go.Bar(
                            x=[f["importance"] for f in top_features[:20]],
                            y=[f["feature"] for f in top_features[:20]],
                            orientation="h",
                            marker_color="rgb(102,126,234)",
                            hovertemplate="Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>"
                        ))
                        fi_fig.update_layout(
                            title=f"Top Feature Importances ({best_model_name})",
                            xaxis_title="Importance",
                            yaxis=dict(autorange="reversed"),
                            template="plotly_dark",
                            height=max(400, len(top_features[:20]) * 22),
                            margin=dict(l=20, r=20, t=50, b=30)
                        )
                        feature_importance_chart = convert_plotly_figure_to_dict(fi_fig)
                except Exception as fi_err:
                    logger.warning(f"Feature importance chart failed: {fi_err}")

            # ── Model comparison chart ────────────────────────────────────────────
            model_comparison_chart = {}
            try:
                valid = [r for r in model_results if r.get("cv_f1_mean") is not None]
                if valid:
                    names_chart = [r["model"] for r in valid]
                    acc_vals = [r["accuracy"] for r in valid]
                    cv_vals = [r["cv_f1_mean"] for r in valid]
                    colors = ["rgb(255,215,0)" if r["is_best"] else "rgb(102,126,234)" for r in valid]
                    mc_fig = go.Figure()
                    mc_fig.add_trace(go.Bar(
                        name="Test Accuracy",
                        x=names_chart, y=acc_vals,
                        marker_color=colors,
                        hovertemplate="%{x}<br>Accuracy: %{y:.4f}<extra></extra>"
                    ))
                    mc_fig.add_trace(go.Bar(
                        name="CV F1 (mean)",
                        x=names_chart, y=cv_vals,
                        marker_color=["rgba(255,215,0,0.5)" if r["is_best"]
                                      else "rgba(102,126,234,0.5)" for r in valid],
                        hovertemplate="%{x}<br>CV F1: %{y:.4f}<extra></extra>"
                    ))
                    mc_fig.update_layout(
                        title="Model Comparison — AutoML Tournament",
                        barmode="group",
                        yaxis_title="Score",
                        template="plotly_dark",
                        height=420,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    model_comparison_chart = convert_plotly_figure_to_dict(mc_fig)
            except Exception as mc_err:
                logger.warning(f"Model comparison chart failed: {mc_err}")

            best_metrics = next((r for r in model_results if r.get("is_best")), {})

            return {
                "success": True,
                "target_column": target_column,
                "task": "classification",
                "classes": classes_list,
                "num_classes": len(classes_list),
                "best_model": best_model_name,
                "best_metrics": {
                    "accuracy": best_metrics.get("accuracy"),
                    "f1_macro": best_metrics.get("f1_macro"),
                    "cv_f1_mean": best_metrics.get("cv_f1_mean"),
                    "cv_f1_std": best_metrics.get("cv_f1_std"),
                    "stability": best_metrics.get("stability"),
                },
                "model_comparison": model_results,
                "confusion_matrix": conf_matrix,
                "classification_report": convert_to_json_serializable(clf_report),
                "class_distribution": imbalance,
                "selected_features": selected_features,
                "top_features": top_features,
                "feature_importance_chart": feature_importance_chart,
                "model_comparison_chart": model_comparison_chart,
                "train_size": len(X_tr),
                "test_size": len(X_te),
                "sampled_rows": len(df_sample),
                "warnings": warnings_list
            }

        # ── Regression ────────────────────────────────────────────────────────────
        else:
            y = pd.to_numeric(data[target_column], errors="coerce")
            data = data[y.notna()].copy()
            y = y[y.notna()]
            X = data[selected_features]

            if len(y) < 20:
                raise HTTPException(400, "Not enough rows for regression analysis.")

            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            warnings_list.append("Regression task detected (continuous target).")
            if len(y) < 200:
                warnings_list.append("Small training sample — regression metrics may be unstable.")

            candidate_reg = [
                ("Ridge Regression", Ridge()),
                ("Decision Tree", DecisionTreeRegressor(max_depth=10, random_state=42)),
                ("Random Forest", RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=42)),
                ("Gradient Boosting", GradientBoostingRegressor(n_estimators=80, learning_rate=0.1,
                                                                 max_depth=4, random_state=42)),
                ("Extra Trees", ExtraTreesRegressor(n_estimators=150, n_jobs=-1, random_state=42)),
            ]

            for name, reg_model in candidate_reg:
                try:
                    pipe = Pipeline([("preprocessor", preprocessor), ("model", reg_model)])
                    cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="r2", n_jobs=-1)
                    pipe.fit(X_tr, y_tr)
                    y_pred = pipe.predict(X_te)
                    r2 = float(r2_score(y_te, y_pred))
                    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
                    mae = float(mean_absolute_error(y_te, y_pred))
                    cv_mean = float(cv_scores.mean())
                    cv_std = float(cv_scores.std())
                    model_results.append({
                        "model": name,
                        "r2": round(r2, 4),
                        "rmse": round(rmse, 4),
                        "mae": round(mae, 4),
                        "cv_r2_mean": round(cv_mean, 4),
                        "cv_r2_std": round(cv_std, 4),
                        "stability": round(max(0.0, 1.0 - cv_std) * 100, 1),
                        "is_best": False
                    })
                    if cv_mean > best_score:
                        best_score = cv_mean
                        best_pipeline = pipe
                        best_model_name = name
                except Exception as me:
                    logger.warning(f"Regression model {name} failed: {me}")
                    model_results.append({"model": name, "r2": None, "rmse": None, "mae": None,
                                          "cv_r2_mean": None, "cv_r2_std": None,
                                          "stability": None, "is_best": False, "error": str(me)})

            for r in model_results:
                if r["model"] == best_model_name:
                    r["is_best"] = True

            # Feature importance
            top_features = []
            feature_importance_chart = {}
            if best_pipeline:
                try:
                    feat_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
                    top_features = _get_feature_importances(best_pipeline, feat_names)
                    if top_features:
                        fi_fig = go.Figure(go.Bar(
                            x=[f["importance"] for f in top_features[:20]],
                            y=[f["feature"] for f in top_features[:20]],
                            orientation="h",
                            marker_color="rgb(102,126,234)",
                            hovertemplate="Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>"
                        ))
                        fi_fig.update_layout(
                            title=f"Top Feature Importances ({best_model_name})",
                            xaxis_title="Importance",
                            yaxis=dict(autorange="reversed"),
                            template="plotly_dark",
                            height=max(400, len(top_features[:20]) * 22),
                            margin=dict(l=20, r=20, t=50, b=30)
                        )
                        feature_importance_chart = convert_plotly_figure_to_dict(fi_fig)
                except Exception as fi_err:
                    logger.warning(f"Regression feature importance chart failed: {fi_err}")

            # Comparison chart
            model_comparison_chart = {}
            try:
                valid = [r for r in model_results if r.get("r2") is not None]
                if valid:
                    colors = ["rgb(255,215,0)" if r["is_best"] else "rgb(102,126,234)" for r in valid]
                    mc_fig = go.Figure()
                    mc_fig.add_trace(go.Bar(
                        name="R² Score",
                        x=[r["model"] for r in valid],
                        y=[r["r2"] for r in valid],
                        marker_color=colors,
                        hovertemplate="%{x}<br>R²: %{y:.4f}<extra></extra>"
                    ))
                    mc_fig.add_trace(go.Bar(
                        name="CV R² (mean)",
                        x=[r["model"] for r in valid],
                        y=[r["cv_r2_mean"] for r in valid],
                        marker_color=["rgba(255,215,0,0.5)" if r["is_best"]
                                      else "rgba(102,126,234,0.5)" for r in valid],
                        hovertemplate="%{x}<br>CV R²: %{y:.4f}<extra></extra>"
                    ))
                    mc_fig.update_layout(
                        title="Regression Model Comparison — AutoML Tournament",
                        barmode="group",
                        yaxis_title="R² Score",
                        template="plotly_dark",
                        height=420,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    model_comparison_chart = convert_plotly_figure_to_dict(mc_fig)
            except Exception as mc_err:
                logger.warning(f"Regression comparison chart failed: {mc_err}")

            best_reg_metrics = next((r for r in model_results if r.get("is_best")), {})

            return {
                "success": True,
                "target_column": target_column,
                "task": "regression",
                "best_model": best_model_name,
                "best_metrics": {
                    "r2": best_reg_metrics.get("r2"),
                    "rmse": best_reg_metrics.get("rmse"),
                    "mae": best_reg_metrics.get("mae"),
                    "cv_r2_mean": best_reg_metrics.get("cv_r2_mean"),
                    "cv_r2_std": best_reg_metrics.get("cv_r2_std"),
                    "stability": best_reg_metrics.get("stability"),
                },
                "model_comparison": model_results,
                "selected_features": selected_features,
                "top_features": top_features,
                "feature_importance_chart": feature_importance_chart,
                "model_comparison_chart": model_comparison_chart,
                "train_size": len(X_tr),
                "test_size": len(X_te),
                "sampled_rows": len(df_sample),
                "warnings": warnings_list
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Predictive analysis error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Predictive analysis failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE DATASET UPLOAD  (ZIP of images organised by class sub-folders)
# ─────────────────────────────────────────────────────────────────────────────

def _suggest_augmentations(stats: dict) -> list:
    """Rule-based augmentation advisor for image datasets."""
    suggestions = []

    total = stats.get("total_images", 0)
    n_classes = stats.get("num_classes", 1)
    avg_per_class = total / max(1, n_classes)
    imbalance_ratio = stats.get("class_imbalance_ratio", 1.0)

    # Volume-based
    if total < 500:
        suggestions.append({
            "technique": "Horizontal Flip",
            "reason": "Very small dataset — doubles the effective training size cheaply.",
            "priority": "High",
            "code_hint": "transforms.RandomHorizontalFlip(p=0.5)"
        })
        suggestions.append({
            "technique": "Vertical Flip",
            "reason": "Useful when object orientation varies (e.g., aerial/medical images).",
            "priority": "Medium",
            "code_hint": "transforms.RandomVerticalFlip(p=0.5)"
        })
        suggestions.append({
            "technique": "Random Rotation (±30°)",
            "reason": "Improves rotational invariance for small datasets.",
            "priority": "High",
            "code_hint": "transforms.RandomRotation(degrees=30)"
        })
    elif total < 2000:
        suggestions.append({
            "technique": "Horizontal Flip",
            "reason": "Standard augmentation for moderate-sized datasets.",
            "priority": "High",
            "code_hint": "transforms.RandomHorizontalFlip(p=0.5)"
        })
        suggestions.append({
            "technique": "Random Rotation (±15°)",
            "reason": "Adds rotational variance without distortion.",
            "priority": "Medium",
            "code_hint": "transforms.RandomRotation(degrees=15)"
        })

    # Class imbalance
    if imbalance_ratio > 3.0:
        suggestions.append({
            "technique": "Oversampling via Augmentation",
            "reason": f"Class imbalance ratio is {imbalance_ratio:.1f}x — augment minority classes more aggressively.",
            "priority": "High",
            "code_hint": "Use WeightedRandomSampler or apply extra transforms to minority classes."
        })

    # Always suggest color jitter and normalize
    suggestions.append({
        "technique": "Color Jitter (brightness, contrast, saturation)",
        "reason": "Improves generalization under different lighting/imaging conditions.",
        "priority": "Medium",
        "code_hint": "transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)"
    })
    suggestions.append({
        "technique": "Random Crop / ResizedCrop",
        "reason": "Forces the model to focus on different regions of the image.",
        "priority": "Medium",
        "code_hint": "transforms.RandomResizedCrop(224, scale=(0.8, 1.0))"
    })
    suggestions.append({
        "technique": "Gaussian Blur / Noise",
        "reason": "Simulates sensor noise and slight defocus.",
        "priority": "Low",
        "code_hint": "transforms.GaussianBlur(kernel_size=3)"
    })
    suggestions.append({
        "technique": "Normalization (ImageNet stats)",
        "reason": "Standardizes pixel distribution for faster convergence with pretrained models.",
        "priority": "High",
        "code_hint": "transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"
    })

    # Domain-specific hints from stats
    avg_w = stats.get("avg_width", 0)
    avg_h = stats.get("avg_height", 0)
    if avg_w > 512 or avg_h > 512:
        suggestions.append({
            "technique": "Resize / Center Crop to 224×224 or 256×256",
            "reason": f"Images are large ({avg_w:.0f}×{avg_h:.0f}px). Resize for efficiency.",
            "priority": "High",
            "code_hint": "transforms.Resize(256), transforms.CenterCrop(224)"
        })

    is_grayscale = stats.get("is_predominantly_grayscale", False)
    if not is_grayscale:
        suggestions.append({
            "technique": "Random Grayscale",
            "reason": "Forces the model to not rely solely on colour cues.",
            "priority": "Low",
            "code_hint": "transforms.RandomGrayscale(p=0.1)"
        })

    # Deduplicate by technique name
    seen = set()
    deduped = []
    for s in suggestions:
        if s["technique"] not in seen:
            seen.add(s["technique"])
            deduped.append(s)

    # Sort by priority
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    deduped.sort(key=lambda x: priority_order.get(x["priority"], 3))
    return deduped


@app.post("/api/upload-images")
async def upload_image_dataset(file: UploadFile = File(...)):
    """
    Upload an image dataset as a ZIP file.
    Expected structure: class_name/image.jpg (sub-folder per class) OR flat folder.
    Returns: dataset_id, class stats, image stats, augmentation suggestions.
    """
    try:
        if not file.filename.endswith(".zip"):
            raise HTTPException(400, "Please upload a ZIP file containing your image dataset.")

        contents = await file.read()
        dataset_id = str(uuid.uuid4())

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
        class_counts: Dict[str, int] = {}
        widths, heights, channels = [], [], []
        total_images = 0
        sample_paths = []
        grayscale_count = 0

        with zipfile.ZipFile(io.BytesIO(contents)) as zf:
            names = zf.namelist()
            for name in names:
                ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
                if ext not in image_extensions:
                    continue
                parts = [p for p in name.replace("\\", "/").split("/") if p]
                if len(parts) >= 2:
                    class_label = parts[-2]
                else:
                    class_label = "__root__"

                class_counts[class_label] = class_counts.get(class_label, 0) + 1
                total_images += 1

                # Sample first 200 images for stats
                if total_images <= 200:
                    try:
                        with zf.open(name) as img_file:
                            img = Image.open(img_file)
                            w, h = img.size
                            widths.append(w)
                            heights.append(h)
                            mode = img.mode
                            if mode == "L":
                                channels.append(1)
                                grayscale_count += 1
                            elif mode in ("RGB", "BGR"):
                                channels.append(3)
                            elif mode == "RGBA":
                                channels.append(4)
                            else:
                                channels.append(len(mode))
                            if total_images <= 5:
                                sample_paths.append(name)
                    except Exception:
                        pass

        if total_images == 0:
            raise HTTPException(400, "No valid images found in the ZIP file.")

        # Compute stats
        counts = list(class_counts.values())
        imbalance_ratio = (max(counts) / max(1, min(counts))) if len(counts) > 1 else 1.0

        stats = {
            "total_images": total_images,
            "num_classes": len(class_counts),
            "class_distribution": class_counts,
            "class_imbalance_ratio": round(float(imbalance_ratio), 2),
            "avg_width": round(float(np.mean(widths)), 1) if widths else 0,
            "avg_height": round(float(np.mean(heights)), 1) if heights else 0,
            "min_width": int(min(widths)) if widths else 0,
            "max_width": int(max(widths)) if widths else 0,
            "min_height": int(min(heights)) if heights else 0,
            "max_height": int(max(heights)) if heights else 0,
            "avg_channels": round(float(np.mean(channels)), 1) if channels else 3,
            "is_predominantly_grayscale": grayscale_count > len(widths) * 0.7 if widths else False,
            "sample_image_paths": sample_paths
        }

        augmentation_suggestions = _suggest_augmentations(stats)

        # Generate AI augmentation commentary
        ai_commentary = ""
        try:
            ai_prompt = f"""You are a computer vision expert. Given this image dataset summary, give 3-4 concise sentences
describing the most impactful data augmentation strategy:

Dataset: {file.filename}
Total images: {total_images}
Classes ({len(class_counts)}): {dict(list(class_counts.items())[:10])}
Avg image size: {stats['avg_width']:.0f}x{stats['avg_height']:.0f}px
Class imbalance ratio: {imbalance_ratio:.1f}x
Grayscale: {stats['is_predominantly_grayscale']}

Focus on practical recommendations. No markdown, no bullet points. Plain text only."""
            ai_commentary = get_gemini_response(ai_prompt, "lite")
        except Exception:
            ai_commentary = "AI commentary unavailable."

        # Store as special image dataset
        datasets[dataset_id] = {
            "df": pd.DataFrame(),       # empty df — not tabular
            "df_sample": pd.DataFrame(),
            "filename": file.filename,
            "uploaded_at": datetime.now().isoformat(),
            "primary_keys": [],
            "is_sampled": False,
            "profile": {},
            "analysis_cache": {},
            "dataset_type": "image",
            "image_stats": stats,
            "augmentation_suggestions": augmentation_suggestions,
            "ai_augmentation_commentary": ai_commentary
        }
        eda_results[dataset_id] = {"columns": {}, "dataset_type": "image"}

        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "dataset_type": "image",
            "stats": stats,
            "augmentation_suggestions": augmentation_suggestions,
            "ai_commentary": ai_commentary
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image upload error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Image dataset upload failed: {str(e)}")


@app.get("/api/augmentation/{dataset_id}")
async def get_augmentation_suggestions(dataset_id: str):
    """Return augmentation suggestions for an image dataset."""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    ds = datasets[dataset_id]
    if ds.get("dataset_type") != "image":
        raise HTTPException(400, "This endpoint is for image datasets only.")
    return {
        "dataset_id": dataset_id,
        "filename": ds["filename"],
        "stats": ds.get("image_stats", {}),
        "suggestions": ds.get("augmentation_suggestions", []),
        "ai_commentary": ds.get("ai_augmentation_commentary", "")
    }


@app.get("/api/health-score/{dataset_id}")
async def get_data_health_score(dataset_id: str):
    """Return a composite data-health score (0-100) for the dataset."""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    ds = datasets[dataset_id]
    if ds.get("dataset_type") == "image":
        return {"dataset_id": dataset_id, "health": {"overall": 100, "grade": "N/A (image dataset)"}}
    df = ds["df"]
    primary_keys = ds.get("primary_keys", [])
    health = compute_data_health_score(df, primary_keys)
    return {"dataset_id": dataset_id, "health": health}


def parse_insights_into_sections(text: str) -> List[Dict]:
    """Parse AI response into sections"""
    sections = []
    current_section = None
    current_items = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        if line.upper().startswith('SECTION:'):
            if current_section:
                sections.append({
                    "title": current_section,
                    "items": current_items
                })
            current_section = line.split(':', 1)[1].strip()
            current_items = []
        elif line.startswith('-') or line.startswith('•'):
            item = line.lstrip('-•').strip()
            if item:
                current_items.append(item)
    
    if current_section:
        sections.append({
            "title": current_section,
            "items": current_items
        })
    
    return sections

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    """Chat with AI about the dataset"""
    if request.dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    
    try:
        df = datasets[request.dataset_id]["df"]
        eda = eda_results[request.dataset_id]
        filename = datasets[request.dataset_id]["filename"]
        
        # Build conversation history
        history_context = ""
        if request.history:
            history_context = "\n\nConversation History:\n"
            for msg in request.history[-5:]:
                history_context += f"User: {msg.get('user', '')}\nAI: {msg.get('ai', '')}\n"
        
        # Get sample data for context
        sample_data = df.head(5).to_dict('records')
        # Convert Timestamps and other non-JSON-serializable objects to strings
        sample_data = convert_to_json_serializable(sample_data)
        
        prompt = f"""You are a data analyst assistant. Answer questions about the dataset conversationally and accurately.

Dataset: {filename}
Rows: {len(df):,}, Columns: {len(df.columns)}

Column Information:
{json.dumps(convert_to_json_serializable(eda.get('columns', {})), indent=2)}

Sample Data (first 5 rows):
{json.dumps(sample_data, indent=2)}
{history_context}

User Question: {request.message}

Provide a clear, helpful, and accurate answer based on the dataset information above. If making calculations or observations, be specific and cite numbers from the data."""
        
        response = get_gemini_response(prompt, "flash")
        
        return {
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Chat failed: {str(e)}")

@app.post("/api/query")
async def query_dataset(request: QueryRequest):
    """Execute natural language query on dataset with pagination"""
    if request.dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    
    try:
        df = datasets[request.dataset_id]["df"]
        eda = eda_results[request.dataset_id]
        
        # Generate pandas query using AI
        prompt = f"""Convert this natural language query into pandas operations.

Dataset Columns: {list(df.columns)}
Column Info: {json.dumps(convert_to_json_serializable(eda.get('columns', {})), indent=2)}

Query: {request.query}

Provide ONLY the pandas code to execute this query. Use 'df' as the dataframe variable.
Return code that filters, groups, aggregates, or transforms the data as requested.

Example formats:
- "Show me rows where age > 30" → df[df['age'] > 30]
- "Average salary by department" → df.groupby('department')['salary'].mean()
- "Top 10 highest scores" → df.nlargest(10, 'score')

Only return the code, nothing else."""
        
        pandas_code = get_gemini_response(prompt, "flash").strip()
        
        # Clean up the code
        pandas_code = pandas_code.replace('```python', '').replace('```', '').strip()
        
        logger.info(f"Generated pandas code: {pandas_code}")
        
        # Execute the query safely
        try:
            result_df = eval(pandas_code, {"df": df, "pd": pd, "np": np})
            
            # Handle Series or scalar results
            if isinstance(result_df, pd.Series):
                result_df = result_df.to_frame()
            elif not isinstance(result_df, pd.DataFrame):
                result_df = pd.DataFrame({"result": [result_df]})
            
            # Apply pagination
            total_rows = len(result_df)
            page = max(1, request.page)
            page_size = min(500, max(10, request.page_size))
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            result_df_page = result_df.iloc[start_idx:end_idx]
            
            return {
                "success": True,
                "rows": total_rows,
                "total_rows": len(df),
                "columns": list(result_df.columns),
                "data": result_df_page.fillna("").to_dict('records'),
                "pandas_query": pandas_code,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_rows": total_rows,
                    "total_pages": (total_rows + page_size - 1) // page_size,
                    "has_next": end_idx < total_rows,
                    "has_prev": page > 1
                },
                "message": f"Query executed successfully. Showing page {page} of {(total_rows + page_size - 1) // page_size} ({total_rows:,} total results)."
            }
        except Exception as exec_error:
            logger.error(f"Query execution error: {str(exec_error)}")
            raise HTTPException(400, f"Query execution failed: {str(exec_error)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Query failed: {str(e)}")

@app.get("/api/column/{dataset_id}/{column_name}")
async def get_column_details(dataset_id: str, column_name: str):
    """Get detailed information about a specific column"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    
    try:
        df = datasets[dataset_id]["df"]
        primary_keys = datasets[dataset_id].get("primary_keys", [])
        
        if column_name not in df.columns:
            raise HTTPException(404, f"Column '{column_name}' not found")
        
        col_data = df[column_name]
        
        details = {
            "name": column_name,
            "dtype": str(col_data.dtype),
            "total_count": len(col_data),
            "null_count": int(col_data.isna().sum()),
            "null_percentage": round(col_data.isna().sum() / len(col_data) * 100, 2),
            "unique_count": int(col_data.nunique()),
            "is_numeric": is_numeric_column(df, column_name),
            "is_categorical": is_categorical_column(df, column_name, primary_keys),
            "is_primary_key": column_name in primary_keys
        }
        
        # Numeric column details
        if details["is_numeric"]:
            numeric_series = pd.to_numeric(col_data, errors='coerce').dropna()
            if len(numeric_series) > 0:
                details["statistics"] = {
                    "mean": float(numeric_series.mean()),
                    "median": float(numeric_series.median()),
                    "std": float(numeric_series.std()),
                    "min": float(numeric_series.min()),
                    "max": float(numeric_series.max()),
                    "q25": float(numeric_series.quantile(0.25)),
                    "q75": float(numeric_series.quantile(0.75))
                }
        
        # Categorical column details
        if details["is_categorical"] or details["unique_count"] <= 100:
            value_counts = col_data.value_counts().head(50)
            details["top_values"] = [
                {"value": str(k), "count": int(v), "percentage": round(v / len(col_data) * 100, 2)}
                for k, v in value_counts.items()
            ]
        
        return details
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Column details error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Failed to get column details: {str(e)}")

@app.get("/api/export/{dataset_id}/ppt")
async def export_ppt(dataset_id: str):
    """Export PPT report"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    
    try:
        df = datasets[dataset_id]["df"]
        eda = eda_results[dataset_id]
        filename = datasets[dataset_id]["filename"] if "filename" in datasets[dataset_id] else "EDA_Report.pptx"
        ppt_buffer = generate_eda_report_ppt(
            eda_metadata=eda,
            df=df,
            dataset_name=filename
        )
        return StreamingResponse(
            ppt_buffer,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={"Content-Disposition": f"attachment; filename=EDA_Report.pptx"}
        )
    
    except Exception as e:
        logger.error(f"Export error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Export failed: {str(e)}")

@app.get("/api/export/{dataset_id}/csv")
async def export_csv(dataset_id: str):
    """Export dataset as CSV"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    
    try:
        df = datasets[dataset_id]["df"]
        filename = datasets[dataset_id]["filename"]
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}_processed.csv"}
        )
    
    except Exception as e:
        logger.error(f"CSV export error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"CSV export failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "datasets_loaded": len(datasets),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "DataSet Querying LLM API",
        "version": "2.0",
        "endpoints": {
            "upload": "POST /api/upload",
            "list": "GET /api/datasets",
            "info": "GET /api/dataset/{dataset_id}",
            "numerical_analysis": "GET /api/analyze/{dataset_id}/numerical",
            "categorical_analysis": "GET /api/analyze/{dataset_id}/categorical",
            "correlations": "GET /api/analyze/{dataset_id}/correlations",
            "explore": "POST /api/explore",
            "insights": "GET /api/insights/{dataset_id}",
            "chat": "POST /api/chat",
            "query": "POST /api/query",
            "predictive": "POST /api/predictive",
            "upload_images": "POST /api/upload-images",
            "augmentation": "GET /api/augmentation/{dataset_id}",
            "health_score": "GET /api/health-score/{dataset_id}",
            "column_details": "GET /api/column/{dataset_id}/{column_name}",
            "export_ppt": "GET /api/export/{dataset_id}/ppt",
            "export_csv": "GET /api/export/{dataset_id}/csv",
            "delete": "DELETE /api/dataset/{dataset_id}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)