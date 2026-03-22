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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report,
                              mean_squared_error, r2_score, mean_absolute_error,
                              precision_score, recall_score)
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, RandomForestRegressor,
                               GradientBoostingRegressor, ExtraTreesRegressor,
                               BaggingRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import zipfile
import base64
from PIL import Image

# Import your existing modules
from clean_and_EDA_generate import enhanced_eda_json, clean_data, read_and_validate_file
from generate_report import generate_eda_report_ppt
from utils import get_gemini_response, get_ollama_vision_response, get_gemini_api_response

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
        
        # Try converting to numeric with strict conversion ratio against non-null source values
        src_non_null = df[col].dropna()
        if len(src_non_null) == 0:
            return False

        test_series = pd.to_numeric(src_non_null, errors='coerce')
        non_null_converted = test_series.dropna()
        conversion_ratio = len(non_null_converted) / len(src_non_null)

        # Require high convertibility to avoid treating mixed alphanumeric IDs as numeric
        if conversion_ratio >= 0.9:
            unique_count = non_null_converted.nunique()
            unique_ratio = unique_count / len(non_null_converted) if len(non_null_converted) > 0 else 0
            return unique_count > 5 or unique_ratio > 0.05
        
        return False
    except Exception as e:
        logger.warning(f"Error checking numeric for {col}: {str(e)}")
        return False

def is_sensible_numeric_column(df: pd.DataFrame, col: str, eda_info: dict = None, primary_keys: List[str] = None) -> bool:
    """
    Heuristic-only check (no LLM calls) — fast and deterministic.
    Returns False for ID columns, phone numbers, zip-codes, and other non-analysable
    numeric columns.  Returns True for measurements, scores, prices, ages, etc.
    """
    try:
        # ── Primary-key list check ────────────────────────────────────────────
        if primary_keys and col in primary_keys:
            logger.info(f"Column '{col}' is a detected primary key – skipping numeric analysis")
            return False

        col_lower = col.lower().replace('_', '').replace(' ', '')
        total_count = len(df)
        if total_count == 0:
            return False
        unique_count = df[col].nunique(dropna=True)
        unique_ratio = unique_count / total_count

        # ── Near-unique columns are very likely identifiers ───────────────────
        if unique_ratio > 0.97:
            logger.info(f"Column '{col}' skipped (unique ratio {unique_ratio:.2%} – likely identifier)")
            return False

        # ── ID / reference / code patterns in column name ─────────────────────
        id_name_patterns = [
            'id', 'identifier', 'key', 'pk', 'guid', 'uuid',
            'code', 'ref', 'reference', 'serial', 'index',
            'number', 'num', 'no', 'nr',
        ]
        if any(col_lower == p or col_lower.endswith(p) or col_lower.startswith(p)
               for p in id_name_patterns):
            if unique_ratio > 0.80:
                logger.info(f"Column '{col}' skipped (ID-like name + unique ratio {unique_ratio:.2%})")
                return False

        # ── Phone / mobile number patterns ────────────────────────────────────
        phone_name_patterns = ['mobile', 'phone', 'contact', 'tel', 'cell', 'fax', 'whatsapp']
        if any(p in col_lower for p in phone_name_patterns):
            sample_vals = pd.to_numeric(df[col].dropna().head(20), errors='coerce').dropna()
            if len(sample_vals) > 0:
                min_v, max_v = float(sample_vals.min()), float(sample_vals.max())
                # Phone numbers: 7-15 digit integers
                if min_v >= 1_000_000 and max_v < 1e16 and unique_ratio > 0.70:
                    logger.info(f"Column '{col}' skipped (phone/mobile number pattern)")
                    return False

        # ── Zip / postal / SSN / PIN / Aadhaar patterns ───────────────────────
        special_id_patterns = ['zip', 'postal', 'pincode', 'pin', 'ssn', 'aadhaar',
                                'passport', 'nic', 'pancard', 'reg', 'account', 'acct']
        if any(p in col_lower for p in special_id_patterns):
            if unique_ratio > 0.60:
                logger.info(f"Column '{col}' skipped (special ID pattern: {col})")
                return False

        # ── Columns with very few unique values and name suggests a code ──────
        if unique_count <= 1:
            return False  # constant column – useless for analysis

        return True

    except Exception as e:
        logger.warning(f"Error checking sensible numeric for '{col}': {e}")
        return True  # Default: include column


def _get_sensible_numeric_cols(df: pd.DataFrame, dataset_info: dict) -> List[str]:
    """
    Return list of numeric columns that are sensible for analysis, using the
    pre-computed sensible_cache stored during upload for instant lookup.
    Falls back to calling is_sensible_numeric_column directly if no cache.
    """
    primary_keys = dataset_info.get("primary_keys", [])
    sensible_cache = dataset_info.get("sensible_cache")
    result = []
    for col in df.columns:
        if not is_numeric_column(df, col):
            continue
        if sensible_cache is not None:
            if sensible_cache.get(col, True):
                result.append(col)
        else:
            if is_sensible_numeric_column(df, col, primary_keys=primary_keys):
                result.append(col)
    return result

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
        if pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
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
            elif file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
                df = pd.read_excel(io.BytesIO(contents))
            elif file.filename.endswith('.json'):
                try:
                    json_data = json.loads(contents.decode('utf-8'))
                    if isinstance(json_data, list):
                        df = pd.DataFrame(json_data)
                    elif isinstance(json_data, dict):
                        # Try records orientation first, then orient='index', then wrap
                        if any(isinstance(v, dict) for v in json_data.values()):
                            df = pd.DataFrame.from_dict(json_data, orient='index')
                        else:
                            df = pd.DataFrame([json_data])
                    else:
                        raise HTTPException(400, "JSON must be an array of objects or a flat object.")
                except (json.JSONDecodeError, UnicodeDecodeError) as je:
                    raise HTTPException(400, f"Invalid JSON file: {str(je)}")
            else:
                raise HTTPException(400, "Unsupported file format. Use CSV, XLSX, or JSON.")
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

        # Pre-compute sensible-numeric cache so all downstream analysis endpoints
        # reuse the result instead of re-checking column by column every time.
        sensible_cache: Dict[str, bool] = {}
        for _col in df_sample.columns:
            if is_numeric_column(df_sample, _col):
                sensible_cache[_col] = is_sensible_numeric_column(df_sample, _col, eda, primary_keys)

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
            "analysis_cache": {},
            "sensible_cache": sensible_cache
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
        primary_keys = datasets[dataset_id].get("primary_keys", [])

        all_numeric_cols = [col for col in df_visual.columns if is_numeric_column(df_visual, col)]
        sensible_numeric_cols = _get_sensible_numeric_cols(df_visual, datasets[dataset_id])
        skipped_cols = [c for c in all_numeric_cols if c not in sensible_numeric_cols]
        logger.info(f"Numerical: {len(sensible_numeric_cols)} sensible / {len(all_numeric_cols)} total numeric cols")

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
        primary_keys = datasets[dataset_id].get("primary_keys", [])

        all_numeric_cols = [col for col in df_visual.columns if is_numeric_column(df_visual, col)]
        numeric_cols = _get_sensible_numeric_cols(df_visual, datasets[dataset_id])
        logger.info(f"Correlation: {len(numeric_cols)} sensible / {len(all_numeric_cols)} total numeric cols")
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
        primary_keys = datasets[dataset_id].get("primary_keys", [])

        all_numeric_cols = [col for col in df_visual.columns if is_numeric_column(df_visual, col)]
        numeric_cols = _get_sensible_numeric_cols(df_visual, datasets[dataset_id])
        logger.info(f"Outliers: {len(numeric_cols)} sensible / {len(all_numeric_cols)} total numeric cols")

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
        
        # Detect date/time columns robustly:
        # Step 1 – columns already parsed to datetime64 by clean_data()
        # Step 2 – string/object columns that can be parsed as dates (coerce mode, 50% threshold)
        date_cols = []
        for col in df_visual.columns:
            # ── Step 1: dtype is already datetime64 ──────────────────────────
            if pd.api.types.is_datetime64_any_dtype(df_visual[col]):
                date_cols.append(col)
                logger.info(f"Detected pre-parsed datetime column: {col} (dtype={df_visual[col].dtype})")
                continue

            # ── Skip pure numeric types (int / float = measurements, not dates) ─
            if pd.api.types.is_numeric_dtype(df_visual[col]):
                continue

            # ── Step 2: try parsing string/object column as datetime ──────────
            try:
                # infer_datetime_format works across pandas 1.x and 2.x
                try:
                    test_series = pd.to_datetime(
                        df_visual[col], errors='coerce', infer_datetime_format=True
                    )
                except TypeError:
                    # pandas 2.2+ deprecated infer_datetime_format
                    test_series = pd.to_datetime(df_visual[col], errors='coerce')

                total_count = len(df_visual)
                if total_count == 0:
                    continue

                valid_count = int(test_series.notna().sum())
                valid_ratio = valid_count / total_count

                # Accept column if at least 50 % of rows parsed successfully
                if valid_ratio >= 0.5 and valid_count >= 5:
                    valid_dates = test_series.dropna()
                    try:
                        min_date = valid_dates.min()
                        max_date = valid_dates.max()
                        if hasattr(min_date, 'year') and 1900 <= min_date.year and max_date.year <= 2100:
                            date_cols.append(col)
                            logger.info(
                                f"Detected datetime column by parsing: '{col}' "
                                f"({valid_ratio:.0%} valid, {min_date.date()} → {max_date.date()})"
                            )
                    except Exception:
                        pass
            except Exception as dt_err:
                logger.debug(f"Column '{col}' not parseable as datetime: {dt_err}")
                continue
        
        primary_keys = datasets[dataset_id].get("primary_keys", [])

        all_numeric_cols = [col for col in df_visual.columns if is_numeric_column(df_visual, col)]
        numeric_cols = _get_sensible_numeric_cols(df_visual, datasets[dataset_id])

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


# ─────────────────────────────────────────────────────────────────────────────
#  TIME SERIES FORECAST  (linear trend + exponential smoothing extrapolation)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/analyze/{dataset_id}/forecast")
async def get_forecast_analysis(dataset_id: str):
    """
    Simple time series forecasting: fits a linear trend + centred moving average
    on each date×numeric pair and extrapolates 10 future periods.
    Works entirely offline — no extra ML library required.
    """
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    cache = datasets[dataset_id].setdefault("analysis_cache", {})
    if "forecast" in cache:
        return cache["forecast"]
    try:
        df = datasets[dataset_id]["df"]
        primary_keys = datasets[dataset_id].get("primary_keys", [])
        df_visual = downsample_ordered(df, MAX_EDA_SAMPLE_ROWS)

        # ── Detect datetime columns (same logic as timeseries endpoint) ──────
        date_cols: List[str] = []
        for col in df_visual.columns:
            if pd.api.types.is_datetime64_any_dtype(df_visual[col]):
                date_cols.append(col)
                continue
            if pd.api.types.is_numeric_dtype(df_visual[col]):
                continue
            try:
                try:
                    ts = pd.to_datetime(df_visual[col], errors='coerce', infer_datetime_format=True)
                except TypeError:
                    ts = pd.to_datetime(df_visual[col], errors='coerce')
                vr = ts.notna().sum() / max(1, len(df_visual))
                if vr >= 0.5 and ts.notna().sum() >= 5:
                    vd = ts.dropna()
                    if hasattr(vd.min(), 'year') and 1900 <= vd.min().year and vd.max().year <= 2100:
                        date_cols.append(col)
            except Exception:
                continue

        numeric_cols = _get_sensible_numeric_cols(df_visual, datasets[dataset_id])

        if not date_cols or not numeric_cols:
            result = {"type": "forecast", "error": "No time series data detected",
                      "visualizations": [], "forecasts": {}}
            cache["forecast"] = result
            return result

        visualizations: List[dict] = []
        forecasts: Dict[str, Any] = {}

        for date_col in date_cols[:1]:   # use first datetime column
            for num_col in numeric_cols[:3]:   # forecast up to 3 metrics
                try:
                    ts_df = df_visual[[date_col, num_col]].copy()
                    ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors='coerce')
                    ts_df[num_col] = pd.to_numeric(ts_df[num_col], errors='coerce')
                    ts_df = ts_df.dropna().sort_values(date_col).reset_index(drop=True)

                    if len(ts_df) < 10:
                        continue

                    # ── Aggregate to monthly to reduce noise ─────────────────
                    ts_df.set_index(date_col, inplace=True)
                    monthly = ts_df[num_col].resample('ME').mean().dropna()
                    if len(monthly) < 6:
                        monthly = ts_df[num_col].resample('W').mean().dropna()
                    if len(monthly) < 4:
                        continue

                    x = np.arange(len(monthly))
                    y = monthly.values.astype(float)

                    # ── Fit linear trend ─────────────────────────────────────
                    slope, intercept = np.polyfit(x, y, 1)

                    # ── Exponential smoothing (alpha = 0.3) ───────────────────
                    alpha = 0.3
                    smoothed = np.zeros(len(y))
                    smoothed[0] = y[0]
                    for i in range(1, len(y)):
                        smoothed[i] = alpha * y[i] + (1 - alpha) * smoothed[i - 1]

                    # ── Extrapolate 10 future periods ─────────────────────────
                    n_future = 10
                    freq = monthly.index.freq or pd.tseries.frequencies.to_offset('ME')
                    last_date = monthly.index[-1]
                    future_dates = pd.date_range(start=last_date, periods=n_future + 1, freq=freq)[1:]
                    future_x = np.arange(len(monthly), len(monthly) + n_future)
                    trend_forecast = slope * future_x + intercept
                    last_smoothed = smoothed[-1]
                    exp_forecast = np.zeros(n_future)
                    prev = last_smoothed
                    for i in range(n_future):
                        trend_val = trend_forecast[i]
                        prev = alpha * trend_val + (1 - alpha) * prev
                        exp_forecast[i] = prev

                    # ── Confidence interval (±1 std of historical residuals) ──
                    hist_residuals = y - (slope * x + intercept)
                    ci = float(np.std(hist_residuals))

                    # ── Plotly figure ─────────────────────────────────────────
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=monthly.index.strftime('%Y-%m-%d').tolist(),
                        y=y.tolist(),
                        mode='lines+markers',
                        name='Historical',
                        line=dict(color='rgb(102,126,234)', width=2),
                        marker=dict(size=4)
                    ))
                    fig.add_trace(go.Scatter(
                        x=monthly.index.strftime('%Y-%m-%d').tolist(),
                        y=smoothed.tolist(),
                        mode='lines',
                        name='Smoothed (α=0.3)',
                        line=dict(color='rgb(255,193,7)', width=2, dash='dot')
                    ))
                    fd_str = future_dates.strftime('%Y-%m-%d').tolist()
                    fig.add_trace(go.Scatter(
                        x=fd_str + fd_str[::-1],
                        y=(exp_forecast + ci).tolist() + (exp_forecast - ci).tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255,100,100,0.15)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Interval',
                        showlegend=True
                    ))
                    fig.add_trace(go.Scatter(
                        x=fd_str,
                        y=exp_forecast.tolist(),
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='rgb(231,76,60)', width=2, dash='dash'),
                        marker=dict(size=5, symbol='diamond')
                    ))
                    fig.update_layout(
                        title=f"Forecast: {num_col} (next {n_future} periods)",
                        xaxis_title=date_col,
                        yaxis_title=num_col,
                        template='plotly_dark',
                        hovermode='x unified',
                        height=500
                    )

                    uid = f"{date_col}_{num_col}"
                    visualizations.append({
                        "type": "forecast",
                        "date_column": date_col,
                        "value_column": num_col,
                        "figure": convert_plotly_figure_to_dict(fig),
                        "unique_id": uid
                    })
                    forecasts[uid] = {
                        "date_column": date_col,
                        "value_column": num_col,
                        "periods_forecast": n_future,
                        "last_historical_value": round(float(y[-1]), 4),
                        "forecast_end_value": round(float(exp_forecast[-1]), 4),
                        "trend_direction": "upward" if slope > 0 else "downward",
                        "slope_per_period": round(float(slope), 6),
                        "confidence_interval": round(ci, 4),
                        "method": "Exponential Smoothing + Linear Trend (α=0.3)"
                    }

                except Exception as fc_err:
                    logger.warning(f"Forecast skipped {date_col}×{num_col}: {fc_err}")
                    continue

        result = {
            "type": "forecast",
            "date_columns": date_cols,
            "numeric_columns": numeric_cols,
            "visualizations": visualizations,
            "forecasts": forecasts,
            "method": "Exponential Smoothing with Linear Trend Extrapolation"
        }
        cache["forecast"] = result
        return result

    except Exception as e:
        logger.error(f"Forecast error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Forecast analysis failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
#  ANOMALY DETECTION  (IQR + Z-score + Isolation Forest)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/analyze/{dataset_id}/anomaly")
async def get_anomaly_detection(dataset_id: str):
    """
    Multi-method anomaly detection:
    ① Per-column IQR fences  ② Per-column Z-score (|z|>3)
    ③ Multi-variate Isolation Forest on all sensible numeric columns
    Returns anomaly scores, row-level flags, and interactive scatter plots.
    """
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    cache = datasets[dataset_id].setdefault("analysis_cache", {})
    if "anomaly" in cache:
        return cache["anomaly"]
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler as _Scaler
        from sklearn.impute import SimpleImputer as _Imputer

        df = datasets[dataset_id]["df"]
        df_visual = datasets[dataset_id].get("df_sample", df)
        primary_keys = datasets[dataset_id].get("primary_keys", [])
        numeric_cols = _get_sensible_numeric_cols(df_visual, datasets[dataset_id])

        if not numeric_cols:
            result = {"type": "anomaly", "error": "No numeric columns for anomaly detection",
                      "visualizations": [], "summary": {}}
            cache["anomaly"] = result
            return result

        visualizations: List[dict] = []
        column_results: Dict[str, Any] = {}

        # ── Per-column univariate anomaly ─────────────────────────────────────
        for col in numeric_cols[:8]:
            try:
                series = pd.to_numeric(df_visual[col], errors='coerce').dropna()
                if len(series) < 10:
                    continue

                q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
                iqr = q3 - q1
                lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                iqr_mask = (series < lb) | (series > ub)

                mean_v, std_v = float(series.mean()), float(series.std())
                z_mask = (np.abs((series - mean_v) / std_v) > 3) if std_v > 0 else pd.Series([False] * len(series))

                combined_mask = iqr_mask | z_mask
                anomaly_vals = series[combined_mask]

                column_results[col] = {
                    "iqr_anomalies": int(iqr_mask.sum()),
                    "zscore_anomalies": int(z_mask.sum()),
                    "combined_anomalies": int(combined_mask.sum()),
                    "anomaly_rate_pct": round(combined_mask.sum() / len(series) * 100, 2),
                    "iqr_bounds": {"lower": round(lb, 4), "upper": round(ub, 4)},
                    "sample_anomaly_values": sorted(anomaly_vals.tolist())[:10]
                }

                # Scatter plot with anomalies highlighted
                normal_idx = series[~combined_mask].index.tolist()
                anomaly_idx = series[combined_mask].index.tolist()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(series[~combined_mask]))),
                    y=series[~combined_mask].tolist(),
                    mode='markers', name='Normal',
                    marker=dict(color='rgb(102,126,234)', size=4, opacity=0.6),
                    hovertemplate='Value: %{y}<extra>Normal</extra>'
                ))
                if len(anomaly_vals) > 0:
                    fig.add_trace(go.Scatter(
                        x=list(range(len(anomaly_vals))),
                        y=anomaly_vals.tolist(),
                        mode='markers', name='Anomaly',
                        marker=dict(color='rgb(231,76,60)', size=9, symbol='x',
                                    line=dict(width=2, color='red')),
                        hovertemplate='Anomaly Value: %{y}<extra></extra>'
                    ))
                fig.add_hline(y=ub, line_dash='dash', line_color='orange',
                              annotation_text=f'IQR Upper: {ub:.2f}')
                fig.add_hline(y=lb, line_dash='dash', line_color='orange',
                              annotation_text=f'IQR Lower: {lb:.2f}')
                fig.update_layout(
                    title=f"{col} — Anomaly Detection ({int(combined_mask.sum())} anomalies)",
                    yaxis_title=col, xaxis_title='Row Index',
                    template='plotly_dark', height=420
                )
                visualizations.append({"type": "anomaly_scatter", "column": col,
                                        "figure": convert_plotly_figure_to_dict(fig)})
            except Exception as col_err:
                logger.warning(f"Anomaly univariate skipped {col}: {col_err}")

        # ── Multi-variate Isolation Forest ────────────────────────────────────
        iso_result: Dict[str, Any] = {}
        try:
            mv_cols = numeric_cols[:10]
            X_mv = df_visual[mv_cols].copy()
            imputer = _Imputer(strategy='median')
            scaler = _Scaler()
            X_imp = imputer.fit_transform(X_mv)
            X_sc = scaler.fit_transform(X_imp)

            n_contam = min(0.05, max(0.01, 0.05))  # 5% contamination assumption
            iso = IsolationForest(n_estimators=50, contamination=n_contam,
                                  random_state=42, n_jobs=-1)
            preds = iso.fit_predict(X_sc)          # -1 = anomaly, 1 = normal
            scores = iso.decision_function(X_sc)   # more negative = more anomalous

            anomaly_mask = preds == -1
            iso_result = {
                "columns_used": mv_cols,
                "total_anomalies": int(anomaly_mask.sum()),
                "anomaly_rate_pct": round(float(anomaly_mask.mean()) * 100, 2),
                "avg_anomaly_score": round(float(scores[anomaly_mask].mean()), 4) if anomaly_mask.any() else None
            }

            # 2-D scatter of first 2 numeric cols, coloured by isolation score
            if len(mv_cols) >= 2:
                c1, c2 = mv_cols[0], mv_cols[1]
                fig_iso = go.Figure()
                normal_df = df_visual[[c1, c2]].copy()
                normal_df[c1] = pd.to_numeric(normal_df[c1], errors='coerce')
                normal_df[c2] = pd.to_numeric(normal_df[c2], errors='coerce')
                fig_iso.add_trace(go.Scatter(
                    x=normal_df[~anomaly_mask][c1].tolist(),
                    y=normal_df[~anomaly_mask][c2].tolist(),
                    mode='markers', name='Normal',
                    marker=dict(color='rgb(102,126,234)', size=5, opacity=0.5),
                    hovertemplate=f'{c1}: %{{x}}<br>{c2}: %{{y}}<extra>Normal</extra>'
                ))
                fig_iso.add_trace(go.Scatter(
                    x=normal_df[anomaly_mask][c1].tolist(),
                    y=normal_df[anomaly_mask][c2].tolist(),
                    mode='markers', name='Anomaly (Isolation Forest)',
                    marker=dict(color='rgb(231,76,60)', size=9, symbol='x',
                                line=dict(width=2, color='red')),
                    hovertemplate=f'{c1}: %{{x}}<br>{c2}: %{{y}}<extra>Anomaly</extra>'
                ))
                fig_iso.update_layout(
                    title=f"Isolation Forest Anomalies: {c1} vs {c2}",
                    xaxis_title=c1, yaxis_title=c2,
                    template='plotly_dark', height=480
                )
                visualizations.append({"type": "isolation_forest",
                                        "figure": convert_plotly_figure_to_dict(fig_iso)})
        except Exception as iso_err:
            logger.warning(f"Isolation Forest failed: {iso_err}")
            iso_result = {"error": str(iso_err)}

        result = {
            "type": "anomaly",
            "numeric_columns": numeric_cols,
            "column_results": column_results,
            "isolation_forest": iso_result,
            "visualizations": visualizations,
            "summary": {
                "columns_analyzed": len(column_results),
                "total_univariate_anomalies": sum(
                    v.get("combined_anomalies", 0) for v in column_results.values()
                ),
                "multivariate_anomalies": iso_result.get("total_anomalies", 0)
            }
        }
        cache["anomaly"] = result
        return result

    except Exception as e:
        logger.error(f"Anomaly detection error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Anomaly detection failed: {str(e)}")


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
        primary_keys = datasets[dataset_id].get("primary_keys", [])

        all_numeric_cols = [col for col in df_visual.columns if is_numeric_column(df_visual, col)]
        numeric_cols = _get_sensible_numeric_cols(df_visual, datasets[dataset_id])

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

def _build_insights_metadata(df: pd.DataFrame, eda: dict) -> str:
    """
    Build a compact, token-efficient metadata string from EDA for AI insights.
    Strips histogram bins, raw value-count lists, and other bulky arrays so the
    LLM prompt stays small and fast while still containing all useful statistics.
    """
    lines = []
    lines.append(f"Rows: {eda.get('num_rows', len(df)):,}  Columns: {eda.get('num_columns', len(df.columns))}")

    total_cells = len(df) * len(df.columns)
    total_missing = int(df.isna().sum().sum())
    dup_rows = int(df.duplicated().sum())
    lines.append(f"Missing cells: {total_missing} ({round(total_missing/max(1,total_cells)*100,1)}%)  Duplicate rows: {dup_rows}")

    cols_meta = []
    for col, info in eda.get("columns", {}).items():
        dtype = info.get("dtype", "?")
        miss_pct = info.get("missing_percent", 0)
        entry = f"  {col} [{dtype}] missing={miss_pct}%"

        num_stats = info.get("numeric_stats")
        if num_stats:
            entry += (
                f"  mean={num_stats.get('mean')}  std={num_stats.get('std')}"
                f"  min={num_stats.get('min')}  max={num_stats.get('max')}"
                f"  skew={info.get('skewness')}  outliers={info.get('outlier_count', 0)}"
            )
        else:
            # Categorical – show top-5 values only
            vc = info.get("value_counts") or info.get("top_values") or {}
            if isinstance(vc, dict):
                top = list(vc.items())[:5]
            elif isinstance(vc, list):
                top = [(v.get("value", v), v.get("count", "")) for v in vc[:5]]
            else:
                top = []
            unique = info.get("unique_count", info.get("n_unique", "?"))
            entry += f"  unique={unique}  top={top}"

        cols_meta.append(entry)

    lines.append("Columns:")
    lines.extend(cols_meta)
    return "\n".join(lines)


def _build_compact_columns_context(df: pd.DataFrame, eda: dict, max_cols: int = 25) -> str:
    """Compact schema summary to keep LLM prompts under free-tier token limits."""
    lines = [f"Rows: {len(df):,}, Columns: {len(df.columns)}"]
    columns_meta = eda.get("columns", {}) if isinstance(eda, dict) else {}

    shown_cols = list(df.columns)[:max_cols]
    for col in shown_cols:
        info = columns_meta.get(col, {}) if isinstance(columns_meta, dict) else {}
        dtype = info.get("dtype", str(df[col].dtype))
        miss = info.get("missing_percent", round(float(df[col].isna().mean() * 100), 2))
        unique = info.get("unique_count", int(df[col].nunique(dropna=True)))
        line = f"- {col} [{dtype}] missing={miss}% unique={unique}"

        num_stats = info.get("numeric_stats") if isinstance(info, dict) else None
        if isinstance(num_stats, dict):
            line += f" mean={num_stats.get('mean')} min={num_stats.get('min')} max={num_stats.get('max')}"

        lines.append(line)

    hidden = len(df.columns) - len(shown_cols)
    if hidden > 0:
        lines.append(f"- ... {hidden} additional columns omitted for brevity")

    return "\n".join(lines)


@app.get("/api/insights/{dataset_id}")
async def generate_insights(dataset_id: str):
    """Generate AI insights"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")

    # Return cached result if already generated for this dataset
    cache = datasets[dataset_id].setdefault("analysis_cache", {})
    if "ai_insights" in cache:
        return cache["ai_insights"]

    try:
        df = datasets[dataset_id]["df"]
        eda = eda_results[dataset_id]
        filename = datasets[dataset_id]["filename"]

        # Use compact metadata instead of full EDA JSON to keep the prompt small
        meta = _build_insights_metadata(df, eda)

        prompt = f"""Analyze this dataset and provide insights in a structured format.

Dataset: {filename}
{meta}

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

        raw_insights = get_gemini_api_response(prompt)
        if raw_insights.startswith("Error:"):
            raise HTTPException(500, raw_insights)

        sections = parse_insights_into_sections(raw_insights)
        if not sections:
            raise HTTPException(500, "Gemini returned an unexpected format for insights.")

        result = {"insights": sections, "raw": raw_insights}
        cache["ai_insights"] = result
        return result

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
    def _coerce_numeric_frame(X):
        # Ensures mixed/object numeric-like columns are converted before median imputation.
        if isinstance(X, pd.DataFrame):
            return X.apply(pd.to_numeric, errors="coerce")
        return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")

    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline([
            ("to_numeric", FunctionTransformer(_coerce_numeric_frame, validate=False)),
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

            # Limit n_estimators to 50 for fast turnaround; still competitive accuracy
            candidate_models = [
                ("Logistic Regression",
                 LogisticRegression(max_iter=300, n_jobs=-1, solver="saga",
                                    class_weight="balanced")),
                ("Decision Tree",
                 DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42)),
                ("Random Forest",
                 RandomForestClassifier(n_estimators=50, n_jobs=-1, class_weight="balanced",
                                        max_features="sqrt", random_state=42)),
                ("Gradient Boosting",
                 GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                            max_depth=4, random_state=42)),
                ("Extra Trees",
                 ExtraTreesClassifier(n_estimators=50, n_jobs=-1, class_weight="balanced",
                                      random_state=42)),
                ("K-Nearest Neighbors",
                 KNeighborsClassifier(n_neighbors=min(5, len(X_tr) // 10 or 1), n_jobs=-1)),
            ]

            cv_folds = min(3, cv_folds)  # cap at 3 for speed
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
            cv = KFold(n_splits=3, shuffle=True, random_state=42)  # 3 folds for speed

            warnings_list.append("Regression task detected (continuous target).")
            if len(y) < 200:
                warnings_list.append("Small training sample — regression metrics may be unstable.")

            # Limit n_estimators to 50 for fast turnaround
            candidate_reg = [
                ("Ridge Regression", Ridge()),
                ("Decision Tree", DecisionTreeRegressor(max_depth=8, random_state=42)),
                ("Random Forest", RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)),
                ("Gradient Boosting", GradientBoostingRegressor(n_estimators=50, learning_rate=0.1,
                                                                 max_depth=4, random_state=42)),
                ("Extra Trees", ExtraTreesRegressor(n_estimators=50, n_jobs=-1, random_state=42)),
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
    """Rule-based augmentation advisor for tabular image dataset stats (ZIP uploads)."""
    suggestions = []

    total = stats.get("total_images", 0)
    n_classes = stats.get("num_classes", 1)
    imbalance_ratio = stats.get("class_imbalance_ratio", 1.0)
    avg_w = stats.get("avg_width", 224)
    avg_h = stats.get("avg_height", 224)
    is_grayscale = stats.get("is_predominantly_grayscale", False)

    # ── Volume-based geometric augmentations ──────────────────────────────────
    if total < 200:
        suggestions.append({
            "technique": "Random Horizontal Flip",
            "reason": f"Tiny dataset ({total} images) — flip instantly doubles effective training samples.",
            "priority": "High",
            "code_hint": "transforms.RandomHorizontalFlip(p=0.5)"
        })
        suggestions.append({
            "technique": "Random Rotation (±30°)",
            "reason": "Small dataset benefits from strong geometric diversity — 30° rotation adds significant variance.",
            "priority": "High",
            "code_hint": "transforms.RandomRotation(degrees=30)"
        })
        suggestions.append({
            "technique": "Random Vertical Flip",
            "reason": "For aerial/medical/microscopy images with <200 samples, vertical flip adds useful orientation variance.",
            "priority": "Medium",
            "code_hint": "transforms.RandomVerticalFlip(p=0.3)"
        })
        suggestions.append({
            "technique": "Random Affine Transform",
            "reason": f"Very few images ({total}) — affine shear/translate/scale combos maximise training diversity.",
            "priority": "High",
            "code_hint": "transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))"
        })
    elif total < 1000:
        suggestions.append({
            "technique": "Random Horizontal Flip",
            "reason": f"Moderate dataset ({total} images) — horizontal flip is the safest baseline augmentation.",
            "priority": "High",
            "code_hint": "transforms.RandomHorizontalFlip(p=0.5)"
        })
        suggestions.append({
            "technique": "Random Rotation (±20°)",
            "reason": "Adds rotational variance without excessive padding artefacts for medium-sized datasets.",
            "priority": "Medium",
            "code_hint": "transforms.RandomRotation(degrees=20)"
        })
    elif total < 5000:
        suggestions.append({
            "technique": "Random Horizontal Flip",
            "reason": "Standard baseline — effective at any dataset size.",
            "priority": "High",
            "code_hint": "transforms.RandomHorizontalFlip(p=0.5)"
        })
        suggestions.append({
            "technique": "Random Rotation (±10°)",
            "reason": "Light rotation is sufficient for datasets with adequate diversity (1k-5k images).",
            "priority": "Low",
            "code_hint": "transforms.RandomRotation(degrees=10)"
        })
    else:
        suggestions.append({
            "technique": "Random Horizontal Flip",
            "reason": f"Large dataset ({total:,} images) — flip remains the standard baseline augmentation.",
            "priority": "High",
            "code_hint": "transforms.RandomHorizontalFlip(p=0.5)"
        })

    # ── Class imbalance ───────────────────────────────────────────────────────
    if imbalance_ratio > 5.0:
        suggestions.append({
            "technique": "Oversample Minority Classes (WeightedRandomSampler)",
            "reason": f"Severe class imbalance ({imbalance_ratio:.1f}×). Weighted sampling ensures equal class exposure per epoch.",
            "priority": "High",
            "code_hint": "torch.utils.data.WeightedRandomSampler(weights, num_samples)"
        })
        suggestions.append({
            "technique": "Aggressive Augmentation for Minority Classes",
            "reason": f"Critical: {imbalance_ratio:.1f}× imbalance — apply stronger transforms only to minority-class images.",
            "priority": "High",
            "code_hint": "Apply transforms.Compose([...]) only in minority class dataset branches."
        })
    elif imbalance_ratio > 3.0:
        suggestions.append({
            "technique": "Oversampling via Augmentation",
            "reason": f"Imbalance ratio {imbalance_ratio:.1f}× detected — augment minority classes more aggressively to balance.",
            "priority": "High",
            "code_hint": "Use class_weight='balanced' or oversample with augmented copies."
        })

    # ── Resolution / size ─────────────────────────────────────────────────────
    if avg_w > 512 or avg_h > 512:
        suggestions.append({
            "technique": "Resize + Random Resized Crop to 224×224",
            "reason": f"Images average {avg_w:.0f}×{avg_h:.0f}px — resize significantly reduces memory and training time.",
            "priority": "High",
            "code_hint": "transforms.Resize(256), transforms.RandomResizedCrop(224, scale=(0.8,1.0))"
        })
    elif avg_w < 64 or avg_h < 64:
        suggestions.append({
            "technique": "Bicubic Upscale to ≥224×224",
            "reason": f"Very small images ({avg_w:.0f}×{avg_h:.0f}px) — upscale before augmenting to avoid pixelation artefacts.",
            "priority": "High",
            "code_hint": "transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BICUBIC)"
        })
    else:
        suggestions.append({
            "technique": "Random Resized Crop",
            "reason": "Crops force the model to learn from different spatial regions within each image.",
            "priority": "Medium",
            "code_hint": "transforms.RandomResizedCrop(224, scale=(0.8, 1.0))"
        })

    # ── Colour / photometric ──────────────────────────────────────────────────
    if not is_grayscale:
        suggestions.append({
            "technique": "Color Jitter (brightness, contrast, saturation)",
            "reason": "Simulates varying lighting, exposure, and colour temperature across images.",
            "priority": "Medium",
            "code_hint": "transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)"
        })
        suggestions.append({
            "technique": "Random Grayscale",
            "reason": "Prevents the CNN from relying solely on colour — improves texture-based learning.",
            "priority": "Low",
            "code_hint": "transforms.RandomGrayscale(p=0.1)"
        })
        suggestions.append({
            "technique": "Normalize (ImageNet statistics)",
            "reason": "Standardizes inputs to match ImageNet pretrained weight distributions for fine-tuning.",
            "priority": "High",
            "code_hint": "transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"
        })
    else:
        suggestions.append({
            "technique": "Random AutoContrast",
            "reason": "Grayscale dataset — autocontrast stretches histogram to improve feature visibility.",
            "priority": "Medium",
            "code_hint": "transforms.RandomAutocontrast(p=0.5)"
        })
        suggestions.append({
            "technique": "Normalize (Grayscale)",
            "reason": "Zero-centres grayscale pixel distribution for stable training.",
            "priority": "High",
            "code_hint": "transforms.Normalize(mean=[0.5], std=[0.5])"
        })

    # ── Noise / blur ──────────────────────────────────────────────────────────
    suggestions.append({
        "technique": "Gaussian Blur (random kernel)",
        "reason": "Simulates lens defocus and sensor blur — important for models deployed in real environments.",
        "priority": "Low",
        "code_hint": "transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))"
    })

    # ── Deduplicate & sort ────────────────────────────────────────────────────
    seen = set()
    deduped = []
    for s in suggestions:
        if s["technique"] not in seen:
            seen.add(s["technique"])
            deduped.append(s)
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    deduped.sort(key=lambda x: priority_order.get(x["priority"], 3))
    return deduped


def _suggest_augmentations_for_single_image(image_stats: dict) -> list:
    """
    Generate content-aware augmentation suggestions from a single image's pixel statistics.
    Every suggestion is driven by measurable properties of that specific image.
    """
    suggestions = []

    w = image_stats.get("width", 224)
    h = image_stats.get("height", 224)
    is_gray = image_stats.get("is_grayscale", False)
    aspect_ratio = image_stats.get("aspect_ratio", 1.0)
    overall_mean = image_stats.get("overall_mean", 128.0)
    overall_std = image_stats.get("overall_std", 50.0)
    mean_r = image_stats.get("mean_red", overall_mean)
    mean_g = image_stats.get("mean_green", overall_mean)
    mean_b = image_stats.get("mean_blue", overall_mean)

    # ── Geometric ─────────────────────────────────────────────────────────────
    suggestions.append({
        "technique": "Random Horizontal Flip",
        "reason": "Universal baseline — doubles training samples with no distortion.",
        "priority": "High",
        "code_hint": "transforms.RandomHorizontalFlip(p=0.5)"
    })

    # Vertical flip only when image is roughly square (not a portrait/landscape doc)
    if 0.65 <= aspect_ratio <= 1.55:
        suggestions.append({
            "technique": "Random Vertical Flip",
            "reason": f"Near-square aspect ratio ({aspect_ratio:.2f}) suggests aerial/medical/texture image — vertical flip is valid.",
            "priority": "Medium",
            "code_hint": "transforms.RandomVerticalFlip(p=0.3)"
        })

    # Rotation strength depends on aspect ratio
    if 0.8 <= aspect_ratio <= 1.25:
        suggestions.append({
            "technique": "Random Rotation (±30°)",
            "reason": f"Square image (ratio {aspect_ratio:.2f}) tolerates large rotation without padding waste.",
            "priority": "High",
            "code_hint": "transforms.RandomRotation(degrees=30)"
        })
    else:
        suggestions.append({
            "technique": "Random Rotation (±15°)",
            "reason": f"Rectangular image (ratio {aspect_ratio:.2f}) — gentle rotation avoids padding artefacts at corners.",
            "priority": "Medium",
            "code_hint": "transforms.RandomRotation(degrees=15)"
        })

    # ── Resolution ────────────────────────────────────────────────────────────
    if w < 64 or h < 64:
        suggestions.append({
            "technique": "Bicubic Upscale to 224×224",
            "reason": f"Image is very small ({w}×{h}px) — upscale to a standard CNN input size before augmenting.",
            "priority": "High",
            "code_hint": "transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BICUBIC)"
        })
    elif w > 512 or h > 512:
        suggestions.append({
            "technique": "Random Resized Crop (scale 0.7–1.0)",
            "reason": f"Large image ({w}×{h}px) — random crops expose diverse regions and reduce compute cost.",
            "priority": "High",
            "code_hint": "transforms.RandomResizedCrop(224, scale=(0.7, 1.0))"
        })
    else:
        suggestions.append({
            "technique": "Random Crop with Padding",
            "reason": f"Image ({w}×{h}px) is standard resolution — padding + crop adds spatial translation invariance.",
            "priority": "Medium",
            "code_hint": "transforms.RandomCrop(min(224, {min(w,h)}), padding=8)"
        })

    # ── Brightness ────────────────────────────────────────────────────────────
    if overall_mean < 60:
        suggestions.append({
            "technique": "Brightness Boost Augmentation",
            "reason": f"Image is very dark (mean pixel = {overall_mean:.0f}/255). Use a brightness range that includes brighter variants for training diversity.",
            "priority": "High",
            "code_hint": "transforms.ColorJitter(brightness=(0.8, 3.0))"
        })
    elif overall_mean < 100:
        suggestions.append({
            "technique": "Brightness Jitter (lean bright)",
            "reason": f"Image is underexposed (mean = {overall_mean:.0f}/255) — bias augmentation toward brighter samples.",
            "priority": "High",
            "code_hint": "transforms.ColorJitter(brightness=(0.6, 2.0))"
        })
    elif overall_mean > 210:
        suggestions.append({
            "technique": "Brightness Jitter (lean dark)",
            "reason": f"Image is overexposed (mean = {overall_mean:.0f}/255) — include darker augmentations to balance exposure range.",
            "priority": "High",
            "code_hint": "transforms.ColorJitter(brightness=(0.2, 0.9))"
        })
    elif overall_mean > 160:
        suggestions.append({
            "technique": "Brightness & Contrast Jitter",
            "reason": f"Bright image (mean = {overall_mean:.0f}/255) — symmetric brightness and contrast variation covers exposure spectrum.",
            "priority": "Medium",
            "code_hint": "transforms.ColorJitter(brightness=0.4, contrast=0.4)"
        })
    else:
        suggestions.append({
            "technique": "Brightness & Contrast Jitter",
            "reason": f"Well-exposed image (mean = {overall_mean:.0f}/255) — ±30% brightness/contrast variation is sufficient.",
            "priority": "Medium",
            "code_hint": "transforms.ColorJitter(brightness=0.3, contrast=0.3)"
        })

    # ── Contrast ──────────────────────────────────────────────────────────────
    if overall_std < 25:
        suggestions.append({
            "technique": "Contrast Enhancement (RandomAutocontrast)",
            "reason": f"Very low pixel variance (std = {overall_std:.1f}) — the image is nearly flat. Strong contrast augmentation is critical.",
            "priority": "High",
            "code_hint": "transforms.RandomAutocontrast(p=0.7)"
        })
    elif overall_std < 50:
        suggestions.append({
            "technique": "Contrast Jitter",
            "reason": f"Below-average contrast (std = {overall_std:.1f}) — boosting contrast range improves edge feature learning.",
            "priority": "Medium",
            "code_hint": "transforms.ColorJitter(contrast=(0.5, 2.0))"
        })
    elif overall_std > 90:
        suggestions.append({
            "technique": "Random Equalization",
            "reason": f"Very high pixel variance (std = {overall_std:.1f}) — histogram equalization helps the model handle high-dynamic-range images.",
            "priority": "Low",
            "code_hint": "transforms.RandomEqualize(p=0.3)"
        })

    # ── Colour cast / saturation (colour images only) ─────────────────────────
    if not is_gray:
        channel_means = [mean_r, mean_g, mean_b]
        channel_spread = max(channel_means) - min(channel_means)
        dominant_idx = channel_means.index(max(channel_means))
        dominant_name = ["Red", "Green", "Blue"][dominant_idx]

        if channel_spread > 60:
            suggestions.append({
                "technique": "Hue & Saturation Jitter",
                "reason": f"Strong {dominant_name} colour cast detected (R={mean_r:.0f} G={mean_g:.0f} B={mean_b:.0f}, spread={channel_spread:.0f}). Hue jitter prevents the model from depending on this specific cast.",
                "priority": "High",
                "code_hint": "transforms.ColorJitter(hue=0.25, saturation=0.5)"
            })
        elif channel_spread > 30:
            suggestions.append({
                "technique": "Saturation Jitter",
                "reason": f"Mild {dominant_name} channel dominance (spread={channel_spread:.0f}) — saturation augmentation handles varying white-balance conditions.",
                "priority": "Medium",
                "code_hint": "transforms.ColorJitter(saturation=0.35, hue=0.1)"
            })
        else:
            suggestions.append({
                "technique": "Mild Saturation Jitter",
                "reason": f"Balanced colour channels (R={mean_r:.0f} G={mean_g:.0f} B={mean_b:.0f}) — light saturation variation covers imaging sensor differences.",
                "priority": "Low",
                "code_hint": "transforms.ColorJitter(saturation=0.2)"
            })

        suggestions.append({
            "technique": "Random Grayscale",
            "reason": "Forces the model to learn from texture/shape rather than colour alone — essential for robustness.",
            "priority": "Low",
            "code_hint": "transforms.RandomGrayscale(p=0.1)"
        })
        suggestions.append({
            "technique": "Normalize (ImageNet statistics)",
            "reason": "Aligns pixel distribution with ImageNet pretrained weights — required for transfer learning.",
            "priority": "High",
            "code_hint": "transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
        })
    else:
        suggestions.append({
            "technique": "Random Invert",
            "reason": "Grayscale image — pixel inversion simulates negative/X-ray variants which commonly appear in medical/document datasets.",
            "priority": "Medium",
            "code_hint": "transforms.RandomInvert(p=0.2)"
        })
        suggestions.append({
            "technique": "Random AutoContrast",
            "reason": "Grayscale histogram stretching improves feature visibility across varying scan/lighting conditions.",
            "priority": "Medium",
            "code_hint": "transforms.RandomAutocontrast(p=0.5)"
        })
        suggestions.append({
            "technique": "Normalize (Grayscale)",
            "reason": "Standardizes pixel distribution to zero mean and unit variance for stable gradient flow.",
            "priority": "High",
            "code_hint": "transforms.Normalize(mean=[0.5], std=[0.5])"
        })

    # ── Noise / sharpness ─────────────────────────────────────────────────────
    if overall_std < 40:
        suggestions.append({
            "technique": "Gaussian Blur + Random Noise",
            "reason": f"Low texture variance (std = {overall_std:.1f}) — adding synthetic noise combats over-smoothed, homogeneous inputs.",
            "priority": "Medium",
            "code_hint": "transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0))"
        })
    else:
        suggestions.append({
            "technique": "Random Sharpness Adjustment",
            "reason": f"Good texture detail (std = {overall_std:.1f}) — sharpness augmentation simulates varying camera focus.",
            "priority": "Low",
            "code_hint": "transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3)"
        })

    # ── Deduplicate & sort ────────────────────────────────────────────────────
    seen = set()
    deduped = []
    for s in suggestions:
        if s["technique"] not in seen:
            seen.add(s["technique"])
            deduped.append(s)
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

        # Extract base64 previews of up to 3 sample images so LLaVA can analyse them later
        sample_b64_previews: List[Dict] = []
        try:
            with zipfile.ZipFile(io.BytesIO(contents)) as zf2:
                collected = 0
                for name in zf2.namelist():
                    ext2 = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
                    if ext2 not in image_extensions:
                        continue
                    try:
                        with zf2.open(name) as img_file2:
                            img2 = Image.open(img_file2)
                            preview_b64 = _pil_to_base64(img2.convert("RGB"))
                            parts2 = [p for p in name.replace("\\", "/").split("/") if p]
                            label2 = parts2[-2] + "/" + parts2[-1] if len(parts2) >= 2 else name
                            sample_b64_previews.append({"label": label2, "b64": preview_b64})
                            collected += 1
                            if collected >= 3:
                                break
                    except Exception:
                        continue
        except Exception:
            pass

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
            "ai_augmentation_commentary": ai_commentary,
            "sample_b64_previews": sample_b64_previews,
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
    """Return augmentation suggestions for an image dataset (ZIP, single, or multi)."""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    ds = datasets[dataset_id]
    ds_type = ds.get("dataset_type", "")
    if ds_type not in ("image", "single_image", "multi_image"):
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
        
        # Keep sample payload compact to reduce Gemini token usage
        sample_cols = list(df.columns)[:12]
        sample_data = df[sample_cols].head(3).to_dict('records')
        sample_data = convert_to_json_serializable(sample_data)
        compact_schema = _build_compact_columns_context(df, eda, max_cols=25)
        
        prompt = f"""You are a data analyst assistant. Answer questions about the dataset conversationally and accurately.

Dataset: {filename}
Rows: {len(df):,}, Columns: {len(df.columns)}

Column Information (compact):
{compact_schema}

Sample Data (first 3 rows, max 12 columns):
{json.dumps(sample_data, indent=2)}
{history_context}

User Question: {request.message}

Provide a clear, helpful, and accurate answer based on the dataset information above. If making calculations or observations, be specific and cite numbers from the data."""
        
        response = get_gemini_api_response(prompt)
        if response.startswith("Error:"):
            raise HTTPException(500, response)
        
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
        
        # Build compact schema context to avoid token-heavy requests
        compact_schema = _build_compact_columns_context(df, eda, max_cols=30)

        # Generate pandas query using AI
        prompt = f"""Convert this natural language query into pandas operations.

Dataset Columns: {list(df.columns)}
    Column Info (compact):
    {compact_schema}

Query: {request.query}

Provide ONLY the pandas code to execute this query. Use 'df' as the dataframe variable.
Return code that filters, groups, aggregates, or transforms the data as requested.

Example formats:
- "Show me rows where age > 30" → df[df['age'] > 30]
- "Average salary by department" → df.groupby('department')['salary'].mean()
- "Top 10 highest scores" → df.nlargest(10, 'score')

Only return the code, nothing else."""
        
        pandas_code = get_gemini_api_response(prompt).strip()
        if pandas_code.startswith("Error:"):
            raise HTTPException(500, pandas_code)
        
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

# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE IMAGE UPLOAD  (analyze and show augmentation previews)
# ─────────────────────────────────────────────────────────────────────────────

def _pil_to_base64(img: "Image.Image", fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _apply_augmentations_to_pil(img: "Image.Image") -> List[Dict[str, str]]:
    """Apply common augmentations to a PIL image and return base64 previews."""
    import random

    previews = []

    # Resize to max 300px for preview
    MAX_PREVIEW = 300
    w, h = img.size
    if max(w, h) > MAX_PREVIEW:
        scale = MAX_PREVIEW / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Ensure RGB for display (in case RGBA or P)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # 1 – Original
    previews.append({"name": "Original", "b64": _pil_to_base64(img)})

    # 2 – Horizontal Flip
    previews.append({"name": "Horizontal Flip", "b64": _pil_to_base64(img.transpose(Image.FLIP_LEFT_RIGHT))})

    # 3 – Vertical Flip
    previews.append({"name": "Vertical Flip", "b64": _pil_to_base64(img.transpose(Image.FLIP_TOP_BOTTOM))})

    # 4 – Rotation 30°
    previews.append({"name": "Rotation 30°", "b64": _pil_to_base64(img.rotate(30, expand=False, fillcolor=0))})

    # 5 – Rotation -30°
    previews.append({"name": "Rotation -30°", "b64": _pil_to_base64(img.rotate(-30, expand=False, fillcolor=0))})

    # 6 – Brightness boost (+70)
    try:
        from PIL import ImageEnhance
        b_img = ImageEnhance.Brightness(img).enhance(1.6)
        previews.append({"name": "Brightness Boost", "b64": _pil_to_base64(b_img)})
    except Exception:
        pass

    # 7 – Contrast boost
    try:
        from PIL import ImageEnhance
        c_img = ImageEnhance.Contrast(img).enhance(1.8)
        previews.append({"name": "Contrast Boost", "b64": _pil_to_base64(c_img)})
    except Exception:
        pass

    # 8 – Saturation boost (color jitter)
    try:
        from PIL import ImageEnhance
        s_img = ImageEnhance.Color(img).enhance(2.0)
        previews.append({"name": "Saturation Boost", "b64": _pil_to_base64(s_img)})
    except Exception:
        pass

    # 9 – Grayscale
    previews.append({"name": "Grayscale", "b64": _pil_to_base64(img.convert("L").convert("RGB"))})

    # 10 – Center Crop (80%)
    if img.width > 30 and img.height > 30:
        cw, ch = int(img.width * 0.8), int(img.height * 0.8)
        left = (img.width - cw) // 2
        top = (img.height - ch) // 2
        cropped = img.crop((left, top, left + cw, top + ch))
        previews.append({"name": "Center Crop 80%", "b64": _pil_to_base64(cropped)})

    # 11 – Gaussian Blur
    try:
        from PIL import ImageFilter
        blurred = img.filter(ImageFilter.GaussianBlur(radius=2))
        previews.append({"name": "Gaussian Blur", "b64": _pil_to_base64(blurred)})
    except Exception:
        pass

    # 12 – Sharpness boost
    try:
        from PIL import ImageEnhance
        sharp = ImageEnhance.Sharpness(img).enhance(3.0)
        previews.append({"name": "Sharpness Boost", "b64": _pil_to_base64(sharp)})
    except Exception:
        pass

    return previews


def _analyze_single_image(img: "Image.Image") -> Dict[str, Any]:
    """Return basic stats for a PIL image."""
    w, h = img.size
    mode = img.mode
    channels = len(img.getbands())

    stats: Dict[str, Any] = {
        "width": w,
        "height": h,
        "aspect_ratio": round(w / max(1, h), 3),
        "mode": mode,
        "channels": channels,
        "is_grayscale": mode in ("L", "LA", "1"),
        "total_pixels": w * h,
    }

    # Per-channel mean/std
    try:
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        for i, ch_name in enumerate(["Red", "Green", "Blue"]):
            stats[f"mean_{ch_name.lower()}"] = round(float(np.mean(arr[:, :, i])), 2)
            stats[f"std_{ch_name.lower()}"] = round(float(np.std(arr[:, :, i])), 2)
        stats["overall_mean"] = round(float(np.mean(arr)), 2)
        stats["overall_std"] = round(float(np.std(arr)), 2)
    except Exception:
        pass

    return stats


@app.post("/api/upload-single-image")
async def upload_single_image(file: UploadFile = File(...)):
    """
    Upload a single image.
    Returns: dataset_id, image stats, augmentation previews (base64), and AI suggestions.
    """
    allowed_types = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}
    ext = "." + (file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "")
    if ext not in allowed_types:
        raise HTTPException(400, f"Unsupported image format: {ext}. Use JPG, PNG, BMP, TIFF, WEBP.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Analyse image
        image_stats = _analyze_single_image(img)

        # Build augmentation previews
        previews = _apply_augmentations_to_pil(img)

        # Build augmentation suggestions based on actual pixel content
        augmentation_suggestions = _suggest_augmentations_for_single_image(image_stats)

        # AI commentary
        ai_commentary = ""
        try:
            ai_prompt = f"""You are a computer vision expert. Analyze this single image and suggest data augmentation techniques.

Image: {file.filename}
Dimensions: {image_stats['width']}x{image_stats['height']}px
Color mode: {image_stats['mode']} ({image_stats['channels']} channel(s))
Grayscale: {image_stats.get('is_grayscale', False)}
Pixel mean (R/G/B): {image_stats.get('mean_red', '?')}/{image_stats.get('mean_green', '?')}/{image_stats.get('mean_blue', '?')}
Pixel std (R/G/B): {image_stats.get('std_red', '?')}/{image_stats.get('std_green', '?')}/{image_stats.get('std_blue', '?')}

Give 3-4 concise sentences: what kind of image this appears to be (medical, natural scene, document, satellite, etc.) and which augmentations would be most valuable for ML training. No markdown. Plain text only."""
            ai_commentary = get_gemini_response(ai_prompt, "lite")
        except Exception:
            ai_commentary = "AI commentary unavailable."

        # Store in datasets
        dataset_id = str(uuid.uuid4())
        datasets[dataset_id] = {
            "df": pd.DataFrame(),
            "df_sample": pd.DataFrame(),
            "filename": file.filename,
            "uploaded_at": datetime.now().isoformat(),
            "primary_keys": [],
            "is_sampled": False,
            "profile": {},
            "analysis_cache": {},
            "dataset_type": "single_image",
            "image_stats": image_stats,
            "augmentation_suggestions": augmentation_suggestions,
            "augmentation_previews": previews,
            "ai_augmentation_commentary": ai_commentary,
        }
        eda_results[dataset_id] = {"columns": {}, "dataset_type": "single_image"}

        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "dataset_type": "single_image",
            "stats": image_stats,
            "augmentation_suggestions": augmentation_suggestions,
            "augmentation_previews": previews,
            "ai_commentary": ai_commentary,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single image upload error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Single image upload failed: {str(e)}")


@app.get("/api/image-analysis/{dataset_id}")
async def get_single_image_analysis(dataset_id: str):
    """Return full image analysis including previews for a single-image or multi-image dataset."""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    ds = datasets[dataset_id]
    if ds.get("dataset_type") not in ("single_image", "multi_image"):
        raise HTTPException(400, "This endpoint is for single/multi-image datasets only.")
    return {
        "dataset_id": dataset_id,
        "filename": ds["filename"],
        "dataset_type": ds.get("dataset_type", "single_image"),
        "stats": ds.get("image_stats", {}),
        "augmentation_suggestions": ds.get("augmentation_suggestions", []),
        "augmentation_previews": ds.get("augmentation_previews", []),
        "ai_commentary": ds.get("ai_augmentation_commentary", ""),
    }


def _parse_insight_sections(raw_text: str) -> list:
    """Parse 'SECTION: title / - bullet' formatted text into structured sections."""
    sections = []
    current_title = None
    current_items: List[str] = []

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("SECTION:"):
            if current_title and current_items:
                sections.append({"title": current_title, "items": current_items})
            current_title = line[8:].strip().strip("*").strip()
            current_items = []
        elif line.startswith("- ") or line.startswith("• ") or line.startswith("* "):
            current_items.append(line[2:].strip())
        elif re.match(r'^\d+\.\s', line):
            current_items.append(re.sub(r'^\d+\.\s*', '', line).strip())

    if current_title and current_items:
        sections.append({"title": current_title, "items": current_items})

    # Fallback: wrap everything in a single section
    if not sections and raw_text.strip():
        items = [l.strip("- •*").strip() for l in raw_text.splitlines()
                 if l.strip() and l.strip() not in ["-", "*", "•"]]
        if items:
            sections = [{"title": "AI Analysis", "items": items[:8]}]

    return sections


def _get_domain_specific_augmentations(domain: str, stats: dict) -> list:
    """Return only the augmentations relevant to the detected visual domain (max 6)."""
    total     = stats.get("total_images", 0)
    imbalance = stats.get("class_imbalance_ratio", 1.0)
    avg_w     = stats.get("avg_width", 224)
    avg_h     = stats.get("avg_height", 224)
    is_gray   = stats.get("is_predominantly_grayscale", False)

    norm_std  = "transforms.Normalize(mean=[0.5], std=[0.5])" if is_gray else \
                "transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"

    domain_strategies: Dict[str, list] = {
        "medical_imaging": [
            {"technique": "Random Rotation (±15°)", "reason": "Medical images may appear at slight angles; small rotation adds robustness.", "priority": "High", "code_hint": "transforms.RandomRotation(degrees=15)"},
            {"technique": "Elastic Deformation", "reason": "Simulates tissue deformation — critical for robust medical segmentation.", "priority": "High", "code_hint": "from torchvision.transforms import ElasticTransform; ElasticTransform(alpha=50.0)"},
            {"technique": "Random Contrast Enhancement", "reason": "Medical scans often have poor contrast — augmenting it improves model robustness.", "priority": "High", "code_hint": "transforms.RandomAutocontrast(p=0.5)"},
            {"technique": "Normalize", "reason": "Required for stable training. Using grayscale stats." if is_gray else "Required for pretrained backbone fine-tuning.", "priority": "High", "code_hint": norm_std},
            {"technique": "Skip Color Jitter", "reason": "Color is diagnostically meaningful in medical images — DO NOT apply color jitter.", "priority": "High", "code_hint": "# Omit ColorJitter for medical datasets"},
        ],
        "satellite_aerial": [
            {"technique": "Random H+V Flip", "reason": "Aerial imagery has no canonical orientation — both flips are equally valid.", "priority": "High", "code_hint": "transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)"},
            {"technique": "Random Rotation (90° steps)", "reason": "Satellite imagery is rotation-invariant — 90° steps avoid padding artifacts.", "priority": "High", "code_hint": "transforms.RandomRotation(degrees=[0,90,180,270])"},
            {"technique": "Random Resized Crop (0.5–1.0)", "reason": "Forces the model to detect features at multiple spatial scales.", "priority": "High", "code_hint": "transforms.RandomResizedCrop(224, scale=(0.5, 1.0))"},
            {"technique": "Light Color Jitter", "reason": "Accounts for atmospheric haze and seasonal color variation.", "priority": "Medium", "code_hint": "transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)"},
            {"technique": "Normalize (ImageNet)", "reason": "Standard normalization for ResNet/EfficientNet fine-tuning.", "priority": "High", "code_hint": "transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"},
        ],
        "natural_scenes": [
            {"technique": "Random Horizontal Flip", "reason": "Natural scenes are horizontally symmetric — safe and highly effective.", "priority": "High", "code_hint": "transforms.RandomHorizontalFlip(p=0.5)"},
            {"technique": "Color Jitter (medium)", "reason": "Natural lighting varies widely — simulate dawn, dusk, and overcast conditions.", "priority": "High", "code_hint": "transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1)"},
            {"technique": "Random Resized Crop (0.6–1.0)", "reason": "Encourage scale-invariant feature learning across scene regions.", "priority": "High", "code_hint": "transforms.RandomResizedCrop(224, scale=(0.6, 1.0))"},
            {"technique": "Normalize (ImageNet)", "reason": "Match distribution of pretrained models for fine-tuning.", "priority": "High", "code_hint": "transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"},
        ],
        "faces_portraits": [
            {"technique": "Random Horizontal Flip", "reason": "Faces are bilaterally symmetric — horizontal flip is safe.", "priority": "High", "code_hint": "transforms.RandomHorizontalFlip(p=0.5)"},
            {"technique": "Light Color Jitter", "reason": "Simulates different lighting and camera exposure across skin tones.", "priority": "Medium", "code_hint": "transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2)"},
            {"technique": "Random Resized Crop (0.85–1.0)", "reason": "Ensures face features are centered; robust to slight misalignment.", "priority": "High", "code_hint": "transforms.RandomResizedCrop(224, scale=(0.85, 1.0))"},
            {"technique": "Skip Vertical Flip", "reason": "Vertical flip of faces creates unnatural images that harm learning.", "priority": "High", "code_hint": "# Omit RandomVerticalFlip for face datasets"},
            {"technique": "Normalize (ImageNet)", "reason": "Standard normalization for FaceNet/ResNet/EfficientNet.", "priority": "High", "code_hint": "transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"},
        ],
        "documents": [
            {"technique": "Random Rotation (±5°)", "reason": "Document scans may be slightly skewed — small rotation adds robustness.", "priority": "High", "code_hint": "transforms.RandomRotation(degrees=5)"},
            {"technique": "Random Perspective", "reason": "Simulates perspective distortion from scanning at an angle.", "priority": "Medium", "code_hint": "transforms.RandomPerspective(distortion_scale=0.2, p=0.3)"},
            {"technique": "Random Grayscale", "reason": "Documents appear in color or grayscale scans — improve cross-modality generalization.", "priority": "Medium", "code_hint": "transforms.RandomGrayscale(p=0.3)"},
            {"technique": "Skip Color Jitter", "reason": "Text/document color is semantically meaningful — avoid color distortion.", "priority": "High", "code_hint": "# Omit ColorJitter for document datasets"},
            {"technique": "Normalize (Standard)", "reason": "Required for stable training with pretrained models.", "priority": "High", "code_hint": "transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"},
        ],
        "industrial": [
            {"technique": "Random Rotation (full 360°)", "reason": "Industrial parts can appear at any orientation on inspection lines.", "priority": "High", "code_hint": "transforms.RandomRotation(degrees=360)"},
            {"technique": "Random Erasing (simulate defects)", "reason": "Randomly erase patches to simulate occlusion and partial defect visibility.", "priority": "High", "code_hint": "transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))"},
            {"technique": "Gaussian Blur", "reason": "Simulates camera defocus and motion blur in industrial inspection setups.", "priority": "Medium", "code_hint": "transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))"},
            {"technique": "Normalize (Standard)", "reason": "Required for pretrained backbone fine-tuning.", "priority": "High", "code_hint": "transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"},
        ],
        "food_beverage": [
            {"technique": "Random Horizontal Flip", "reason": "Food images have no canonical orientation.", "priority": "High", "code_hint": "transforms.RandomHorizontalFlip(p=0.5)"},
            {"technique": "Color Jitter (medium-high)", "reason": "Lighting strongly affects food appearance — simulate restaurant vs. natural light.", "priority": "High", "code_hint": "transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.4, hue=0.05)"},
            {"technique": "Random Rotation (±30°)", "reason": "Food photos are captured at varied angles.", "priority": "Medium", "code_hint": "transforms.RandomRotation(degrees=30)"},
            {"technique": "Normalize (ImageNet)", "reason": "Standard normalization for pretrained food classification models.", "priority": "High", "code_hint": "transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"},
        ],
        "animals_wildlife": [
            {"technique": "Random Horizontal Flip", "reason": "Animals can face either direction — horizontal flip is safe and effective.", "priority": "High", "code_hint": "transforms.RandomHorizontalFlip(p=0.5)"},
            {"technique": "Color Jitter (medium)", "reason": "Simulates varying lighting in wildlife photography.", "priority": "High", "code_hint": "transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3)"},
            {"technique": "Random Resized Crop (0.5–1.0)", "reason": "Animals appear at various scales — force scale-invariant learning.", "priority": "High", "code_hint": "transforms.RandomResizedCrop(224, scale=(0.5, 1.0))"},
            {"technique": "Normalize (ImageNet)", "reason": "Standard normalization for fine-tuning on nature/wildlife datasets.", "priority": "High", "code_hint": "transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"},
        ],
        "objects_products": [
            {"technique": "Random Horizontal Flip", "reason": "Product images have no fixed orientation.", "priority": "High", "code_hint": "transforms.RandomHorizontalFlip(p=0.5)"},
            {"technique": "Color Jitter (light)", "reason": "Simulates varying e-commerce lighting and white balance.", "priority": "Medium", "code_hint": "transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2)"},
            {"technique": "Random Resized Crop", "reason": "Products appear at various scales; crop encourages scale robustness.", "priority": "High", "code_hint": "transforms.RandomResizedCrop(224, scale=(0.7, 1.0))"},
            {"technique": "Normalize (ImageNet)", "reason": "Standard normalization for pretrained product classifiers.", "priority": "High", "code_hint": "transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"},
        ],
    }

    domain_key = domain.lower().replace(" ", "_").replace("-", "_")
    base = domain_strategies.get(domain_key, [
        {"technique": "Random Horizontal Flip", "reason": "Standard baseline — doubles samples with no distortion.", "priority": "High", "code_hint": "transforms.RandomHorizontalFlip(p=0.5)"},
        {"technique": "Color Jitter", "reason": "Simulates varying lighting conditions.", "priority": "Medium", "code_hint": "transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)"},
        {"technique": "Random Resized Crop", "reason": "Forces scale-invariant feature learning.", "priority": "Medium", "code_hint": "transforms.RandomResizedCrop(224, scale=(0.7, 1.0))"},
        {"technique": "Normalize (ImageNet)", "reason": "Standard normalization for pretrained models.", "priority": "High", "code_hint": "transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"},
    ])

    extra: list = []
    if imbalance > 3.0:
        extra.append({"technique": "WeightedRandomSampler", "reason": f"Class imbalance {imbalance:.1f}× — weighted sampling ensures balanced class exposure per epoch.", "priority": "High", "code_hint": "torch.utils.data.WeightedRandomSampler(weights, num_samples=len(dataset))"})
    if total < 500:
        extra.append({"technique": "MixUp / CutMix", "reason": f"Only {total} images — MixUp/CutMix create powerful synthetic training samples.", "priority": "High", "code_hint": "# torchvision.transforms.v2.MixUp() or CutMix()"})
    if avg_w > 512 or avg_h > 512:
        extra.append({"technique": f"Resize to 224×224", "reason": f"Images are {avg_w:.0f}×{avg_h:.0f}px — resize reduces memory and speeds up training.", "priority": "High", "code_hint": "transforms.Resize((224, 224))"})

    seen = {e["technique"] for e in extra}
    combined = extra + [s for s in base if s["technique"] not in seen]
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    combined.sort(key=lambda x: priority_order.get(x["priority"], 3))
    return combined[:6]


@app.get("/api/image-ai-insights/{dataset_id}")
async def get_image_ai_insights(dataset_id: str):
    """
    Use LLaVA to analyse up to 2 representative sample images, detect visual domain,
    and return structured insights (same section format as the CSV insights tab) plus
    domain-specific augmentation recommendations.  Only 1 LLaVA call per image.
    """
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    ds = datasets[dataset_id]
    ds_type = ds.get("dataset_type", "")
    if ds_type not in ("image", "single_image", "multi_image"):
        raise HTTPException(400, "This endpoint is for image datasets only.")

    cache = ds.setdefault("analysis_cache", {})
    if "image_ai_insights" in cache:
        return cache["image_ai_insights"]

    # ── Collect up to 2 sample images ──────────────────────────────────────
    previews = ds.get("augmentation_previews", [])
    sample_images_b64: List[Dict] = []

    if ds_type == "single_image":
        orig = next((p for p in previews if p.get("name") == "Original"), None)
        if orig:
            sample_images_b64.append({"label": ds.get("filename", "image"), "b64": orig["b64"]})
    elif ds_type == "multi_image":
        for batch in previews[:2]:
            orig = next((p for p in batch.get("previews", []) if p.get("name") == "Original"), None)
            if orig:
                sample_images_b64.append({"label": batch.get("filename", "image"), "b64": orig["b64"]})
    elif ds_type == "image":
        for item in ds.get("sample_b64_previews", [])[:2]:
            sample_images_b64.append(item)

    if not sample_images_b64:
        return {
            "dataset_id": dataset_id,
            "domain": "unknown",
            "overall_description": "No sample images available for vision analysis.",
            "insights": [],
            "augmentation_tips": [],
        }

    # ── Single combined LLaVA call per image ───────────────────────────────
    all_domains: List[str] = []
    all_scenes:  List[str] = []
    all_visuals: List[str] = []

    for item in sample_images_b64:
        b64 = item.get("b64", "")
        if not b64:
            continue
        prompt = (
            "Analyze this image. Respond on exactly 3 lines:\n"
            "DOMAIN: [one of: medical_imaging, satellite_aerial, natural_scenes, documents, "
            "industrial, food_beverage, fashion_apparel, sports_action, faces_portraits, "
            "animals_wildlife, objects_products, microscopy, art_drawings, other]\n"
            "SCENE: [what is in this image — 1 to 2 sentences]\n"
            "VISUAL: [lighting, color profile, texture complexity — 1 sentence]\n"
            "Respond ONLY in that 3-line format, no extra text."
        )
        response = get_ollama_vision_response(prompt, b64)
        dm = re.search(r'DOMAIN:\s*(.+)',  response, re.IGNORECASE)
        sc = re.search(r'SCENE:\s*(.+)',   response, re.IGNORECASE)
        vi = re.search(r'VISUAL:\s*(.+)',  response, re.IGNORECASE)
        if dm: all_domains.append(dm.group(1).strip().lower().replace(" ", "_").split(",")[0])
        if sc: all_scenes.append(sc.group(1).strip())
        if vi: all_visuals.append(vi.group(1).strip())

    dominant_domain = max(set(all_domains), key=all_domains.count) if all_domains else "other"
    combined_scene  = " ".join(all_scenes)
    combined_visual = " ".join(all_visuals)

    # ── Structured insights via text model ─────────────────────────────────
    img_stats   = ds.get("image_stats", {})
    class_dist  = img_stats.get("class_distribution", {})
    total_imgs  = img_stats.get("total_images", len(sample_images_b64))
    n_classes   = len(class_dist) or 1
    imbalance   = img_stats.get("class_imbalance_ratio", 1.0)
    avg_w       = img_stats.get("avg_width", "?")
    avg_h       = img_stats.get("avg_height", "?")
    color_mode  = "grayscale" if img_stats.get("is_predominantly_grayscale") else "color"
    classes_str = str(list(class_dist.keys())[:8])

    insight_prompt = (
        f"You are a computer vision data scientist analyzing a dataset.\n"
        f"Dataset facts: {total_imgs} images, {n_classes} class(es) {classes_str}, "
        f"visual domain: {dominant_domain.replace('_', ' ')}, "
        f"avg size {avg_w}x{avg_h}px, {color_mode}, imbalance ratio {imbalance:.1f}x.\n"
        f"Image content: {combined_scene}\n"
        f"Visual characteristics: {combined_visual}\n\n"
        f"Generate exactly 4 insight sections using this format:\n"
        f"SECTION: <title>\n"
        f"- <bullet point 1>\n"
        f"- <bullet point 2>\n"
        f"- <bullet point 3>\n\n"
        f"Use these exact section titles:\n"
        f"1. Dataset Overview\n"
        f"2. Visual Characteristics\n"
        f"3. CV Task Recommendations\n"
        f"4. Preprocessing Recommendations\n\n"
        f"Each section: 3 bullet points, specific to this dataset's domain and stats. No extra text."
    )
    raw_insights = get_gemini_response(insight_prompt)
    insights = _parse_insight_sections(raw_insights)

    # ── Domain-specific augmentation tips ──────────────────────────────────
    aug_tips = _get_domain_specific_augmentations(dominant_domain, img_stats)

    result = {
        "dataset_id": dataset_id,
        "domain": dominant_domain,
        "overall_description": combined_scene,
        "insights": insights,
        "augmentation_tips": aug_tips,
    }
    cache["image_ai_insights"] = result
    return result


@app.post("/api/upload-multiple-images")
async def upload_multiple_images(files: List[UploadFile] = File(...)):
    """
    Upload multiple image files at once (no ZIP needed).
    File naming convention for class labels:  classname_anything.ext  OR  classname/anything.ext
    Returns: dataset_id, per-image stats, augmentation previews (up to 3 images), and suggestions.
    """
    allowed_types = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}
    if not files:
        raise HTTPException(400, "No files uploaded.")

    valid_files = []
    for f in files:
        ext = "." + (f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else "")
        if ext in allowed_types:
            valid_files.append((f, ext))

    if not valid_files:
        raise HTTPException(400, "No valid image files found. Supported: JPG, PNG, BMP, TIFF, WEBP.")

    try:
        widths, heights, channels_list, means_all, stds_all = [], [], [], [], []
        grayscale_count = 0
        class_counts: Dict[str, int] = {}
        per_image_stats = []
        all_previews = []  # previews from first 3 images
        combined_image_stats: Dict[str, Any] = {}

        for idx, (upload_file, ext) in enumerate(valid_files):
            contents = await upload_file.read()
            try:
                img = Image.open(io.BytesIO(contents))
                stats_i = _analyze_single_image(img)

                widths.append(stats_i["width"])
                heights.append(stats_i["height"])
                channels_list.append(stats_i["channels"])
                if stats_i.get("is_grayscale"):
                    grayscale_count += 1
                if "overall_mean" in stats_i:
                    means_all.append(stats_i["overall_mean"])
                if "overall_std" in stats_i:
                    stds_all.append(stats_i["overall_std"])

                # Infer class label from filename prefix (e.g. "cat_001.jpg" → "cat")
                basename = upload_file.filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                name_without_ext = basename.rsplit(".", 1)[0]
                parts = [p for p in name_without_ext.replace("-", "_").split("_") if p]
                class_label = parts[0].lower() if parts else "unknown"
                class_counts[class_label] = class_counts.get(class_label, 0) + 1

                per_image_stats.append({
                    "filename": upload_file.filename,
                    "class_label": class_label,
                    **stats_i
                })

                # Collect previews for first 3 valid images
                if idx < 3:
                    previews_i = _apply_augmentations_to_pil(img)
                    all_previews.append({
                        "filename": upload_file.filename,
                        "previews": previews_i
                    })

            except Exception as img_err:
                logger.warning(f"Could not process {upload_file.filename}: {img_err}")
                continue

        if not widths:
            raise HTTPException(400, "No valid images could be opened.")

        total = len(widths)
        counts = list(class_counts.values())
        imbalance = round(max(counts) / max(1, min(counts)), 2) if len(counts) > 1 else 1.0
        is_gray_dominant = grayscale_count > total * 0.7

        avg_mean = round(float(np.mean(means_all)), 2) if means_all else 128.0
        avg_std = round(float(np.mean(stds_all)), 2) if stds_all else 50.0

        # Build aggregated image stats
        combined_image_stats = {
            "total_images": total,
            "num_classes": len(class_counts),
            "class_distribution": class_counts,
            "class_imbalance_ratio": imbalance,
            "avg_width": round(float(np.mean(widths)), 1),
            "avg_height": round(float(np.mean(heights)), 1),
            "min_width": int(min(widths)),
            "max_width": int(max(widths)),
            "min_height": int(min(heights)),
            "max_height": int(max(heights)),
            "is_predominantly_grayscale": is_gray_dominant,
            "overall_mean": avg_mean,
            "overall_std": avg_std,
            "width": int(np.mean(widths)),
            "height": int(np.mean(heights)),
            "aspect_ratio": round(float(np.mean(widths)) / max(1.0, float(np.mean(heights))), 3),
            "is_grayscale": is_gray_dominant,
            "channels": int(round(float(np.mean(channels_list)))),
            "mode": "L" if is_gray_dominant else "RGB",
        }

        # Use content-aware suggestions if single-image-like; dataset-level if batch
        if total == 1 and per_image_stats:
            augmentation_suggestions = _suggest_augmentations_for_single_image(per_image_stats[0])
        else:
            # Use average pixel stats to drive content-aware suggestions
            proxy_stats = {**combined_image_stats}
            augmentation_suggestions = _suggest_augmentations_for_single_image(proxy_stats)

        # AI commentary
        ai_commentary = ""
        try:
            class_str = ", ".join(f"{k}: {v}" for k, v in list(class_counts.items())[:10])
            ai_prompt = f"""You are a computer vision expert. Analyse this batch of {total} uploaded images and give 3-4 concise sentences on the best augmentation strategy for ML training.

Images: {total} files
Inferred classes: {class_str}
Avg dimensions: {combined_image_stats['avg_width']:.0f}×{combined_image_stats['avg_height']:.0f}px
Grayscale dominant: {is_gray_dominant}
Mean brightness: {avg_mean:.1f}/255  |  Pixel std: {avg_std:.1f}
Class imbalance: {imbalance:.1f}×

No markdown. Plain text only."""
            ai_commentary = get_gemini_response(ai_prompt, "lite")
        except Exception:
            ai_commentary = "AI commentary unavailable."

        dataset_id = str(uuid.uuid4())
        datasets[dataset_id] = {
            "df": pd.DataFrame(),
            "df_sample": pd.DataFrame(),
            "filename": f"{total} images uploaded",
            "uploaded_at": datetime.now().isoformat(),
            "primary_keys": [],
            "is_sampled": False,
            "profile": {},
            "analysis_cache": {},
            "dataset_type": "multi_image",
            "image_stats": combined_image_stats,
            "per_image_stats": per_image_stats,
            "augmentation_suggestions": augmentation_suggestions,
            "augmentation_previews": all_previews,
            "ai_augmentation_commentary": ai_commentary,
        }
        eda_results[dataset_id] = {"columns": {}, "dataset_type": "multi_image"}

        return {
            "dataset_id": dataset_id,
            "filename": f"{total} images uploaded",
            "dataset_type": "multi_image",
            "total_images": total,
            "stats": combined_image_stats,
            "augmentation_suggestions": augmentation_suggestions,
            "augmentation_previews": all_previews,
            "ai_commentary": ai_commentary,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-image upload error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Multi-image upload failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
#  ML MODEL RECOMMENDATIONS  (tabular datasets)
# ─────────────────────────────────────────────────────────────────────────────

_ML_CATALOG = {
    "classification": {
        "small": [
            {
                "name": "Logistic Regression",
                "description": "Probabilistic linear classifier, fast and interpretable.",
                "pros": ["Very fast to train", "Interpretable coefficients", "Works well for linearly separable data", "Returns calibrated probabilities"],
                "cons": ["Assumes linear decision boundary", "Needs feature scaling", "Struggles with complex interactions"],
                "best_for": "Binary/multi-class classification with numeric features, baseline model",
                "sklearn": "sklearn.linear_model.LogisticRegression",
                "complexity": "Low",
                "eval_metrics": ["Accuracy", "F1 Score", "ROC-AUC", "Log Loss"],
            },
            {
                "name": "Decision Tree",
                "description": "Splits data hierarchically on best features.",
                "pros": ["Highly interpretable", "No scaling required", "Handles mixed feature types", "Fast inference"],
                "cons": ["Prone to overfitting", "Unstable — small data changes alter tree", "Poor generalisation on deep trees"],
                "best_for": "Rule extraction, explainability use-cases",
                "sklearn": "sklearn.tree.DecisionTreeClassifier",
                "complexity": "Low",
                "eval_metrics": ["Accuracy", "F1 Score", "Confusion Matrix"],
            },
            {
                "name": "K-Nearest Neighbors",
                "description": "Classifies based on majority vote of k nearest training points.",
                "pros": ["Simple, no training phase", "Naturally multi-class", "Adapts to complex boundaries"],
                "cons": ["Slow at inference for large datasets", "Sensitive to irrelevant features", "Requires feature scaling"],
                "best_for": "Small datasets with well-defined local structure",
                "sklearn": "sklearn.neighbors.KNeighborsClassifier",
                "complexity": "Low",
                "eval_metrics": ["Accuracy", "F1 Score"],
            },
        ],
        "medium": [
            {
                "name": "Random Forest",
                "description": "Ensemble of decision trees with bagging and random feature selection.",
                "pros": ["Robust to overfitting", "Handles missing values", "Built-in feature importance", "Works on mixed data types"],
                "cons": ["Less interpretable than single tree", "Memory-intensive for large forests", "Slower than linear models"],
                "best_for": "General-purpose classification, mixed numeric/categorical data",
                "sklearn": "sklearn.ensemble.RandomForestClassifier",
                "complexity": "Medium",
                "eval_metrics": ["Accuracy", "F1 Score (macro)", "ROC-AUC", "Feature Importance"],
            },
            {
                "name": "Gradient Boosting (XGBoost-style)",
                "description": "Sequential ensemble that corrects residual errors from previous trees.",
                "pros": ["State-of-the-art on tabular data", "Handles missing values natively", "Regularisation built-in", "Feature importance"],
                "cons": ["Many hyperparameters to tune", "Slower to train than Random Forest", "Can overfit on small datasets"],
                "best_for": "Competitive accuracy on structured/tabular datasets",
                "sklearn": "sklearn.ensemble.GradientBoostingClassifier",
                "complexity": "High",
                "eval_metrics": ["Accuracy", "F1 Score", "ROC-AUC", "Log Loss"],
            },
            {
                "name": "Support Vector Machine (SVM)",
                "description": "Finds maximum-margin hyperplane separating classes.",
                "pros": ["Effective in high-dimensional space", "Kernel trick for non-linear boundaries", "Memory efficient"],
                "cons": ["Does not scale to large datasets", "Feature scaling required", "Slow prediction for large kernels"],
                "best_for": "Text/image features, medium datasets, high-dimensional spaces",
                "sklearn": "sklearn.svm.SVC",
                "complexity": "Medium",
                "eval_metrics": ["Accuracy", "F1 Score", "Confusion Matrix"],
            },
        ],
        "large": [
            {
                "name": "LightGBM / XGBoost",
                "description": "Highly optimised gradient boosting libraries for large-scale data.",
                "pros": ["Fastest among boosting methods", "Scales to millions of rows", "Categorical feature support (LightGBM)", "Top-ranked on Kaggle"],
                "cons": ["Complex hyperparameter tuning", "Less interpretable than trees", "Can overfit without regularisation"],
                "best_for": "Large tabular datasets where accuracy is critical",
                "sklearn": "lightgbm.LGBMClassifier or xgboost.XGBClassifier",
                "complexity": "High",
                "eval_metrics": ["Accuracy", "F1 Score", "ROC-AUC", "Log Loss"],
            },
            {
                "name": "Neural Network (MLP)",
                "description": "Multi-layer perceptron with non-linear activation functions.",
                "pros": ["Can learn complex patterns", "Flexible architecture", "Works with any feature type after encoding"],
                "cons": ["Requires large amounts of data", "Black-box predictions", "Sensitive to scaling and initialisation"],
                "best_for": "Large datasets with complex feature interactions",
                "sklearn": "sklearn.neural_network.MLPClassifier",
                "complexity": "High",
                "eval_metrics": ["Accuracy", "F1 Score", "Loss Curve"],
            },
            {
                "name": "Extra Trees",
                "description": "More randomised variant of Random Forest — faster training.",
                "pros": ["Faster than Random Forest", "Low variance due to full randomisation", "Good on noisy data"],
                "cons": ["Slightly lower accuracy than Random Forest on clean data", "Less common in production"],
                "best_for": "Large noisy datasets where speed matters",
                "sklearn": "sklearn.ensemble.ExtraTreesClassifier",
                "complexity": "Medium",
                "eval_metrics": ["Accuracy", "F1 Score", "Feature Importance"],
            },
        ],
    },
    "regression": {
        "small": [
            {
                "name": "Ridge / Lasso Regression",
                "description": "Regularised linear regression (L2/L1 penalty).",
                "pros": ["Fast training", "Interpretable", "Handles multicollinearity (Ridge)", "Feature selection (Lasso)"],
                "cons": ["Assumes linear relationships", "Requires feature scaling", "Poor on non-linear data"],
                "best_for": "Continuous targets with linear relationships, baseline model",
                "sklearn": "sklearn.linear_model.Ridge / Lasso",
                "complexity": "Low",
                "eval_metrics": ["R² Score", "RMSE", "MAE", "MAPE"],
            },
            {
                "name": "Decision Tree Regressor",
                "description": "Splits data into regions and predicts mean value in each region.",
                "pros": ["No scaling required", "Handles non-linearity", "Interpretable"],
                "cons": ["Overfits easily", "High variance", "Poor extrapolation"],
                "best_for": "Non-linear regression with interpretability requirement",
                "sklearn": "sklearn.tree.DecisionTreeRegressor",
                "complexity": "Low",
                "eval_metrics": ["R² Score", "RMSE", "MAE"],
            },
            {
                "name": "K-Nearest Neighbors Regressor",
                "description": "Predicts target as average of k nearest training points.",
                "pros": ["Non-parametric", "Captures local patterns", "No training phase"],
                "cons": ["Slow inference on large data", "Sensitive to scale", "Affected by irrelevant features"],
                "best_for": "Small, locally structured regression problems",
                "sklearn": "sklearn.neighbors.KNeighborsRegressor",
                "complexity": "Low",
                "eval_metrics": ["R² Score", "RMSE"],
            },
        ],
        "medium": [
            {
                "name": "Random Forest Regressor",
                "description": "Ensemble of decision trees averaging predictions.",
                "pros": ["Robust to outliers", "Handles mixed types", "Feature importance built-in", "Low variance"],
                "cons": ["Memory intensive", "Slower than linear models", "Black-box predictions"],
                "best_for": "General-purpose regression, mixed datasets",
                "sklearn": "sklearn.ensemble.RandomForestRegressor",
                "complexity": "Medium",
                "eval_metrics": ["R² Score", "RMSE", "MAE", "Feature Importance"],
            },
            {
                "name": "Gradient Boosting Regressor",
                "description": "Sequential ensemble minimising regression residuals.",
                "pros": ["State-of-the-art accuracy on tabular data", "Built-in regularisation", "Handles outliers"],
                "cons": ["Hyperparameter tuning needed", "Risk of overfitting", "Slower training"],
                "best_for": "Maximum accuracy on structured data, Kaggle-style problems",
                "sklearn": "sklearn.ensemble.GradientBoostingRegressor",
                "complexity": "High",
                "eval_metrics": ["R² Score", "RMSE", "MAE"],
            },
            {
                "name": "Support Vector Regressor (SVR)",
                "description": "Regression equivalent of SVM — fits an ε-tube around predictions.",
                "pros": ["Works in high dimensions", "Kernel trick for non-linearity", "Robust to outliers with ε-insensitive loss"],
                "cons": ["Does not scale to large datasets", "Slow training", "Requires feature scaling"],
                "best_for": "Medium datasets, non-linear targets in moderate dimensions",
                "sklearn": "sklearn.svm.SVR",
                "complexity": "Medium",
                "eval_metrics": ["R² Score", "RMSE", "MAE"],
            },
        ],
        "large": [
            {
                "name": "LightGBM / XGBoost Regressor",
                "description": "Optimised gradient boosting for large-scale regression.",
                "pros": ["Handles millions of rows efficiently", "Categorical support", "Best-in-class accuracy", "GPU support"],
                "cons": ["Complex tuning", "Less interpretable", "Risk of over-engineering"],
                "best_for": "Large-scale tabular regression where accuracy is critical",
                "sklearn": "lightgbm.LGBMRegressor or xgboost.XGBRegressor",
                "complexity": "High",
                "eval_metrics": ["R² Score", "RMSE", "MAE", "MAPE"],
            },
            {
                "name": "Neural Network Regressor (MLP)",
                "description": "Deep learning-style multi-layer perceptron for regression.",
                "pros": ["Can learn very complex patterns", "Scales with data", "Feature extraction automatic"],
                "cons": ["Requires lots of data", "Needs careful tuning", "Computationally expensive"],
                "best_for": "Large datasets with complex non-linear relationships",
                "sklearn": "sklearn.neural_network.MLPRegressor",
                "complexity": "High",
                "eval_metrics": ["R² Score", "RMSE", "MAE", "Loss Curve"],
            },
            {
                "name": "Extra Trees Regressor",
                "description": "More randomised variant of Random Forest for regression.",
                "pros": ["Faster than Random Forest", "Less overfitting", "Handles large feature sets"],
                "cons": ["Slightly lower accuracy", "Higher bias"],
                "best_for": "Large noisy datasets requiring fast training",
                "sklearn": "sklearn.ensemble.ExtraTreesRegressor",
                "complexity": "Medium",
                "eval_metrics": ["R² Score", "RMSE", "Feature Importance"],
            },
        ],
    },
    "clustering": [
        {
            "name": "K-Means",
            "description": "Assigns each point to the nearest centroid.",
            "pros": ["Fast and scalable", "Easy to implement", "Well-understood"],
            "cons": ["Requires specifying K", "Assumes spherical clusters", "Sensitive to outliers"],
            "best_for": "General-purpose clustering of numeric data",
            "sklearn": "sklearn.cluster.KMeans",
            "complexity": "Low",
            "eval_metrics": ["Silhouette Score", "Inertia (Elbow Method)", "Davies-Bouldin Index"],
        },
        {
            "name": "DBSCAN",
            "description": "Density-based clustering — finds arbitrarily shaped clusters.",
            "pros": ["No need to specify K", "Detects outliers as noise", "Finds non-spherical clusters"],
            "cons": ["Sensitive to eps/min_samples parameters", "Struggles with varying densities"],
            "best_for": "Datasets with noise and non-spherical cluster shapes",
            "sklearn": "sklearn.cluster.DBSCAN",
            "complexity": "Medium",
            "eval_metrics": ["Silhouette Score", "Number of clusters detected"],
        },
        {
            "name": "Hierarchical Clustering",
            "description": "Builds a tree of clusters (dendrogram) via agglomerative or divisive methods.",
            "pros": ["No need to specify K upfront", "Visual dendrogram", "Good for small datasets"],
            "cons": ["O(n²) — slow on large datasets", "Hard to interpret for many clusters"],
            "best_for": "Small datasets, exploratory analysis, gene expression data",
            "sklearn": "sklearn.cluster.AgglomerativeClustering",
            "complexity": "Medium",
            "eval_metrics": ["Dendrogram", "Cophenetic Correlation"],
        },
    ],
}


def _build_ml_recommendations(df: pd.DataFrame, primary_keys: list, eda: dict, filename: str) -> Dict[str, Any]:
    """Analyse dataset characteristics and return top ML model recommendations."""
    n_rows = len(df)
    n_cols = len(df.columns)

    numeric_cols = [c for c in df.columns if c not in primary_keys and is_numeric_column(df, c)]
    categorical_cols = [c for c in df.columns if c not in primary_keys and is_categorical_column(df, c, primary_keys)]

    candidate_targets = detect_candidate_targets(df, primary_keys)

    # Detect likely task type
    if candidate_targets:
        target_series = df[candidate_targets[0]].dropna()
        task = detect_task_type(target_series)
    else:
        # No obvious categorical target → regression or clustering
        # If no numeric with enough variance, default to clustering
        # Otherwise regression
        has_numeric_outcome = any(
            df[c].nunique() > 20
            for c in numeric_cols
        )
        task = "regression" if has_numeric_outcome else "clustering"

    # Size bracket
    if n_rows < 1000:
        size_bracket = "small"
        size_label = f"Small ({n_rows:,} rows)"
    elif n_rows < 50000:
        size_bracket = "medium"
        size_label = f"Medium ({n_rows:,} rows)"
    else:
        size_bracket = "large"
        size_label = f"Large ({n_rows:,} rows)"

    # Pull model recommendations
    if task == "clustering":
        recs = _ML_CATALOG["clustering"][:3]
    else:
        db = _ML_CATALOG.get(task, _ML_CATALOG["classification"])
        bracket_recs = db.get(size_bracket, db["medium"])
        recs = bracket_recs[:3]

    # Preprocessing suggestions
    has_missing = df.isnull().values.any()
    has_categoricals = len(categorical_cols) > 0
    has_numeric = len(numeric_cols) > 0
    is_imbalanced = False
    if candidate_targets and task == "classification":
        try:
            counts = df[candidate_targets[0]].value_counts()
            if len(counts) > 1 and counts.max() / counts.min() > 3:
                is_imbalanced = True
        except Exception:
            pass

    preprocessing = []
    if has_missing:
        preprocessing.append({
            "step": "Missing Value Imputation",
            "reason": "Dataset contains missing values that most models cannot handle natively.",
            "options": ["SimpleImputer(strategy='median') for numeric", "SimpleImputer(strategy='most_frequent') for categorical", "IterativeImputer for complex patterns"]
        })
    if has_categoricals:
        preprocessing.append({
            "step": "Categorical Encoding",
            "reason": f"{len(categorical_cols)} categorical column(s) need to be converted to numeric.",
            "options": ["OneHotEncoder for nominal features (few categories)", "OrdinalEncoder for ordinal features", "TargetEncoder for high-cardinality features"]
        })
    if has_numeric:
        preprocessing.append({
            "step": "Feature Scaling",
            "reason": "Numeric features may have different magnitudes — scaling helps distance-based and linear models.",
            "options": ["StandardScaler (zero mean, unit variance) — recommended default", "MinMaxScaler (0–1 range) for bounded features", "RobustScaler if outliers are present"]
        })
    if is_imbalanced:
        preprocessing.append({
            "step": "Class Imbalance Handling",
            "reason": "Target class distribution is imbalanced — models may be biased toward majority class.",
            "options": ["class_weight='balanced' in most sklearn classifiers", "SMOTE (imbalanced-learn) for synthetic oversampling", "Threshold tuning on prediction probabilities"]
        })
    preprocessing.append({
        "step": "Feature Selection",
        "reason": "Removing irrelevant or redundant features reduces noise and speeds up training.",
        "options": ["SelectKBest with f_classif/f_regression for filter method", "RFE (Recursive Feature Elimination) for wrapper method", "Feature importance from tree models for embedded method"]
    })

    # Evaluation metrics
    if task == "classification":
        n_classes = df[candidate_targets[0]].nunique() if candidate_targets else 2
        eval_metrics = [
            {"metric": "Accuracy", "when": "Balanced class distribution"},
            {"metric": "F1 Score (macro/weighted)", "when": "Imbalanced classes — balances precision and recall"},
            {"metric": "ROC-AUC", "when": "Binary classification — measures discriminative ability"},
            {"metric": "Confusion Matrix", "when": "Understanding type of errors per class"},
            {"metric": "Precision / Recall", "when": "When false positives or false negatives have different costs"},
        ] if n_classes <= 2 else [
            {"metric": "F1 Score (macro)", "when": "Multi-class with imbalanced distribution"},
            {"metric": "Accuracy", "when": "Balanced multi-class datasets"},
            {"metric": "Confusion Matrix", "when": "Diagnosing per-class errors in multi-class problems"},
            {"metric": "Cohen's Kappa", "when": "Multi-class agreement beyond chance"},
        ]
    elif task == "regression":
        eval_metrics = [
            {"metric": "R² Score", "when": "Measure how much variance the model explains (1.0 = perfect)"},
            {"metric": "RMSE", "when": "Penalises large errors — good when outliers matter"},
            {"metric": "MAE", "when": "Robust to outliers — average absolute error"},
            {"metric": "MAPE (%)", "when": "Percentage error — useful for business interpretation"},
        ]
    else:
        eval_metrics = [
            {"metric": "Silhouette Score", "when": "Measures cluster cohesion and separation (higher = better)"},
            {"metric": "Elbow Method (Inertia)", "when": "Finding optimal K for K-Means"},
            {"metric": "Davies-Bouldin Index", "when": "Lower = better separated clusters"},
        ]

    return {
        "task": task,
        "dataset_size": size_label,
        "size_bracket": size_bracket,
        "n_rows": n_rows,
        "n_features": n_cols - len(primary_keys),
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols),
        "candidate_targets": candidate_targets[:5],
        "is_imbalanced": is_imbalanced,
        "recommendations": recs,
        "preprocessing_steps": preprocessing,
        "evaluation_metrics": eval_metrics,
    }


def _advisor_model_from_name(name: str, task: str):
    """Map recommendation names to available sklearn estimators."""
    n = (name or "").lower()

    if task == "classification":
        if "logistic" in n:
            return LogisticRegression(max_iter=300, n_jobs=-1, solver="saga", class_weight="balanced")
        if "decision tree" in n:
            return DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42)
        if "random forest" in n:
            return RandomForestClassifier(n_estimators=50, n_jobs=-1, class_weight="balanced", max_features="sqrt", random_state=42)
        if "gradient boosting" in n or "xgboost" in n or "lightgbm" in n:
            return GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42)
        if "extra trees" in n:
            return ExtraTreesClassifier(n_estimators=50, n_jobs=-1, class_weight="balanced", random_state=42)
        if "nearest" in n or "knn" in n:
            return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        if "support vector" in n or n.strip() == "svm":
            return SVC(C=1.0, kernel="rbf", probability=False)
        if "neural network" in n or "mlp" in n:
            return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
        return None

    if task == "regression":
        if "ridge" in n or "lasso" in n:
            return Ridge()
        if "decision tree" in n:
            return DecisionTreeRegressor(max_depth=8, random_state=42)
        if "random forest" in n:
            return RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
        if "gradient boosting" in n or "xgboost" in n or "lightgbm" in n:
            return GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42)
        if "extra trees" in n:
            return ExtraTreesRegressor(n_estimators=50, n_jobs=-1, random_state=42)
        if "nearest" in n or "knn" in n:
            return KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
        if "linear" in n:
            return LinearRegression()
        if "support vector" in n or "svr" in n:
            return SVR(C=1.0, epsilon=0.1, kernel="rbf")
        if "neural network" in n or "mlp" in n:
            return MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
        return None

    return None


def _build_regression_researcher_summary(benchmark: Dict[str, Any]) -> str:
    """Create a concise researcher-facing explanation using benchmark evidence."""
    rows = [r for r in benchmark.get("models", []) if not r.get("error")]
    if not rows:
        return "Benchmark completed, but none of the regression models produced valid scores. Review feature quality and target consistency before model selection."

    rows_sorted = sorted(rows, key=lambda r: (r.get("cv_r2_mean") is not None, r.get("cv_r2_mean", -np.inf)), reverse=True)
    best = rows_sorted[0]
    second = rows_sorted[1] if len(rows_sorted) > 1 else None

    best_name = best.get("model", "Best model")
    best_r2 = best.get("r2_score")
    best_adj_r2 = best.get("adj_r_square")
    best_rmse = best.get("root_mean_squared_error_rmse")
    best_mae = best.get("mean_absolute_error_mae")
    best_cv = best.get("cv_r2_mean")
    best_stability = best.get("stability")

    gap_text = ""
    if second and second.get("cv_r2_mean") is not None and best_cv is not None:
        gap = round(float(best_cv - second.get("cv_r2_mean", 0.0)), 4)
        gap_text = f" It outperformed the next best model ({second.get('model')}) by {gap} points on CV R²."

    return (
        f"Based on the obtained benchmark metrics, {best_name} is the most suitable model for this dataset. "
        f"It achieved R²={best_r2}, Adjusted R²={best_adj_r2}, RMSE={best_rmse}, and MAE={best_mae}, "
        f"with cross-validation mean R²={best_cv} and stability={best_stability}%."
        f"{gap_text} "
        f"These results indicate the best balance of predictive strength and generalization for researcher-facing analysis."
    )


def _run_ml_advisor_benchmark(df: pd.DataFrame, primary_keys: list, rec_result: Dict[str, Any]) -> Dict[str, Any]:
    """Train candidate models, evaluate, and return a metric comparison table."""
    task = rec_result.get("task")
    if task not in ("classification", "regression"):
        return {
            "available": False,
            "reason": "Benchmarking currently supports classification and regression tasks.",
            "task": task,
            "models": [],
        }

    candidate_targets = rec_result.get("candidate_targets") or detect_candidate_targets(df, primary_keys)
    if not candidate_targets:
        return {
            "available": False,
            "reason": "No suitable target column detected for supervised training.",
            "task": task,
            "models": [],
        }

    target_column = candidate_targets[0]
    data = sample_dataframe(df, MAX_PRED_SAMPLE_ROWS).copy().dropna(subset=[target_column])
    if data.empty:
        return {
            "available": False,
            "reason": "No rows remain after dropping missing target values.",
            "task": task,
            "models": [],
        }

    selected_features = select_prediction_features(data, target_column, primary_keys)
    if not selected_features:
        return {
            "available": False,
            "reason": "No suitable features found for model training.",
            "task": task,
            "models": [],
        }

    # Extra guard: remove non-sensible numeric columns (ID-like fields) from benchmark features
    filtered_features = []
    for c in selected_features:
        if is_numeric_column(data, c) and (not is_sensible_numeric_column(data, c, primary_keys=primary_keys)):
            continue
        filtered_features.append(c)
    selected_features = filtered_features

    if not selected_features:
        return {
            "available": False,
            "reason": "All candidate features were identifier-like or unsuitable after filtering.",
            "task": task,
            "models": [],
        }

    data = data[selected_features + [target_column]].copy()
    X = data[selected_features]
    numeric_features = [c for c in selected_features if is_numeric_column(data, c)]
    categorical_features = [c for c in selected_features if c not in numeric_features]
    preprocessor = _build_preprocessor(numeric_features, categorical_features)

    def _compact_err(exc: Exception) -> str:
        msg = str(exc).replace("\n", " ").strip()
        if "Cannot use median strategy with non-numeric data" in msg:
            return "Numeric preprocessing failed due to mixed string values in a numeric feature."
        if len(msg) > 220:
            return msg[:220] + "..."
        return msg

    # Build model set
    models = []

    if task == "classification":
        # Keep recommendation-driven behavior for classification
        seen = set()
        for rec in rec_result.get("recommendations", []):
            m_name = rec.get("name", "")
            estimator = _advisor_model_from_name(m_name, task)
            if estimator is None or m_name in seen:
                continue
            seen.add(m_name)
            models.append((m_name, estimator))

    if not models and task == "classification":
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=300, n_jobs=-1, solver="saga", class_weight="balanced")),
            ("Random Forest", RandomForestClassifier(n_estimators=50, n_jobs=-1, class_weight="balanced", max_features="sqrt", random_state=42)),
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42)),
        ]

    if task == "regression":
        # Evaluate the full requested regression algorithm set
        models = [
            ("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=50, n_jobs=-1, random_state=42)),
            ("RandomForestRegressor", RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)),
            ("BaggingRegressor", BaggingRegressor(n_estimators=50, random_state=42)),
            ("DecisionTreeRegressor", DecisionTreeRegressor(max_depth=8, random_state=42)),
            ("KNeighborsRegressor", KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
            ("GradientBoostingRegressor", GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42)),
            ("LinearRegression", LinearRegression()),
            ("Ridge Regression", Ridge()),
            ("Lasso Regression", Lasso(alpha=0.01, max_iter=5000, random_state=42)),
        ]

        # Optional XGBoost if installed
        try:
            from xgboost import XGBRegressor  # type: ignore
            models.insert(3, ("XGBRegressor", XGBRegressor(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror",
            )))
        except Exception:
            # Keep a placeholder so researcher sees why it's missing
            results.append({"model": "XGBRegressor", "error": "xgboost package not installed in environment."})

    results = []
    best_model_name = ""
    best_score = -np.inf

    if task == "classification":
        y = data[target_column].astype(str)
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        if len(np.unique(y_enc)) < 2:
            return {
                "available": False,
                "reason": "Target column must have at least 2 classes.",
                "task": task,
                "models": [],
            }

        class_counts = np.bincount(y_enc)
        can_stratify = class_counts.min() >= 2
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_enc, test_size=0.2, random_state=42,
            stratify=y_enc if can_stratify else None
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) if can_stratify else KFold(n_splits=3, shuffle=True, random_state=42)

        for model_name, estimator in models:
            try:
                pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
                cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_te)

                row = {
                    "model": model_name,
                    "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
                    "precision_macro": round(float(precision_score(y_te, y_pred, average="macro", zero_division=0)), 4),
                    "recall_macro": round(float(recall_score(y_te, y_pred, average="macro", zero_division=0)), 4),
                    "f1_macro": round(float(f1_score(y_te, y_pred, average="macro", zero_division=0)), 4),
                    "cv_f1_mean": round(float(np.mean(cv_scores)), 4),
                    "cv_f1_std": round(float(np.std(cv_scores)), 4),
                    "stability": round(float(max(0.0, 1.0 - np.std(cv_scores)) * 100), 1),
                }
                results.append(row)

                if row["cv_f1_mean"] > best_score:
                    best_score = row["cv_f1_mean"]
                    best_model_name = model_name
            except Exception as me:
                results.append({"model": model_name, "error": _compact_err(me)})

        for r in results:
            r["is_best"] = r.get("model") == best_model_name

        return {
            "available": True,
            "task": "classification",
            "target_column": target_column,
            "metrics_compared": ["accuracy", "precision_macro", "recall_macro", "f1_macro", "cv_f1_mean", "cv_f1_std", "stability"],
            "models": results,
            "best_model": best_model_name,
            "train_size": len(X_tr),
            "test_size": len(X_te),
        }

    # regression
    y = pd.to_numeric(data[target_column], errors="coerce")
    data = data[y.notna()].copy()
    y = y[y.notna()]
    X = data[selected_features]
    if len(y) < 20:
        return {
            "available": False,
            "reason": "Not enough rows for reliable regression benchmarking.",
            "task": task,
            "models": [],
        }

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    for model_name, estimator in models:
        try:
            pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
            cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="r2", n_jobs=-1)
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)

            y_te_arr = np.asarray(y_te)
            y_pred_arr = np.asarray(y_pred)
            nonzero_mask = y_te_arr != 0
            mape = float(np.mean(np.abs((y_te_arr[nonzero_mask] - y_pred_arr[nonzero_mask]) / y_te_arr[nonzero_mask])) * 100) if nonzero_mask.any() else None

            n_test = len(y_te_arr)
            p_feats = max(1, len(selected_features))
            r2_val = float(r2_score(y_te, y_pred))
            adj_r2 = None
            if n_test > (p_feats + 1):
                adj_r2 = 1.0 - (1.0 - r2_val) * ((n_test - 1) / (n_test - p_feats - 1))

            mse_val = float(mean_squared_error(y_te, y_pred))
            rmse_val = float(np.sqrt(mse_val))

            rmsle_val = None
            if np.all(y_te_arr >= 0) and np.all(y_pred_arr >= 0):
                try:
                    rmsle_val = float(np.sqrt(np.mean((np.log1p(y_pred_arr) - np.log1p(y_te_arr)) ** 2)))
                except Exception:
                    rmsle_val = None

            row = {
                "model": model_name,
                "model_name": model_name,
                "r2": round(r2_val, 4),
                "r2_score": round(r2_val, 4),
                "adj_r_square": round(float(adj_r2), 4) if adj_r2 is not None else None,
                "mean_absolute_error_mae": round(float(mean_absolute_error(y_te, y_pred)), 4),
                "root_mean_squared_error_rmse": round(rmse_val, 4),
                "mean_absolute_percentage_error_mape": round(mape, 4) if mape is not None else None,
                "mean_squared_error_mse": round(mse_val, 4),
                "root_mean_squared_log_error_rmsle": round(rmsle_val, 4) if rmsle_val is not None else None,
                "mae": round(float(mean_absolute_error(y_te, y_pred)), 4),
                "rmse": round(rmse_val, 4),
                "mape": round(mape, 4) if mape is not None else None,
                "cv_r2_mean": round(float(np.mean(cv_scores)), 4),
                "cv_r2_std": round(float(np.std(cv_scores)), 4),
                "stability": round(float(max(0.0, 1.0 - np.std(cv_scores)) * 100), 1),
            }
            results.append(row)

            if row["cv_r2_mean"] > best_score:
                best_score = row["cv_r2_mean"]
                best_model_name = model_name
        except Exception as me:
            results.append({"model": model_name, "error": _compact_err(me)})

    for r in results:
        r["is_best"] = r.get("model") == best_model_name

    return {
        "available": True,
        "task": "regression",
        "target_column": target_column,
        "metrics_compared": [
            "adj_r_square",
            "mean_absolute_error_mae",
            "root_mean_squared_error_rmse",
            "mean_absolute_percentage_error_mape",
            "mean_squared_error_mse",
            "root_mean_squared_log_error_rmsle",
            "r2_score",
            "cv_r2_mean",
            "cv_r2_std",
            "stability",
        ],
        "models": results,
        "best_model": best_model_name,
        "train_size": len(X_tr),
        "test_size": len(X_te),
        "researcher_summary": _build_regression_researcher_summary({
            "models": results,
            "best_model": best_model_name,
        }),
    }


@app.get("/api/ml-recommendations/{dataset_id}")
async def get_ml_recommendations(dataset_id: str):
    """Return ML model recommendations for a tabular dataset."""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    ds = datasets[dataset_id]
    if ds.get("dataset_type") in ("image", "single_image"):
        raise HTTPException(400, "ML recommendations are for tabular datasets only. Use the Image tab for image datasets.")

    cache = ds.setdefault("analysis_cache", {})
    if "ml_recommendations" in cache:
        return cache["ml_recommendations"]

    try:
        df = ds["df"]
        primary_keys = ds.get("primary_keys", [])
        eda = eda_results.get(dataset_id, {})
        filename = ds["filename"]

        result = _build_ml_recommendations(df, primary_keys, eda, filename)

        # AI-powered reasoning
        try:
            ai_prompt = f"""You are a senior ML engineer. Based on this dataset profile, write 3-4 concise sentences explaining
which machine learning approach is most suitable and why. Be specific about the dataset characteristics that drive your recommendation.

Dataset: {filename}
Rows: {result['n_rows']:,}
Features: {result['n_features']} ({result['numeric_features']} numeric, {result['categorical_features']} categorical)
Detected task: {result['task']}
Dataset size bracket: {result['dataset_size']}
Class imbalanced: {result['is_imbalanced']}
Top recommended model: {result['recommendations'][0]['name'] if result['recommendations'] else 'N/A'}

No markdown. Plain text only. Focus on why this model fits THIS dataset."""
            result["ai_reasoning"] = get_gemini_response(ai_prompt, "lite")
        except Exception:
            result["ai_reasoning"] = "AI reasoning unavailable."

        cache["ml_recommendations"] = result
        return result

    except Exception as e:
        logger.error(f"ML recommendations error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"ML recommendations failed: {str(e)}")


@app.post("/api/ml-benchmark/{dataset_id}")
async def run_ml_benchmark(dataset_id: str, force: bool = False):
    """Run full ML benchmark for Advisor and compare evaluation metrics across models."""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")

    ds = datasets[dataset_id]
    if ds.get("dataset_type") in ("image", "single_image", "multi_image"):
        raise HTTPException(400, "ML benchmark is for tabular datasets only.")

    cache = ds.setdefault("analysis_cache", {})
    if (not force) and ("ml_benchmark" in cache):
        return cache["ml_benchmark"]

    try:
        df = ds["df"]
        primary_keys = ds.get("primary_keys", [])
        eda = eda_results.get(dataset_id, {})
        filename = ds.get("filename", "dataset")

        rec_result = cache.get("ml_recommendations")
        if not rec_result:
            rec_result = _build_ml_recommendations(df, primary_keys, eda, filename)

        benchmark = _run_ml_advisor_benchmark(df, primary_keys, rec_result)
        cache["ml_benchmark"] = benchmark
        return benchmark

    except Exception as e:
        logger.error(f"ML benchmark error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"ML benchmark failed: {str(e)}")


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