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
        
        # Generate EDA
        logger.info(f"Generating EDA for: {file.filename}")
        try:
            eda = enhanced_eda_json(df)
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
        
        # Store in memory
        datasets[dataset_id] = {
            "df": df,
            "filename": file.filename,
            "uploaded_at": datetime.now().isoformat(),
            "primary_keys": primary_keys
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
            "eda": eda
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
        "eda": enhanced_eda,
        "preview": df.head(10).fillna("").to_dict('records')
    }

@app.get("/api/analyze/{dataset_id}/numerical")
async def get_numerical_analysis(dataset_id: str):
    """Get numerical analysis with robust statistics, outlier computation, information-rich Plotly charts"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    try:
        df = datasets[dataset_id]["df"]
        eda = eda_results.get(dataset_id, {})
        primary_keys = datasets[dataset_id].get("primary_keys", [])
        
        # First, get all numeric columns
        all_numeric_cols = [col for col in df.columns if is_numeric_column(df, col)]
        logger.info(f"Numerical: {len(all_numeric_cols)} numeric columns found: {all_numeric_cols}")
        
        # Filter out non-sensible numeric columns (IDs, mobile numbers, etc.)
        sensible_numeric_cols = []
        skipped_cols = []
        
        for col in all_numeric_cols:
            if is_sensible_numeric_column(df, col, eda, primary_keys):
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
        
        visualizations = []
        statistics = {}
        for col in sensible_numeric_cols[:10]:  # Limit to 10 columns
            try:
                series = get_clean_series(df, col)
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
        
        return {
            "type": "numerical", 
            "columns": sensible_numeric_cols, 
            "count": len(sensible_numeric_cols), 
            "visualizations": visualizations, 
            "statistics": statistics,
            "skipped_columns": skipped_cols if skipped_cols else None,
            "total_numeric_columns": len(all_numeric_cols)
        }
    except Exception as e:
        logger.error(f"Numerical analysis error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error generating numerical analysis: {str(e)}")

@app.get("/api/analyze/{dataset_id}/categorical")
async def get_categorical_analysis(dataset_id: str):
    """Get categorical analysis with visualizations"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    
    try:
        df = datasets[dataset_id]["df"]
        primary_keys = datasets[dataset_id].get("primary_keys", [])
        
        # Get all potential categorical columns and track skipped identifiers
        all_potential_categorical = []
        skipped_identifiers = []
        
        for col in df.columns:
            unique_count = df[col].nunique()
            total_count = len(df)
            unique_ratio = unique_count / total_count if total_count > 0 else 0
            
            # Check if it's categorical (this function now filters out high-uniqueness columns and primary keys)
            if is_categorical_column(df, col, primary_keys):
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
        
        visualizations = []
        
        for col in categorical_cols[:10]:
            try:
                series = get_clean_series(df, col)
                
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
        
        return {
            "type": "categorical",
            "columns": categorical_cols,
            "count": len(categorical_cols),
            "visualizations": visualizations,
            "skipped_identifiers": skipped_identifiers if skipped_identifiers else None
        }
    
    except Exception as e:
        logger.error(f"Categorical analysis error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error generating categorical analysis: {str(e)}")

@app.get("/api/analyze/{dataset_id}/correlations")
async def get_correlation_analysis(dataset_id: str):
    """Get correlation analysis with heatmap. Handles NaNs and zero-variance columns. Only shows strong/high correlations."""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    try:
        df = datasets[dataset_id]["df"]
        eda = eda_results.get(dataset_id, {})
        primary_keys = datasets[dataset_id].get("primary_keys", [])
        
        # Get all numeric columns and filter out non-sensible ones
        all_numeric_cols = [col for col in df.columns if is_numeric_column(df, col)]
        numeric_cols = [col for col in all_numeric_cols if is_sensible_numeric_column(df, col, eda, primary_keys)]
        logger.info(f"Correlation: {len(numeric_cols)} sensible numeric columns found (out of {len(all_numeric_cols)} total): {numeric_cols}")
        if len(numeric_cols) < 2:
            return {"type": "correlations", "error": "Not enough numeric columns (at least 2 required)", "numeric_columns_found": len(numeric_cols), "columns": numeric_cols, "strong_correlations": [], "visualizations": []}
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
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
        return {"type": "correlations", "columns": corr_matrix.columns.tolist(), "strong_correlations": strong_corr, "visualizations": [{"type": "heatmap", "figure": convert_plotly_figure_to_dict(fig)}]}
    except Exception as e:
        logger.error(f"Correlation analysis error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error generating correlation analysis: {str(e)}")

@app.get("/api/analyze/{dataset_id}/outliers")
async def get_outliers_analysis(dataset_id: str):
    """Get comprehensive outliers analysis with IQR, Z-score, and visualization"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    try:
        df = datasets[dataset_id]["df"]
        eda = eda_results.get(dataset_id, {})
        primary_keys = datasets[dataset_id].get("primary_keys", [])
        
        # Get all numeric columns and filter out non-sensible ones (including primary keys)
        all_numeric_cols = [col for col in df.columns if is_numeric_column(df, col)]
        numeric_cols = [col for col in all_numeric_cols if is_sensible_numeric_column(df, col, eda, primary_keys)]
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
                series = get_clean_series(df, col)
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
        
        return {
            "type": "outliers",
            "columns": numeric_cols,
            "count": len(numeric_cols),
            "visualizations": visualizations,
            "statistics": statistics,
            "outlier_details": outlier_details
        }
    
    except Exception as e:
        logger.error(f"Outliers analysis error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error generating outliers analysis: {str(e)}")

@app.get("/api/analyze/{dataset_id}/timeseries")
async def get_timeseries_analysis(dataset_id: str):
    """Get time series analysis with trend detection, seasonality, and forecasting"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    try:
        df = datasets[dataset_id]["df"]
        
        # Try to detect date/time columns - be strict: only accept actual datetime columns
        date_cols = []
        for col in df.columns:
            # Skip if column is already numeric (not a date)
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Try strict datetime conversion
            try:
                # Attempt to convert entire column to datetime
                test_series = pd.to_datetime(df[col], errors='raise', format='mixed')
                
                # Check if conversion was successful and has valid datetime values
                valid_count = test_series.notna().sum()
                if valid_count > len(df) * 0.8:  # At least 80% valid datetime values
                    # Check if it's actually a datetime type or datetime64
                    if pd.api.types.is_datetime64_any_dtype(test_series) or isinstance(test_series.dtype, pd.DatetimeTZDtype):
                        date_cols.append(col)
                        logger.info(f"Detected datetime column: {col}")
                    # Also check if the values are actually dates (not just strings that look like dates)
                    elif valid_count == len(df):
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
        all_numeric_cols = [col for col in df.columns if is_numeric_column(df, col)]
        numeric_cols = [col for col in all_numeric_cols if is_sensible_numeric_column(df, col, eda, primary_keys)]
        
        if not date_cols:
            return {
                "type": "timeseries",
                "error": "No time series detected. Dataset does not contain a valid date/time column or timestamp.",
                "date_columns_found": 0,
                "numeric_columns_found": len(numeric_cols),
                "visualizations": [],
                "message": "No time series detected"
            }
        
        if not numeric_cols:
            return {
                "type": "timeseries",
                "error": "No numeric columns found for time series analysis",
                "date_columns_found": len(date_cols),
                "numeric_columns_found": 0,
                "visualizations": [],
                "message": "No time series detected"
            }
        
        visualizations = []
        analyses = {}
        
        # Analyze each numeric column with each date column
        valid_combinations = 0
        for date_col in date_cols[:2]:  # Limit to 2 date columns
            for num_col in numeric_cols[:5]:  # Limit to 5 numeric columns
                try:
                    # Prepare time series data
                    ts_df = df[[date_col, num_col]].copy()
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
            return {
                "type": "timeseries",
                "error": "No time series detected. Dataset does not contain valid sequential date/time data.",
                "date_columns_found": len(date_cols),
                "numeric_columns_found": len(numeric_cols),
                "visualizations": [],
                "message": "No time series detected"
            }
        
        return {
            "type": "timeseries",
            "date_columns": date_cols,
            "numeric_columns": numeric_cols,
            "visualizations": visualizations,
            "analyses": analyses
        }
    
    except Exception as e:
        logger.error(f"Time series analysis error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error generating time series analysis: {str(e)}")

@app.get("/api/analyze/{dataset_id}/contour")
async def get_contour_analysis(dataset_id: str):
    """Get contour box plots for numeric column pairs"""
    if dataset_id not in datasets:
        raise HTTPException(404, "Dataset not found")
    try:
        df = datasets[dataset_id]["df"]
        eda = eda_results.get(dataset_id, {})
        primary_keys = datasets[dataset_id].get("primary_keys", [])
        
        # Get all numeric columns and filter out non-sensible ones (including primary keys)
        all_numeric_cols = [col for col in df.columns if is_numeric_column(df, col)]
        numeric_cols = [col for col in all_numeric_cols if is_sensible_numeric_column(df, col, eda, primary_keys)]
        
        if len(numeric_cols) < 2:
            return {
                "type": "contour",
                "error": "Need at least 2 numeric columns for contour plots",
                "numeric_columns_found": len(numeric_cols),
                "visualizations": []
            }
        
        visualizations = []
        
        # Create contour plots for pairs of numeric columns
        for i, col1 in enumerate(numeric_cols[:5]):
            for col2 in numeric_cols[i+1:6]:  # Limit pairs
                try:
                    data_df = df[[col1, col2]].copy()
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
        
        return {
            "type": "contour",
            "columns": numeric_cols,
            "visualizations": visualizations
        }
    
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
            "column_details": "GET /api/column/{dataset_id}/{column_name}",
            "export_ppt": "GET /api/export/{dataset_id}/ppt",
            "export_csv": "GET /api/export/{dataset_id}/csv",
            "delete": "DELETE /api/dataset/{dataset_id}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)