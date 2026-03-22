# Autonomous Data Intelligence Platform (ADIP)

ADIP is a FastAPI + single-page HTML app for two analytics modes in one interface:

- Tabular analytics for CSV/XLSX/JSON datasets.
- Image analytics for ZIP datasets, single images, or multiple image files.

The backend stores datasets in memory and caches computed analyses per dataset for fast tab switching.

## What This Project Does

### Tabular workflow

- Upload tabular data (`.csv`, `.xlsx`, `.xls`, `.json`)
- Clean and normalize data (remove fully empty rows/columns, normalize strings, deduplicate, detect IDs)
- Auto EDA with column-level stats and missing/duplicate summaries
- Interactive analytics endpoints:
  - Numerical distributions + boxplots
  - Categorical distributions (bar/pie)
  - Correlation heatmaps
  - Outlier analysis (IQR + Z-score)
  - Time series analysis
  - Forecasting
  - Anomaly detection (univariate + Isolation Forest)
  - Contour/density plots
- Data health score (`completeness`, `uniqueness`, `consistency`, `validity`)
- AI insights and dataset chat
- Natural-language query to executable pandas expression
- ML Advisor:
  - model recommendations
  - model benchmarking
  - AutoML-style predictive endpoint (classification/regression)
- Export:
  - PowerPoint EDA report
  - Processed CSV

### Image workflow

- Upload modes:
  - ZIP dataset (`class_name/image.ext` preferred)
  - Single image
  - Multiple images
- Auto image profiling:
  - class distribution and imbalance
  - dimensions/channels/grayscale detection
  - representative previews
- Augmentation intelligence:
  - rule-based suggestions
  - domain-aware recommendations
  - generated preview variants (flip/rotation/contrast/blur/etc.)
- AI vision insights using LLaVA + structured section output for CV planning

## Architecture

- Backend: FastAPI app in `main.py`
- Frontend: static SPA in `index.html` (Tailwind + Plotly)
- Data/EDA helpers: `clean_and_EDA_generate.py`
- PPT export: `generate_report.py`
- AI adapters: `utils.py`
- SQL helper module: `smart_query.py` (present in repo; not currently wired into FastAPI routes)

## AI Model Usage (Important)

This project uses both local and cloud AI paths:

- Local via Ollama:
  - text model: `llama3.1:8b`
  - vision model: `llava`
  - used for image commentary/vision analysis and several local text-generation tasks
- Cloud via Gemini API:
  - used by `/api/insights/{dataset_id}`, `/api/chat`, and `/api/query`
  - requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`

If Gemini keys are missing, Gemini-backed endpoints return explicit error messages.

## Requirements

- Python 3.10+ recommended
- Ollama installed and running (`ollama serve`)
- Optional but recommended: Gemini API key for tabular AI endpoints

## Setup

1. Create and activate a virtual environment.

Windows (PowerShell):

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env`.

```env
GEMINI_API_KEY=your_key_here
# optional
GEMINI_MODEL=gemini-2.5-flash-lite
```

4. Pull Ollama models.

```bash
ollama pull llama3.1:8b
ollama pull llava
```

## Run

Use two terminals.

Terminal 1 (backend):

```bash
python main.py
```

Terminal 2 (frontend static server):

```bash
python -m http.server 3000
```

Open:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

## API Endpoints

### General

- `GET /`
- `GET /health`
- `GET /api/datasets`
- `GET /api/dataset/{dataset_id}`
- `DELETE /api/dataset/{dataset_id}`

### Tabular upload + analysis

- `POST /api/upload`
- `GET /api/analyze/{dataset_id}/numerical`
- `GET /api/analyze/{dataset_id}/categorical`
- `GET /api/analyze/{dataset_id}/correlations`
- `GET /api/analyze/{dataset_id}/outliers`
- `GET /api/analyze/{dataset_id}/timeseries`
- `GET /api/analyze/{dataset_id}/forecast`
- `GET /api/analyze/{dataset_id}/anomaly`
- `GET /api/analyze/{dataset_id}/contour`
- `POST /api/explore`
- `GET /api/insights/{dataset_id}`
- `POST /api/chat`
- `POST /api/query`
- `GET /api/column/{dataset_id}/{column_name}`
- `GET /api/health-score/{dataset_id}`

### ML advisor + predictive

- `GET /api/ml-recommendations/{dataset_id}`
- `POST /api/ml-benchmark/{dataset_id}`
- `POST /api/predictive`

### Image upload + vision

- `POST /api/upload-images` (ZIP datasets)
- `POST /api/upload-single-image`
- `POST /api/upload-multiple-images`
- `GET /api/image-analysis/{dataset_id}`
- `GET /api/image-ai-insights/{dataset_id}`
- `GET /api/augmentation/{dataset_id}`

### Export

- `GET /api/export/{dataset_id}/ppt`
- `GET /api/export/{dataset_id}/csv`

## Frontend Tabs

### Tabular dashboard tabs

- Overview
- Numerical
- Categorical
- Correlations
- Outliers
- Time Series
- AI Insights
- ML Advisor
- Ask AI
- Explorer

### Image dashboard tabs

- Overview
- AI Content Insights
- Augmentation
- Previews
- Vision Models

## Repository Structure

```text
.
├─ main.py
├─ index.html
├─ clean_and_EDA_generate.py
├─ generate_report.py
├─ smart_query.py
├─ utils.py
├─ requirements.txt
├─ server_err.txt
├─ server_out.txt
├─ README.md
└─ LICENSE
```

## Notes From Current Logs

- Server starts correctly on `0.0.0.0:8000`.
- Upload + predictive routes have successful responses in sample logs.
- Primary-key detection can classify highly unique business columns as IDs (expected by current heuristics).

## Troubleshooting

- Backend does not start:
  - reinstall deps: `pip install -r requirements.txt`
  - check port `8000` availability
- Frontend cannot call API:
  - ensure backend is running on `http://localhost:8000`
  - serve frontend from `http://localhost:3000`
- AI endpoints fail:
  - ensure Ollama is running: `ollama serve`
  - ensure required models are pulled (`llama3.1:8b`, `llava`)
  - set `GEMINI_API_KEY` for `/api/insights`, `/api/chat`, `/api/query`

## License

MIT License (see `LICENSE`)