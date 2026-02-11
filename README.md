# AI-Powered EDA Tool 🚀

An intelligent Exploratory Data Analysis (EDA) platform that automates data analysis from CSV and XLSX files. Built with FastAPI backend and a modern web interface, featuring comprehensive statistics, interactive visualizations, and AI-powered natural language interaction using Google's Gemini LLM.

## 🌟 Features

### 📊 Automated Data Analysis
- **Comprehensive EDA**: Automatically generates statistical summaries, distributions, and data quality metrics
- **Multiple Analysis Types**:
  - Numerical Analysis: Histograms, box plots, descriptive statistics, skewness, kurtosis, and outlier detection
  - Categorical Analysis: Value counts, bar charts, and pie charts with missing value handling
  - Correlation Analysis: Interactive heatmaps showing relationships between numerical variables
  - Outlier Detection: IQR-based outlier identification with visualizations
  - Time Series Analysis: Automatic detection and analysis of temporal patterns
  - Contour Plots: 3D surface visualizations for multivariate analysis

### 🤖 AI-Powered Features
- **Intelligent Chatbot**: Ask questions about your dataset in natural language
- **Natural Language Querying**: Execute data queries using conversational language (e.g., "Show me rows where age > 30")
- **Automated Insights**: AI-generated insights organized into structured sections (Data Overview, Quality, Patterns, Findings, Recommendations)
- **Context-Aware Responses**: Maintains conversation history for follow-up questions

### 📈 Interactive Visualizations
- **Plotly Integration**: Interactive, responsive charts with zoom, pan, and hover details
- **Real-time Updates**: Dynamic visualizations that adapt to data filtering
- **Export Ready**: All visualizations can be exported as part of reports

### 🔍 Data Exploration
- **Advanced Filtering**: Filter by numeric ranges, categorical values, or multiple criteria
- **Sorting & Pagination**: Sort by any column with configurable pagination
- **Data Preview**: Interactive table view with search and navigation
- **Column Details**: Deep dive into individual column statistics and distributions

### 💾 Export Capabilities
- **PowerPoint Reports**: Generate comprehensive EDA reports as PowerPoint presentations
- **CSV Export**: Download filtered or processed datasets
- **Visual Export**: All charts are export-ready for presentations

### 🛠️ Data Processing
- **Smart Data Cleaning**: Automatic handling of duplicates, type detection, and missing values
- **Intelligent Type Detection**: Automatically identifies numeric, categorical, datetime, and boolean columns
- **Missing Value Handling**: Comprehensive reporting and visualization of missing data patterns

## 🏗️ Architecture

### Backend
- **FastAPI**: High-performance async API framework (runs on port 8000)
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Plotly**: Interactive visualizations
- **Google Generative AI (Gemini)**: LLM integration for chat and query generation

### Frontend
- **HTML/CSS/JavaScript**: Modern single-page application (served on port 3000)
- **Tailwind CSS**: Utility-first styling with glassmorphism effects
- **Plotly.js**: Client-side interactive chart rendering

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Naganathan194/AI-ENABLED-AUTOMATED-EDA-TOOL-WITH-INTELLIGENT-CHATBOT-FOR-DATA-INSIGHTS-LLM-BI.git
   cd AI-ENABLED-AUTOMATED-EDA-TOOL-WITH-INTELLIGENT-CHATBOT-FOR-DATA-INSIGHTS-LLM-BI
   ```

2. **Create a virtual environment**
   
   On **Windows**:
   ```bash
   python -m venv venv
   ```
   
   On **macOS/Linux**:
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**
   
   On **Windows** (PowerShell):
   ```bash
   venv\Scripts\Activate.ps1
   ```
   
   On **Windows** (Command Prompt):
   ```bash
   venv\Scripts\activate.bat
   ```
   
   On **macOS/Linux**:
   ```bash
   source venv/bin/activate
   ```

   You should see `(venv)` at the beginning of your terminal prompt.

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Gemini API**
   
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
   
   Alternatively, set it as an environment variable:
   
   On **Windows** (PowerShell):
   ```powershell
   $env:GEMINI_API_KEY="your_api_key_here"
   ```
   
   On **Windows** (Command Prompt):
   ```cmd
   set GEMINI_API_KEY=your_api_key_here
   ```
   
   On **macOS/Linux**:
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   ```

## 🚀 Running the Application

The application requires two separate terminals - one for the backend and one for the frontend.

### Terminal 1 - Backend Server (Port 8000)

Make sure your virtual environment is activated, then run:

```bash
python main.py
```

You should see output indicating the FastAPI server is running:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The backend API will be available at `http://localhost:8000`

### Terminal 2 - Frontend Server (Port 3000)

In a **new terminal window** (with virtual environment activated), navigate to the project directory and run:

```bash
python -m http.server 3000
```

You should see output like:
```
Serving HTTP on 0.0.0.0 port 3000 (http://0.0.0.0:3000/) ...
```

### Access the Application

1. Open your web browser
2. Navigate to `http://localhost:3000`
3. The frontend will automatically communicate with the backend API at `http://localhost:8000`

**Note**: Make sure both servers are running simultaneously for the application to work properly.

## 📖 Usage

### Getting Started

1. **Upload a Dataset**
   - Click "Upload Dataset" button
   - Select a CSV or XLSX file
   - The system automatically processes and analyzes your data

2. **Explore Your Data**
   - Navigate through different analysis tabs (Numerical, Categorical, Correlations, etc.)
   - View interactive visualizations
   - Check out automated insights

3. **Ask Questions**
   - Use the Chat tab to ask natural language questions about your dataset
   - Try queries like:
     - "What is the average age in the dataset?"
     - "Show me the distribution of disease types"
     - "What are the top 10 rows by hospital visits?"

4. **Export Results**
   - Generate PowerPoint reports with all visualizations
   - Export filtered datasets as CSV

### API Endpoints

The application provides a RESTful API:

- `POST /api/upload` - Upload and process a dataset
- `GET /api/datasets` - List all uploaded datasets
- `GET /api/dataset/{dataset_id}` - Get dataset information
- `GET /api/analyze/{dataset_id}/numerical` - Numerical analysis
- `GET /api/analyze/{dataset_id}/categorical` - Categorical analysis
- `GET /api/analyze/{dataset_id}/correlations` - Correlation analysis
- `GET /api/analyze/{dataset_id}/outliers` - Outlier detection
- `GET /api/analyze/{dataset_id}/timeseries` - Time series analysis
- `GET /api/analyze/{dataset_id}/contour` - Contour plot analysis
- `POST /api/explore` - Data exploration with filtering
- `POST /api/chat` - Chat with AI about dataset
- `POST /api/query` - Execute natural language queries
- `GET /api/insights/{dataset_id}` - Generate AI insights
- `GET /api/export/{dataset_id}/ppt` - Export PowerPoint report
- `GET /api/export/{dataset_id}/csv` - Export CSV
- `DELETE /api/dataset/{dataset_id}` - Delete dataset

## 🎯 Key Capabilities

### For Technical Users
- Programmatic API access for integration
- Advanced filtering and querying options
- Export capabilities for further analysis
- Detailed statistical summaries

### For Non-Technical Users
- Intuitive web interface
- Natural language interaction
- Automated insights generation
- One-click report generation
- Visual, interactive data exploration

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Required for AI features (chat, queries, insights)

### Customization
- Modify analysis parameters in `clean_and_EDA_generate.py`
- Adjust visualization styles in `main.py`
- Customize frontend styling in `index.html`

## 📝 Project Structure

```
AI-ENABLED-AUTOMATED-EDA-TOOL-WITH-INTELLIGENT-CHATBOT-FOR-DATA-INSIGHTS-LLM-BI/
├── main.py                 # FastAPI backend application
├── index.html              # Frontend interface
├── clean_and_EDA_generate.py  # Data cleaning and EDA generation
├── generate_report.py      # PowerPoint report generation
├── utils.py                # Gemini LLM integration utilities
├── smart_query.py          # Natural language query processing
├── requirements.txt        # Python dependencies
├── venv/                   # Virtual environment (created after setup)
└── .env                    # Environment variables (create this)
```

## 🛠️ Troubleshooting

### Backend not starting
- Ensure port 8000 is not already in use
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify your Gemini API key is set correctly

### Frontend not connecting to backend
- Ensure both servers are running
- Check that the backend is accessible at `http://localhost:8000`
- Verify CORS settings if accessing from a different origin

### Virtual environment issues
- Make sure you've activated the virtual environment (should see `(venv)` in terminal)
- Recreate virtual environment if needed: `deactivate` → delete `venv/` → recreate and reinstall

## 🚀 Future Enhancements

- Database integration for persistent storage
- User authentication and multi-user support
- Additional visualization types
- Real-time collaboration features
- Custom ML model integration
- Advanced statistical tests

## 📄 License

MIT License 2025

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact


- 📧 Email: [naganathan@gmail.com](mailto:naganathan@gmail.com)  
- 🐙 GitHub: [Naganathan194](https://github.com/Naganathan194)  

---

**Note**: This tool is designed to make data exploration accessible to both technical and non-technical users, combining the power of automated analysis with intuitive natural language interaction.

