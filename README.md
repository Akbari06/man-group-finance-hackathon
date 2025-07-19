# BMW Financial Crisis & Recovery Analysis

A comprehensive Streamlit web application that analyzes BMW's stock performance during the 2008 financial crisis and subsequent recovery period, with detailed S&P 500 benchmark comparisons which awared me 2nd place in the Man Group finance hackathon.

## 🚗 Overview

This interactive dashboard provides in-depth analysis of BMW's financial performance from 2007-2016, focusing on:

- **Crisis Impact Analysis**: Detailed examination of the 2008 Lehman Brothers collapse effects
- **Recovery Analysis**: BMW's strategic expansion into China and market recovery
- **S&P 500 Benchmarking**: Comprehensive performance comparison against the broader market
- **Event-Driven Analysis**: Impact assessment of key corporate and market events

## 📊 Features

### Interactive Visualizations
- **Stock Price Overview**: BMW price movements with key events highlighted
- **Normalized Performance Comparison**: BMW vs S&P 500 vs DAX indices
- **4-Panel Benchmark Analysis**: Cumulative returns, correlation, crisis, and recovery periods
- **Annual Returns Comparison**: Side-by-side BMW vs S&P 500 performance
- **Volatility Analysis**: Rolling 30-day volatility measurements
- **Event Impact Charts**: Cumulative returns around specific events

### Key Metrics & Analysis
- Crisis period performance (2008-2009)
- Recovery period performance (2010+)
- Total return comparisons
- Risk-adjusted performance metrics
- Rolling correlation analysis

### Data Sources
- **BMW Stock**: BMW.DE (Frankfurt Stock Exchange)
- **S&P 500 Index**: ^GSPC (Yahoo Finance)
- **DAX Index**: ^GDAXI (German stock index)

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- Internet connection for real-time data fetching

### Required Packages
```bash
pip install streamlit pandas numpy yfinance plotly matplotlib seaborn
```

### Running the Application
```bash
streamlit run app.py --server.port 5000
```

The application will be available at `http://localhost:5000`

## 📈 Usage

1. **Launch the Application**: Run the Streamlit command above
2. **Configure Analysis Parameters**: 
   - Adjust date ranges in the sidebar (default: 2007-2016)
   - Enable/disable analysis sections as needed
3. **Run Analysis**: Click the "Run Analysis" button to load data and generate charts
4. **Explore Results**: 
   - View interactive charts and metrics
   - Compare BMW performance against S&P 500 benchmark
   - Analyze crisis and recovery periods

## 🔍 Key Analysis Sections

### 1. Stock Performance Overview
- BMW stock price timeline with crisis events marked
- Normalized performance comparison against major indices

### 2. Crisis & Recovery Analysis
- Lehman Brothers crash impact (±60 days analysis)
- China JV expansion effects (±90 days analysis)
- Statistical impact measurements

### 3. S&P 500 Benchmark Comparison
- Visual 4-panel comparison chart showing:
  - Cumulative returns over full period
  - Rolling 60-day correlation
  - Crisis period performance (2008-2009)
  - Recovery period performance (2010+)

### 4. Performance Metrics
- Annual returns comparison (BMW vs S&P 500)
- Volatility analysis with rolling measurements
- Key performance indicators for different periods

## 📊 Technical Implementation

### Architecture
- **Frontend**: Streamlit web framework
- **Data Processing**: Pandas and NumPy
- **Visualizations**: Plotly for interactive charts
- **Data Source**: Yahoo Finance API via yfinance

### Key Components
- `BMWEventAnalysis` class: Core analysis engine
- Event impact analysis with customizable time windows
- Robust error handling for data inconsistencies
- Real-time data fetching with caching

### Data Processing
1. **Data Acquisition**: Historical stock data from Yahoo Finance
2. **Returns Calculation**: Daily and cumulative returns computation
3. **Event Analysis**: Time-window based impact assessment
4. **Statistical Analysis**: Volatility, correlation, and performance metrics

## 🎯 Key Events Analyzed

### Crisis Events
- **September 15, 2008**: Lehman Brothers bankruptcy filing
- Impact window: ±60 days analysis
- Metrics: Pre/post event returns, volatility, maximum drawdown

### Recovery Events
- **January 1, 2010**: BMW-Brilliance joint venture expansion in China
- Impact window: ±90 days analysis
- Strategic importance: Market diversification and growth

## 📋 Key Findings

The analysis reveals several important insights:

### Crisis Period (2008-2009)
- BMW experienced higher volatility compared to S&P 500
- Automotive sector was particularly impacted during the crisis
- Significant drawdowns during the Lehman Brothers collapse

### Recovery Period (2010+)
- BMW's China expansion strategy provided outperformance
- Strong recovery momentum compared to broader market
- Geographic diversification proved to be a key success factor

### S&P 500 Benchmark Comparison
- BMW showed resilience during recovery phases
- Premium positioning helped maintain competitive advantages
- Correlation with broader market varied significantly during crisis periods

## 🔧 Configuration

### Streamlit Configuration
The application includes optimized settings in `.streamlit/config.toml`:
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000

[theme]
base = "light"
```

### Analysis Parameters
- **Date Ranges**: Configurable start and end dates
- **Event Windows**: Customizable analysis windows around events
- **Display Options**: Toggle various analysis sections

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py --server.port 5000
```

### Production Deployment
The application is ready for deployment on:
- Streamlit Cloud
- Heroku
- AWS/GCP/Azure container services

## 📝 File Structure

```
├── app.py                 # Main application file
├── README.md             # Project documentation
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── pyproject.toml        # Python dependencies
└── replit.md            # Project architecture documentation
```

## 🔒 Data & Privacy

- All data is sourced from public financial markets via Yahoo Finance
- No personal or proprietary data is collected or stored
- Real-time data fetching ensures up-to-date analysis

## 🤝 Contributing

This project is designed for financial analysis and educational purposes. Contributions for additional features or improvements are welcome.

## 📄 License

This project is for educational and analysis purposes. Financial data is provided by Yahoo Finance and subject to their terms of service.

## ⚠️ Disclaimer

This analysis is for educational and informational purposes only. It should not be considered as financial advice or investment recommendations. Past performance does not guarantee future results.