import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="BMW Crisis & Recovery Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

class BMWEventAnalysis:
    def __init__(self, start_date="2007-01-01", end_date="2016-01-01"):
        self.start_date = start_date
        self.end_date = end_date
        self.data_loaded = False
        
    def load_data(self):
        """Load stock data for BMW, S&P 500, and DAX"""
        try:
            with st.spinner("Loading financial data..."):
                # Download stock data
                self.bmw = yf.download("BMW.DE", start=self.start_date, end=self.end_date, progress=False)
                self.sp500 = yf.download("^GSPC", start=self.start_date, end=self.end_date, progress=False)
                self.dax = yf.download("^GDAXI", start=self.start_date, end=self.end_date, progress=False)
                
                # Calculate returns
                self.bmw['Returns'] = self.bmw['Close'].pct_change()
                self.sp500['Returns'] = self.sp500['Close'].pct_change()
                self.dax['Returns'] = self.dax['Close'].pct_change()
                
                self.data_loaded = True
                return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
            
    def analyze_event_impact(self, event_date, event_name, window_days=30):
        """Analyze the impact of a specific event on BMW stock"""
        event_date = pd.to_datetime(event_date)
        pre_start = event_date - timedelta(days=window_days)
        post_end = event_date + timedelta(days=window_days)
        
        # Filter data around the event
        event_data = self.bmw[pre_start:post_end].copy()
        event_data['Cumulative_Returns'] = (1 + event_data['Returns']).cumprod() - 1
        
        # Split into pre and post event periods
        pre_event = event_data[event_data.index < event_date]
        post_event = event_data[event_data.index > event_date]
        
        # Calculate statistics
        stats_dict = {
            'Event': event_name,
            'Event_Date': event_date,
            'Pre_Event_Return': pre_event['Returns'].mean() * 100 if len(pre_event) > 0 else 0,
            'Post_Event_Return': post_event['Returns'].mean() * 100 if len(post_event) > 0 else 0,
            'Pre_Event_Volatility': pre_event['Returns'].std() * 100 if len(pre_event) > 0 else 0,
            'Post_Event_Volatility': post_event['Returns'].std() * 100 if len(post_event) > 0 else 0,
            'Max_Drawdown': event_data['Cumulative_Returns'].min() * 100 if len(event_data) > 0 else 0,
            'Total_Return_Window': event_data['Cumulative_Returns'].iloc[-1] * 100 if len(event_data) > 0 else 0
        }
        
        return event_data, stats_dict
    
    def calculate_annual_returns(self):
        """Calculate annual returns - FIXED VERSION"""
        try:
            # Resample to get year-end prices
            annual_data = self.bmw['Close'].resample('Y').last()
            # Calculate year-over-year returns
            annual_returns = annual_data.pct_change().dropna() * 100
            # Convert to simple DataFrame with year as index
            annual_returns.index = annual_returns.index.year
            return pd.DataFrame({'BMW': annual_returns})
        except Exception as e:
            st.error(f"Error calculating annual returns: {str(e)}")
            return pd.DataFrame()
    
    def create_comprehensive_dashboard(self):
        """Create comprehensive analysis dashboard"""
        if not self.data_loaded:
            if not self.load_data():
                return None, None
        
        # Key event dates
        lehman_crash = "2008-09-15"
        china_jv_expansion = "2010-01-01"
        
        # Analyze major events
        lehman_data, lehman_stats = self.analyze_event_impact(
            lehman_crash, "Lehman Brothers Crash", 60
        )
        
        china_data, china_stats = self.analyze_event_impact(
            china_jv_expansion, "China JV Expansion", 90
        )
        
        return lehman_stats, china_stats
    
    def plot_stock_price_overview(self):
        """Plot BMW stock price with key events highlighted"""
        fig = go.Figure()
        
        # Add BMW stock price
        fig.add_trace(go.Scatter(
            x=self.bmw.index,
            y=self.bmw['Close'],
            name='BMW Stock Price (€)',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add Lehman Brothers crash marker
        fig.add_shape(
            type="line",
            x0="2008-09-15", x1="2008-09-15",
            y0=0, y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        fig.add_annotation(
            x="2008-09-15", y=0.95,
            text="Lehman Brothers Crash",
            showarrow=False,
            yref="paper",
            textangle=-90,
            font=dict(color="red")
        )
        
        # Add China expansion marker
        fig.add_shape(
            type="line",
            x0="2010-01-01", x1="2010-01-01",
            y0=0, y1=1,
            yref="paper",
            line=dict(color="green", width=2, dash="dash")
        )
        fig.add_annotation(
            x="2010-01-01", y=0.05,
            text="China JV Expansion",
            showarrow=False,
            yref="paper",
            textangle=-90,
            font=dict(color="green")
        )
        
        fig.update_layout(
            title="BMW Stock Price (2007-2015) with Key Events",
            xaxis_title="Date",
            yaxis_title="Stock Price (€)",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_normalized_performance(self):
        """Plot normalized performance comparison"""
        # Calculate normalized prices (base 100)
        bmw_norm = (self.bmw['Close'] / self.bmw['Close'].iloc[0]) * 100
        sp500_norm = (self.sp500['Close'] / self.sp500['Close'].iloc[0]) * 100
        dax_norm = (self.dax['Close'] / self.dax['Close'].iloc[0]) * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.bmw.index,
            y=bmw_norm,
            name='BMW',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.sp500.index,
            y=sp500_norm,
            name='S&P 500',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.dax.index,
            y=dax_norm,
            name='DAX',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title="Normalized Performance Comparison (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_event_impact(self, event_data, event_name):
        """Plot cumulative returns around a specific event"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=event_data.index,
            y=event_data['Cumulative_Returns'] * 100,
            name=f'{event_name} Impact',
            line=dict(width=3),
            fill='tonexty' if event_data['Cumulative_Returns'].iloc[-1] < 0 else 'tozeroy',
            fillcolor='rgba(255,0,0,0.1)' if event_data['Cumulative_Returns'].iloc[-1] < 0 else 'rgba(0,255,0,0.1)'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title=f"Cumulative Returns: {event_name}",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=400
        )
        
        return fig
    
    def plot_volatility_analysis(self):
        """Plot rolling volatility analysis"""
        rolling_vol = self.bmw['Returns'].rolling(30).std() * np.sqrt(252) * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.bmw.index,
            y=rolling_vol,
            name='30-Day Rolling Volatility (%)',
            line=dict(color='orange', width=2)
        ))
        
        # Add average volatility line
        avg_vol = rolling_vol.mean()
        fig.add_hline(
            y=avg_vol,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Average: {avg_vol:.1f}%"
        )
        
        fig.update_layout(
            title="BMW Stock Volatility Analysis (30-Day Rolling)",
            xaxis_title="Date",
            yaxis_title="Annualized Volatility (%)",
            height=400
        )
        
        return fig
    
    def plot_annual_returns(self):
        """Plot annual returns bar chart with S&P 500 comparison"""
        annual_returns = self.calculate_annual_returns()
        
        if annual_returns.empty:
            return None
        
        # Calculate S&P 500 annual returns
        sp500_annual = self.sp500['Close'].resample('Y').last()
        sp500_returns = sp500_annual.pct_change().dropna() * 100
        sp500_returns.index = sp500_returns.index.year
        
        fig = go.Figure()
        
        # BMW returns
        colors_bmw = ['red' if x < 0 else 'blue' for x in annual_returns['BMW']]
        fig.add_trace(go.Bar(
            x=annual_returns.index.astype(str),
            y=annual_returns['BMW'],
            name='BMW Annual Returns',
            marker_color=colors_bmw,
            text=[f"{x:.1f}%" for x in annual_returns['BMW']],
            textposition='auto',
            opacity=0.8
        ))
        
        # S&P 500 returns
        colors_sp500 = ['red' if x < 0 else 'green' for x in sp500_returns]
        fig.add_trace(go.Bar(
            x=sp500_returns.index.astype(str),
            y=sp500_returns,
            name='S&P 500 Annual Returns',
            marker_color=colors_sp500,
            text=[f"{x:.1f}%" for x in sp500_returns],
            textposition='auto',
            opacity=0.6
        ))
        
        fig.add_hline(y=0, line_color="black")
        
        fig.update_layout(
            title="BMW vs S&P 500 Annual Returns Comparison",
            xaxis_title="Year",
            yaxis_title="Annual Return (%)",
            height=400,
            showlegend=True,
            barmode='group'
        )
        
        return fig
    
    def plot_benchmark_comparison(self):
        """Create comprehensive BMW vs S&P 500 benchmark comparison chart"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cumulative Returns Comparison',
                'Rolling 60-Day Correlation',
                'Crisis Period Performance (2008-2009)',
                'Recovery Period Performance (2010+)'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Calculate cumulative returns
        bmw_cumulative = (1 + self.bmw['Returns']).cumprod() - 1
        sp500_cumulative = (1 + self.sp500['Returns']).cumprod() - 1
        
        # Plot 1: Cumulative Returns Comparison
        fig.add_trace(
            go.Scatter(x=self.bmw.index, y=bmw_cumulative*100, name='BMW', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.sp500.index, y=sp500_cumulative*100, name='S&P 500', line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # Plot 2: Rolling Correlation
        rolling_corr = self.bmw['Returns'].rolling(60).corr(self.sp500['Returns'])
        fig.add_trace(
            go.Scatter(x=self.bmw.index, y=rolling_corr, name='60-Day Correlation', line=dict(color='green', width=2)),
            row=1, col=2
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=2)
        
        # Plot 3: Crisis Period (2008-2009)
        crisis_start = "2008-01-01"
        crisis_end = "2009-12-31"
        crisis_bmw = self.bmw[crisis_start:crisis_end]
        crisis_sp500 = self.sp500[crisis_start:crisis_end]
        
        bmw_crisis_cum = (1 + crisis_bmw['Returns']).cumprod() - 1
        sp500_crisis_cum = (1 + crisis_sp500['Returns']).cumprod() - 1
        
        fig.add_trace(
            go.Scatter(x=crisis_bmw.index, y=bmw_crisis_cum*100, name='BMW Crisis', line=dict(color='blue', width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=crisis_sp500.index, y=sp500_crisis_cum*100, name='S&P 500 Crisis', line=dict(color='red', width=2)),
            row=2, col=1
        )
        
        # Plot 4: Recovery Period (2010+)
        recovery_start = "2010-01-01"
        recovery_bmw = self.bmw[recovery_start:]
        recovery_sp500 = self.sp500[recovery_start:]
        
        bmw_recovery_cum = (1 + recovery_bmw['Returns']).cumprod() - 1
        sp500_recovery_cum = (1 + recovery_sp500['Returns']).cumprod() - 1
        
        fig.add_trace(
            go.Scatter(x=recovery_bmw.index, y=bmw_recovery_cum*100, name='BMW Recovery', line=dict(color='blue', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=recovery_sp500.index, y=sp500_recovery_cum*100, name='S&P 500 Recovery', line=dict(color='red', width=2)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="BMW vs S&P 500 Comprehensive Benchmark Analysis",
            showlegend=False
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Correlation", row=1, col=2)
        fig.update_yaxes(title_text="Crisis Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Recovery Return (%)", row=2, col=2)
        
        return fig

def main():
    """Main Streamlit application"""
    
    # App header
    st.title("🚗 BMW Financial Crisis & Recovery Analysis")
    st.markdown("**Interactive analysis of BMW's performance during the 2008 financial crisis and subsequent recovery**")
    
    # Sidebar controls
    st.sidebar.header("Analysis Parameters")
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2007, 1, 1),
            min_value=datetime(2005, 1, 1),
            max_value=datetime(2020, 1, 1)
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime(2016, 1, 1),
            min_value=datetime(2010, 1, 1),
            max_value=datetime(2025, 1, 1)
        )
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    show_crisis_details = st.sidebar.checkbox("Show Crisis Period Details", value=True)
    show_volatility = st.sidebar.checkbox("Show Volatility Analysis", value=True)
    show_annual_returns = st.sidebar.checkbox("Show Annual Returns", value=True)
    
    # Initialize analyzer
    analyzer = BMWEventAnalysis(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    # Load data and run analysis
    if st.button("Run Analysis", type="primary"):
        st.session_state.analysis_run = True
        st.session_state.analyzer = analyzer
    
    # Display results if analysis has been run
    if hasattr(st.session_state, 'analysis_run') and st.session_state.analysis_run:
        analyzer = st.session_state.analyzer
        
        # Run comprehensive analysis
        lehman_stats, china_stats = analyzer.create_comprehensive_dashboard()
        
        if lehman_stats and china_stats:
            # Main overview charts
            st.header("📈 Stock Performance Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(analyzer.plot_stock_price_overview(), use_container_width=True)
            
            with col2:
                st.plotly_chart(analyzer.plot_normalized_performance(), use_container_width=True)
            
            # Crisis analysis section
            if show_crisis_details:
                st.header("⚡ Crisis & Recovery Analysis")
                
                # Event impact charts
                lehman_data, _ = analyzer.analyze_event_impact("2008-09-15", "Lehman Brothers Crash", 60)
                china_data, _ = analyzer.analyze_event_impact("2010-01-01", "China JV Expansion", 90)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(analyzer.plot_event_impact(lehman_data, "Lehman Brothers Crash"), use_container_width=True)
                
                with col2:
                    st.plotly_chart(analyzer.plot_event_impact(china_data, "China JV Expansion"), use_container_width=True)
                
                # Statistics tables
                st.subheader("📊 Event Impact Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Lehman Brothers Crash Impact**")
                    lehman_df = pd.DataFrame([lehman_stats]).T
                    lehman_df.columns = ['Value']
                    lehman_df = lehman_df.drop(['Event', 'Event_Date'])
                    st.dataframe(lehman_df, use_container_width=True)
                
                with col2:
                    st.markdown("**China JV Expansion Impact**")
                    china_df = pd.DataFrame([china_stats]).T
                    china_df.columns = ['Value']
                    china_df = china_df.drop(['Event', 'Event_Date'])
                    st.dataframe(china_df, use_container_width=True)
            
            # Additional analysis sections
            col1, col2 = st.columns(2)
            
            with col1:
                if show_volatility:
                    st.plotly_chart(analyzer.plot_volatility_analysis(), use_container_width=True)
            
            with col2:
                if show_annual_returns:
                    annual_chart = analyzer.plot_annual_returns()
                    if annual_chart:
                        st.plotly_chart(annual_chart, use_container_width=True)
            
            # Key insights with S&P 500 benchmark
            st.header("🔍 Key Insights & S&P 500 Benchmark")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Crisis Impact (2008-2009)",
                    f"{lehman_stats['Total_Return_Window']:.1f}%",
                    delta=f"{lehman_stats['Post_Event_Return']:.1f}% avg daily"
                )
            
            with col2:
                st.metric(
                    "Recovery Period (2010+)",
                    f"{china_stats['Total_Return_Window']:.1f}%",
                    delta=f"{china_stats['Post_Event_Return']:.1f}% avg daily"
                )
            
            with col3:
                if analyzer.data_loaded:
                    bmw_total_return = float(((analyzer.bmw['Close'].iloc[-1] / analyzer.bmw['Close'].iloc[0]) - 1) * 100)
                    st.metric(
                        "BMW Total Return",
                        f"{bmw_total_return:.1f}%",
                        delta="Full period"
                    )
                    
            with col4:
                if analyzer.data_loaded:
                    sp500_total_return = float(((analyzer.sp500['Close'].iloc[-1] / analyzer.sp500['Close'].iloc[0]) - 1) * 100)
                    outperformance = bmw_total_return - sp500_total_return
                    st.metric(
                        "S&P 500 Total Return",
                        f"{sp500_total_return:.1f}%",
                        delta=f"{outperformance:.1f}% vs BMW"
                    )
            
            # Visual Benchmark Comparison Chart
            st.header("📊 BMW vs S&P 500 Visual Benchmark Comparison")
            st.plotly_chart(analyzer.plot_benchmark_comparison(), use_container_width=True)
            
            # Benchmark Analysis Section
            st.header("📊 S&P 500 Benchmark Analysis")
            
            if analyzer.data_loaded:
                # Calculate key benchmark metrics
                bmw_total = float(((analyzer.bmw['Close'].iloc[-1] / analyzer.bmw['Close'].iloc[0]) - 1) * 100)
                sp500_total = float(((analyzer.sp500['Close'].iloc[-1] / analyzer.sp500['Close'].iloc[0]) - 1) * 100)
                
                # Calculate crisis period performance using date ranges
                crisis_bmw_data = analyzer.bmw["2008-01-01":"2009-12-31"]
                crisis_sp500_data = analyzer.sp500["2008-01-01":"2009-12-31"]
                
                if len(crisis_bmw_data) > 0 and len(crisis_sp500_data) > 0:
                    crisis_bmw = float(((crisis_bmw_data['Close'].iloc[-1] / crisis_bmw_data['Close'].iloc[0]) - 1) * 100)
                    crisis_sp500 = float(((crisis_sp500_data['Close'].iloc[-1] / crisis_sp500_data['Close'].iloc[0]) - 1) * 100)
                else:
                    crisis_bmw = 0.0
                    crisis_sp500 = 0.0
                
                # Calculate recovery period performance
                recovery_bmw_data = analyzer.bmw["2010-01-01":]
                recovery_sp500_data = analyzer.sp500["2010-01-01":]
                
                if len(recovery_bmw_data) > 0 and len(recovery_sp500_data) > 0:
                    recovery_bmw = float(((recovery_bmw_data['Close'].iloc[-1] / recovery_bmw_data['Close'].iloc[0]) - 1) * 100)
                    recovery_sp500 = float(((recovery_sp500_data['Close'].iloc[-1] / recovery_sp500_data['Close'].iloc[0]) - 1) * 100)
                else:
                    recovery_bmw = 0.0
                    recovery_sp500 = 0.0
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Crisis Period (2008-2009)")
                    st.metric("BMW Performance", f"{crisis_bmw:.1f}%")
                    st.metric("S&P 500 Performance", f"{crisis_sp500:.1f}%")
                    st.metric("BMW vs S&P 500", f"{crisis_bmw - crisis_sp500:.1f}%")
                
                with col2:
                    st.subheader("Recovery Period (2010+)")
                    st.metric("BMW Performance", f"{recovery_bmw:.1f}%")
                    st.metric("S&P 500 Performance", f"{recovery_sp500:.1f}%")
                    st.metric("BMW vs S&P 500", f"{recovery_bmw - recovery_sp500:.1f}%")
            
            # Summary insights
            st.header("🔍 Key Findings")
            st.info("""
            **BMW vs S&P 500 Performance Analysis:**
            - BMW experienced higher volatility compared to S&P 500 during the crisis
            - The automotive sector was particularly hit hard during the 2008-2009 crisis
            - BMW's China expansion strategy provided significant outperformance in recovery
            - The company showed stronger recovery momentum than the broader market
            - Geographic diversification (especially China) was a key success factor
            
            **Crisis & Recovery Insights:**
            - BMW's premium positioning helped maintain margins during recovery
            - The Lehman Brothers collapse had a substantial negative impact on stock performance
            - BMW's expansion into the Chinese market (JV with Brilliance) was a key recovery driver
            - The company demonstrated strong resilience and recovery capabilities
            - Volatility normalized significantly in the post-crisis period
            """)
    
    else:
        st.info("👆 Click 'Run Analysis' to start the BMW financial analysis")
        
        # Show preview information
        st.header("📋 Analysis Overview")
        st.markdown("""
        This dashboard provides comprehensive analysis of BMW's stock performance during and after the 2008 financial crisis:
        
        **Key Features:**
        - **Stock Price Analysis**: Track BMW's stock price movements with key events highlighted
        - **Comparative Performance**: Compare BMW against S&P 500 and DAX indices
        - **Event Impact Analysis**: Detailed analysis of Lehman Brothers crash and China expansion effects
        - **Volatility Analysis**: Rolling volatility measurements and trends
        - **Annual Returns**: Year-over-year performance breakdown
        
        **Key Events Analyzed:**
        - 📉 **September 15, 2008**: Lehman Brothers bankruptcy filing
        - 🇨🇳 **January 1, 2010**: BMW-Brilliance joint venture expansion in China
        
        **Data Sources:**
        - BMW stock data (BMW.DE)
        - S&P 500 index (^GSPC)
        - DAX index (^GDAXI)
        """)

if __name__ == "__main__":
    main()
