import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Next Up: G-League Call-Up Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load all datasets"""
    try:
        # Get project root (go up from app/app.py to Next-Up/)
        base_path = Path(__file__).parent.parent
        
        # Debug: show what path we're looking for
        # st.write(f"Loading from: {base_path}")
        
        players = pd.read_csv(base_path / 'raw' / 'gleague_players.csv')
        rosters = pd.read_csv(base_path / 'raw' / 'gleague_rosters.csv')
        teams = pd.read_csv(base_path / 'raw' / 'gleague_teams.csv')
        
        # Try to load call-up data
        try:
            callups = pd.read_csv(base_path / 'data' / 'external' / 'callups.csv')
        except FileNotFoundError:
            callups = None
            
        return players, rosters, teams, callups
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error(f"Looking in: {Path(__file__).parent.parent}")
        return None, None, None, None

# TODO: Load trained model (Week 3-4)
@st.cache_resource
def load_model():
    """Load trained model for predictions"""
    # TODO: Implement model loading once model is trained
    # Example: return joblib.load('models/callup_model.pkl')
    return None

# Sidebar navigation
st.sidebar.title("🏀 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "👤 Player Prediction", "🏀 Team Analysis", "📊 Data Explorer", "🔍 Model Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "**Next Up** predicts which G-League players are most likely to be called up to the NBA. "
    "Built by DS3 @ UCSD."
)

# Load data
players_df, rosters_df, teams_df, callups_df = load_data()

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown('<p class="main-header">🏀 Next Up</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predicting NBA G-League Call-Ups</p>', unsafe_allow_html=True)
    
    # Project overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Players",
            value=len(players_df) if players_df is not None else "N/A"
        )
    
    with col2:
        st.metric(
            label="G-League Teams",
            value=len(teams_df) if teams_df is not None else "N/A"
        )
    
    with col3:
        if callups_df is not None:
            callup_rate = callups_df['called_up'].mean() * 100
            st.metric(
                label="Call-Up Rate",
                value=f"{callup_rate:.1f}%"
            )
        else:
            st.metric(label="Call-Up Rate", value="N/A")
    
    st.markdown("---")
    
    # Project description
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📖 About This Project")
        st.write("""
        This application predicts which G-League players are most likely to be called up 
        to the NBA based on:
        
        - **Performance Metrics**: Stats, efficiency, consistency
        - **Player Demographics**: Position, age, physical attributes
        - **Team Context**: G-League team, NBA affiliate
        - **Contextual Factors**: Agent representation, team needs (future)
        
        Use the sidebar to explore player predictions, team analytics, and model insights.
        """)
    
    with col2:
        st.subheader("🎯 Features")
        st.write("""
        **Current Features:**
        - 📊 Data exploration and visualization
        - 🏀 Team-level call-up analysis
        - 👥 Player demographic insights
        
        **Coming Soon (Students will build):**
        - 🤖 ML model predictions for individual players
        - 📈 Feature importance analysis (SHAP plots)
        - 🔮 Call-up probability scores
        - 📉 Historical validation metrics
        """)
    
    # TODO section for students
    with st.expander("📝 TODO: Features to Implement (Week 5-6)"):
        st.markdown("""
        ### For Students to Complete:
        1. **Player Prediction Page**:
           - [ ] Implement model loading and prediction
           - [ ] Add player selection dropdown
           - [ ] Display prediction probability with confidence interval
           - [ ] Show top contributing features
        
        2. **Team Analysis Page**:
           - [ ] Add team comparison tool
           - [ ] Historical call-up trends by team
           - [ ] NBA affiliate impact analysis
        
        3. **Model Insights Page**:
           - [ ] SHAP plots for feature importance
           - [ ] Model performance metrics
           - [ ] Confusion matrix visualization
           - [ ] ROC curve and precision-recall curves
        
        4. **Advanced Features**:
           - [ ] Filter by position, team, date range
           - [ ] Export predictions to CSV
           - [ ] Model confidence indicators
        """)

# ==================== PLAYER PREDICTION PAGE ====================
elif page == "👤 Player Prediction":
    st.header("👤 Player Call-Up Prediction")
    st.write("Predict the likelihood of a G-League player being called up to the NBA")
    
    if players_df is None:
        st.error("Unable to load player data")
    else:
        # Player selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            player_name = st.selectbox(
                "Select a player:",
                options=sorted(players_df['full_name'].dropna().unique())
            )
        
        with col2:
            st.write("")
            st.write("")
            predict_button = st.button("🔮 Predict Call-Up Probability", type="primary")
        
        if predict_button or player_name:
            # Get player data
            player_info = players_df[players_df['full_name'] == player_name].iloc[0]
            
            # Display player info
            st.markdown("---")
            st.subheader(f"Player Profile: {player_name}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Position", player_info.get('position', 'N/A'))
            with col2:
                st.metric("Height", player_info.get('height', 'N/A'))
            with col3:
                st.metric("Weight", player_info.get('weight', 'N/A'))
            with col4:
                st.metric("College", player_info.get('college', 'N/A'))
            
            st.markdown("---")
            
            # TODO: Model prediction
            model = load_model()
            
            if model is None:
                st.warning("⚠️ Model not yet trained. This is a placeholder for Week 3-4.")
                st.info("""
                **TODO for Students (Week 3-4):**
                1. Train your model in `analysis.ipynb`
                2. Save model: `joblib.dump(model, 'models/callup_model.pkl')`
                3. Update `load_model()` function to load your trained model
                4. Implement prediction logic below
                """)
                
                # Placeholder prediction
                st.subheader("📊 Prediction (Placeholder)")
                
                # Mock probability
                mock_prob = np.random.uniform(0.15, 0.85)
                
                # Gauge chart for probability
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = mock_prob * 100,
                    title = {'text': "Call-Up Probability"},
                    delta = {'reference': 25, 'increasing': {'color': "green"}},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "lightyellow"},
                            {'range': [50, 75], 'color': "lightgreen"},
                            {'range': [75, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"""
                **Prediction**: {'**HIGH**' if mock_prob > 0.6 else '**MODERATE**' if mock_prob > 0.3 else '**LOW**'} likelihood of call-up
                
                **Note**: This is placeholder data. Replace with actual model predictions.
                """)
                
            else:
                # TODO: Implement actual prediction
                st.subheader("📊 Prediction")
                st.write("TODO: Implement model.predict() and display results")

# ==================== TEAM ANALYSIS PAGE ====================
elif page == "🏀 Team Analysis":
    st.header("🏀 Team Call-Up Analysis")
    
    if rosters_df is None or teams_df is None:
        st.error("Unable to load team data")
    else:
        # Team selection
        team_name = st.selectbox(
            "Select a team:",
            options=sorted(teams_df['team_name'].dropna().unique())
        )
        
        # Get team roster
        team_roster = rosters_df[rosters_df['team_name'] == team_name]
        
        st.markdown("---")
        
        # Team metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Players", len(team_roster))
        
        with col2:
            # TODO: Calculate call-up rate once callup data is merged
            st.metric("Call-Up Rate", "TBD")
        
        with col3:
            # Position diversity
            positions = team_roster['position'].nunique()
            st.metric("Positions", positions)
        
        st.markdown("---")
        
        # Position distribution
        st.subheader("📊 Position Distribution")
        position_counts = team_roster['position'].value_counts()
        
        fig = px.bar(
            x=position_counts.index,
            y=position_counts.values,
            labels={'x': 'Position', 'y': 'Number of Players'},
            title=f"{team_name} - Position Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Roster table
        st.subheader("📋 Current Roster")
        st.dataframe(
            team_roster[['player_name', 'position']].reset_index(drop=True),
            use_container_width=True
        )
        
        # TODO section
        with st.expander("📝 TODO: Additional Team Analytics (Week 4-5)"):
            st.markdown("""
            **Students should add:**
            - [ ] Historical call-up trends for this team
            - [ ] Comparison with other teams
            - [ ] Average stats for team players
            - [ ] Success rate by position
            - [ ] Timeline of call-ups
            - [ ] NBA affiliate relationship impact
            """)

# ==================== DATA EXPLORER PAGE ====================
elif page == "📊 Data Explorer":
    st.header("📊 Data Explorer")
    st.write("Explore the G-League dataset")
    
    # Dataset selector
    dataset = st.selectbox(
        "Select dataset to explore:",
        ["Players", "Rosters", "Teams", "Call-Ups (if available)"]
    )
    
    if dataset == "Players" and players_df is not None:
        st.subheader("👥 Players Dataset")
        st.write(f"Total players: {len(players_df)}")
        
        # Show data
        st.dataframe(players_df, use_container_width=True)
        
        # Basic visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Position Distribution")
            fig = px.pie(players_df, names='position', title='Players by Position')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Height Distribution")
            fig = px.histogram(players_df, x='height', title='Player Heights')
            st.plotly_chart(fig, use_container_width=True)
    
    elif dataset == "Rosters" and rosters_df is not None:
        st.subheader("📋 Rosters Dataset")
        st.write(f"Total roster entries: {len(rosters_df)}")
        st.dataframe(rosters_df, use_container_width=True)
    
    elif dataset == "Teams" and teams_df is not None:
        st.subheader("🏀 Teams Dataset")
        st.write(f"Total teams: {len(teams_df)}")
        st.dataframe(teams_df, use_container_width=True)
    
    elif dataset == "Call-Ups (if available)":
        if callups_df is not None:
            st.subheader("🎯 Call-Ups Dataset")
            st.write(f"Total records: {len(callups_df)}")
            st.dataframe(callups_df, use_container_width=True)
            
            # Call-up rate
            callup_rate = callups_df['called_up'].mean() * 100
            st.metric("Overall Call-Up Rate", f"{callup_rate:.2f}%")
        else:
            st.warning("Call-up data not yet available")

# ==================== MODEL INSIGHTS PAGE ====================
elif page == "🔍 Model Insights":
    st.header("🔍 Model Insights & Performance")
    
    st.info("""
    **TODO for Students (Week 3-4):**
    This page will display model performance metrics and interpretability visualizations.
    """)
    
    # Placeholder sections
    tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "🎯 Feature Importance", "📈 Model Validation"])
    
    with tab1:
        st.subheader("Model Performance Metrics")
        st.write("TODO: Add confusion matrix, ROC curve, precision-recall curve")
        
        # Placeholder metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", "TBD")
        with col2:
            st.metric("F1 Score", "TBD")
        with col3:
            st.metric("Precision", "TBD")
        with col4:
            st.metric("Recall", "TBD")
    
    with tab2:
        st.subheader("Feature Importance")
        st.write("TODO: Add SHAP plots showing which features most influence predictions")
        st.code("""
# Example code for students:
import shap

# After training model:
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Display in Streamlit:
st.pyplot(shap.summary_plot(shap_values, X_test))
        """)
    
    with tab3:
        st.subheader("Historical Validation")
        st.write("TODO: Show how model predictions compared to actual call-ups in test season")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with by DS3 @ UCSD | Project: Next Up | 2024-2025</p>
</div>
""", unsafe_allow_html=True)

