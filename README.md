# Next Up: Predicting NBA G-League Call-Ups

Next Up is a data science project that explores the most influential factors behind NBA call-ups from the G-League. Beyond performance metrics, it incorporates contextual variables such as team needs, player history, and contract types to predict which G-League players are most likely to be called up to the NBA.

The final deliverable is an interactive Streamlit web application that allows users to simulate call-up probabilities, visualize model insights, and explore historical outcomes.

**Live Application**: https://next-up.streamlit.app/

## Project Overview

This project uses machine learning to predict which G-League players will be called up to the NBA. The model analyzes player performance statistics, demographic information, and contextual factors across multiple seasons (2019-2024) to generate call-up probability predictions.

## What Was Done

### Data Collection

The project collected data from multiple sources:

1. **SportsRadar API**: G-League player statistics, rosters, and team information across multiple seasons (2019-2024)
   - Player season statistics (points, rebounds, assists, shooting percentages, etc.)
   - Player profiles (position, height, weight, age, college, draft information)
   - Team rosters and affiliations
   - Data cached in `raw_json/` directory to minimize API calls

2. **NBA.com Call-Up Data**: Official call-up transactions from NBA.com (2019-2025)
   - 10-day contracts
   - Two-way contract conversions
   - Standard contract signings
   - Aggregated at player-season level in `data/callups_nba_2019_2025_aggregated.csv`

3. **Manual Data Collection**: NBA Rosters to exlcude players in G-League
   - Historical call-up events
   - Contract type classifications
   - Date normalization and season mapping

### Data Processing and Feature Engineering

The data processing pipeline (`data/data.ipynb`) performed the following:

1. **Name Normalization**: Standardized player names across datasets to handle variations (e.g., "Last, First" vs "First Last")
   - Used unidecode for character normalization
   - Handled comma-separated name formats

2. **Data Merging**: Combined player statistics with call-up events
   - Merged on player name and season year
   - Created binary target variable `called_up` (1 if called up, 0 otherwise)
   - Resulted in 2,437 player-season combinations

3. **Feature Engineering** (`eda.ipynb`):
   - Per-game averages (points, rebounds, assists, steals, blocks)
   - Efficiency metrics (field goal percentage, three-point percentage, free throw percentage, true shooting percentage)
   - Usage rate and advanced statistics
   - Position encoding
   - Season year features

4. **Dataset Creation**: Final cleaned dataset (`prediction_dataset_callups_nba_cleaned.csv`)
   - 88 features per player-season
   - Class distribution: 8.33% called up (203 records), 91.67% not called up (2,234 records)
   - Ready for machine learning modeling

### Model Development

The modeling process (`analysis.ipynb`) involved:

1. **Data Preparation**:
   - Stratified train/validation/test split to maintain class distribution
   - Feature scaling and encoding (StandardScaler for numeric, OneHotEncoder for categorical)
   - Handled class imbalance using class weights

2. **Model Training**:
   - **Logistic Regression**: Primary model with class balancing
     - Validation F1-Score: 0.237
     - Validation Precision: 0.138
     - Validation Recall: 0.818
     - Validation ROC-AUC: 0.910
     - Validation PR-AUC: 0.402
   - **XGBoost**: Gradient boosting model for comparison
     - Validation F1-Score: 0.255
     - Validation Precision: 0.167
     - Validation Recall: 0.545
     - Validation ROC-AUC: 0.870
     - Validation PR-AUC: 0.215

3. **Model Evaluation**:
   - Used multiple metrics appropriate for imbalanced data (F1, precision, recall, ROC-AUC, PR-AUC)
   - 5-fold cross-validation on training set
   - Feature importance analysis based on correlation with target variable
   - Model artifacts saved in `models/` directory

4. **Model Selection Rationale**:
   - Logistic Regression provides high recall (0.82) and calibrated probabilities, making it ideal for identifying potential call-up candidates
   - XGBoost offers better precision (0.17) and F1-score (0.26), useful for binary classification flags
   - Both models are deployed in the Streamlit application for different use cases

### Streamlit Web Application

The interactive application (`streamlit_app.py`) provides:

1. **Home Page**: Project overview with key metrics
   - Total players in dataset
   - Number of G-League teams
   - Overall call-up rate

2. **Player Prediction Page**:
   - Individual player call-up probability predictions
   - Interactive gauge visualization showing probability
   - Player profile information (position, season, stats)
   - Comparison of Logistic Regression and XGBoost predictions
   - Historical outcome tracking (whether player was actually called up)
   - Player leaderboard sorted by predicted probability

3. **Team Analysis Page**:
   - Team-level call-up analysis
   - Average predicted call-up probability per team
   - Position distribution visualization
   - Team roster with individual player probabilities
   - Team comparison tool
   - Global team leaderboard

4. **Data Explorer Page**:
   - Interactive exploration of raw datasets
   - Players, rosters, teams, and call-up data
   - Visualizations of call-up trends by season
   - Contract type distributions
   - Position analysis of called-up players
   - Access to cleaned prediction dataset

5. **Model Insights Page**:
   - Model performance metrics (F1, precision, recall, ROC-AUC, PR-AUC)
   - Feature importance visualizations
   - Top features correlated with call-ups
   - Model comparison and rationale

## How to Use the Website

### Accessing the Application

Visit the live application at: https://next-up.streamlit.app/

### Navigation

The application has five main sections accessible via the sidebar:

1. **Home**: Overview of the project and key statistics
2. **Player Prediction**: Individual player analysis and predictions
3. **Team Analysis**: Team-level insights and comparisons
4. **Data Explorer**: Raw data exploration and visualizations
5. **Model Insights**: Model performance and feature importance

### Using Player Predictions

1. Navigate to the "Player Prediction" page
2. Select a player from the dropdown menu
3. View the player's profile information (position, season, key stats)
4. Examine the call-up probability gauge:
   - High (green): >60% probability
   - Moderate (yellow): 30-60% probability
   - Low (gray): <30% probability
5. Compare predictions from both Logistic Regression and XGBoost models
6. Check the historical outcome to see if the player was actually called up
7. Use the leaderboard to see top players by predicted probability

### Using Team Analysis

1. Navigate to the "Team Analysis" page
2. Select a G-League team from the dropdown
3. View team metrics:
   - Total players on roster
   - Average predicted call-up probability
   - Position distribution
4. Examine individual player probabilities for the selected team
5. Use the comparison tool to compare multiple teams side-by-side
6. Review the global team leaderboard to see which teams have the highest average call-up probabilities

### Exploring Data

1. Navigate to the "Data Explorer" page
2. Select a dataset to explore:
   - Players: Individual player profiles and demographics
   - Rosters: Team roster information
   - Teams: G-League team information
   - Call-Ups: Historical call-up events with trends
   - Prediction Dataset: Cleaned modeling dataset
3. Use interactive visualizations to understand data distributions
4. Filter and sort data tables as needed

### Understanding Model Insights

1. Navigate to the "Model Insights" page
2. Review performance metrics:
   - F1-Score: Balance between precision and recall
   - Precision: Of predicted call-ups, how many actually occurred
   - Recall: Of actual call-ups, how many were predicted
   - ROC-AUC: Overall model discrimination ability
   - PR-AUC: Performance on imbalanced data
3. Examine feature importance:
   - Top features most correlated with call-ups
   - Understand which statistics matter most
   - Adjust slider to see more or fewer features

## Technical Details

### Dataset Statistics

- **Total Records**: 2,437 player-season combinations
- **Unique Players**: 1,471
- **Seasons Covered**: 2019, 2021, 2022, 2023, 2024
- **Features**: 88 engineered features per player-season
- **Call-Up Rate**: 8.33% (203 called up, 2,234 not called up)

### Key Features

The model uses various feature categories:

- **Performance Metrics**: Points, rebounds, assists, steals, blocks per game
- **Shooting Statistics**: Field goal %, three-point %, free throw %, true shooting %
- **Efficiency Metrics**: Player efficiency rating, usage rate
- **Demographics**: Position, age, height, weight
- **Contextual**: Season year, team affiliation

### Model Architecture

- **Preprocessing**: StandardScaler for numeric features, OneHotEncoder for categorical features
- **Class Balancing**: Class weights to handle imbalanced data
- **Validation**: Stratified train/validation/test split with 5-fold cross-validation
- **Deployment**: Models saved as joblib artifacts and loaded in Streamlit app

## Repository Structure

```
Next-Up/
├── data/                          # Processed datasets
│   ├── prediction_dataset_callups_nba_cleaned.csv
│   ├── callups_nba_2019_2025_aggregated.csv
│   ├── feature_importance_callups_nba.csv
│   └── ...
├── raw/                           # Raw data from SportsRadar API
│   ├── gleague_player_season_stats_*.csv
│   ├── gleague_players.csv
│   ├── gleague_rosters.csv
│   └── gleague_teams.csv
├── raw_json/                      # Cached API responses
├── models/                        # Trained model artifacts
│   ├── log_reg_callup_pipeline.joblib
│   ├── xgb_callup_pipeline.joblib
│   └── model_validation_summary.csv
├── analysis.ipynb                 # Model training and evaluation
├── eda.ipynb                      # Exploratory data analysis
├── data/data.ipynb                # Data processing pipeline
├── streamlit_app.py               # Streamlit web application
├── requirements.txt               # Python dependencies
├── DESIGN_DOC.md                  # System design documentation
└── README.md                      # This file
```

## Dependencies

Core libraries used:

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning models and preprocessing
- **xgboost**: Gradient boosting classifier
- **streamlit**: Web application framework
- **plotly**: Interactive visualizations
- **matplotlib/seaborn**: Statistical visualizations
- **requests**: API calls
- **beautifulsoup4**: Web scraping

See `requirements.txt` for complete dependency list with versions.

## Installation and Local Setup

To run the application locally:

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure data files are present in `data/` and `raw/` directories
4. Run the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```
5. Open the application in your browser (typically http://localhost:8501)

## Key Findings

1. **High Recall Model**: Logistic Regression achieves 82% recall, meaning it identifies most players who will be called up, though with lower precision (14%)

2. **Feature Importance**: Player efficiency, shooting percentages, and per-game statistics are among the most predictive features

3. **Class Imbalance**: The 8.33% call-up rate requires careful handling through class weights and appropriate evaluation metrics

4. **Model Performance**: Both models achieve strong ROC-AUC scores (0.87-0.91), indicating good discrimination ability despite the imbalanced data

## Future Enhancements

Potential improvements for future iterations:

- Additional contextual features (agent representation, team injury reports, roster needs)
- Advanced interpretability tools (SHAP plots for individual predictions)
- Scenario analysis and what-if simulations
- Real-time data updates from NBA.com
- Historical validation against past seasons
- Ensemble methods combining multiple models
- Spoke to Jeferey Kee, creator of gtvhoops and an important figure in G-League operations, and got
  insights from scouts about certain factors behind call-ups. There are lots of intangible factors such as: player character, agency relationships, and favors for G-League veterans.
  - This shows us that future improvements could attempt to encompass these features in the models.

## Domain Background

The path from the NBA G-League to the NBA is highly competitive and often unpredictable. This project uses data science to analyze how performance, context, and representation impact advancement. The findings provide insight for players, agents, analysts, and teams on which attributes correlate most strongly with call-up opportunities.

## Learning Goals Achieved

- Cleaned, merged, and engineered features from heterogeneous data sources
- Built and evaluated classification models on real-world, temporally imbalanced data
- Interpreted model behavior using feature importance and correlation analysis
- Validated predictions against real-world events and assessed model generalization
- Communicated results through an interactive, user-friendly web application
- Collaborated using GitHub for version control and project transparency

## Project Information

- **Club**: DS3 @ UCSD
- **Project**: Next-Up
- **Year**: 2025-2026
- **Deployment**: Streamlit Cloud
- **Live URL**: https://next-up.streamlit.app/

## License

This project is for educational purposes as part of the DS3 course at UCSD.
