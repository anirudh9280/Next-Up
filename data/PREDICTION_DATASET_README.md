# Prediction Dataset

## Overview

The `prediction_dataset.csv` file contains a merged dataset of all G-League player statistics with a binary `called_up` column (0 or 1) indicating whether each player was called up to the NBA in that season.

## Dataset Details

- **Total Records**: 2,437 player-season combinations
- **Total Columns**: 88 (player stats + callup information)
- **Seasons Covered**: 2019, 2021, 2022, 2023, 2024
- **Unique Players**: 1,471

## Target Variable

- **`called_up`**: Binary column (0 or 1)
  - `1` = Player was called up to NBA in that season
  - `0` = Player was not called up to NBA in that season

### Class Distribution

- **Called Up (1)**: 203 records (8.33%)
- **Not Called Up (0)**: 2,234 records (91.67%)

**Note**: The class imbalance is expected and can be handled with:

- Class weighting in models
- SMOTE for oversampling
- Stratified sampling for train/test splits
- Focus on precision/recall metrics rather than just accuracy

## Key Columns

### Player Identification

- `player_name`: Player's full name
- `player_id`: Unique player identifier
- `season_year`: Season year (2019, 2021, 2022, 2023, 2024)
- `position`: Player position (G, F, C, etc.)

### Callup Information (for called_up=1)

- `callup_date`: Date of call-up
- `callup_nba_team`: NBA team that called up the player
- `callup_contract_type`: Type of contract (10-day, two-way, etc.)

### Player Statistics

The dataset includes comprehensive G-League statistics:

- **Total stats**: `total_games_played`, `total_minutes`, `total_points`, `total_rebounds`, `total_assists`, etc.
- **Average stats**: `avg_points`, `avg_rebounds`, `avg_assists`, `avg_steals`, `avg_blocks`, etc.
- **Shooting stats**: Field goal %, three-point %, free throw %, true shooting %, etc.
- **Advanced stats**: Efficiency, usage rate, etc.

## Data Sources

1. **Player Stats**: Combined from multiple season files:

   - `raw/gleague_player_season_stats_2019_REG.csv`
   - `raw/gleague_player_season_stats_2021_REG.csv`
   - `raw/gleague_player_season_stats_2022_REG.csv`
   - `raw/gleague_player_season_stats_2023_REG.csv`
   - `raw/gleague_player_season_stats_2024_REG.csv`

2. **Callup Data**: Combined from:
   - `callups_10day_tidy.csv` (10-day contracts)
   - `two_way_contracts.csv` (Two-way contracts)
   - `two_way_conversions.csv` (Two-way conversions)

## How It Was Created

The dataset was created using `create_prediction_dataset.py`, which:

1. Loads all player stats from all available season files
2. Loads all callup data (10-day, two-way, conversions)
3. Cleans and standardizes player names for matching
4. Merges callup data with player stats based on `player_name` and `season_year`
5. Creates the `called_up` column (1 if called up, 0 otherwise)
6. Adds callup details (date, team, contract type) for called-up players

## Usage for Modeling

This dataset is ready for machine learning models. Recommended approach:

1. **Feature Selection**: Choose relevant stat columns as features
2. **Handle Class Imbalance**: Use class weights or SMOTE
3. **Train/Test Split**: Use stratified split to maintain class distribution
4. **Evaluation Metrics**: Focus on:
   - Precision (of predicted call-ups, how many actually got called up?)
   - Recall (of actual call-ups, how many did we predict?)
   - F1 Score (harmonic mean of precision and recall)
   - ROC-AUC (overall model performance)

## Example Usage

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('prediction_dataset.csv')

# Select features (example)
feature_cols = [c for c in df.columns if c.startswith('avg_') and c not in ['called_up', 'callup_date', 'callup_nba_team', 'callup_contract_type']]

# Prepare data
X = df[feature_cols].fillna(0)
y = df['called_up']

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model with class weights
model = RandomForestClassifier(
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Notes

- Some callup records (786 total) didn't match with player stats because:

  - They're from seasons before 2019 (we only have stats from 2019-2024)
  - Name matching issues (though we try to handle "Last, First" vs "First Last" formats)
  - Players called up who didn't play in G-League that season

- The 8.33% call-up rate is lower than the expected 20-25% mentioned in the README, but this is still a workable dataset for binary classification.

## Next Steps

1. **Feature Engineering**: Create additional features (e.g., per-game averages, efficiency metrics)
2. **Model Selection**: Try different algorithms (Random Forest, XGBoost, Logistic Regression, etc.)
3. **Hyperparameter Tuning**: Optimize model parameters
4. **Feature Importance**: Identify which stats are most predictive of call-ups
5. **Validation**: Use cross-validation to ensure model generalizes well
