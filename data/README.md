# Data Directory

This directory contains all datasets for the Next Up project.

## 📁 Directory Structure

```
data/
├── raw/                    # Data from SportsRadar API
├── external/               # Manually collected data
├── processed/              # Cleaned and merged datasets
├── raw_json/               # Cached API responses
└── README.md               # This file
```

---

## 📊 Expected Datasets

### Priority 1 (Essential - Collect These First)

#### 1. `raw/gleague_player_stats.csv`

**Source**: SportsRadar NBDL API  
**Description**: Season statistics for G-League players  
**Columns**: player_id, player_name, season, team, games_played, minutes_per_game, points_per_game, rebounds_per_game, assists_per_game, steals_per_game, blocks_per_game, fg_pct, three_pt_pct, ft_pct, turnovers_per_game, usage_rate, per, true_shooting_pct, etc.  
**Records**: ~1,200-1,500 player-seasons  
**Status**: ⏸️ To be collected

---

#### 2. `raw/gleague_players.csv`

**Source**: SportsRadar NBDL API  
**Description**: Player profiles and demographic information  
**Columns**: player_id, full_name, first_name, last_name, position, height_inches, weight_lbs, birth_date, age, college, draft_year, draft_round, draft_pick, draft_team, nationality, years_experience  
**Records**: ~500-700 unique players  
**Status**: ⏸️ To be collected

---

#### 3. `external/callups.csv` ⭐ **TARGET VARIABLE**

**Source**: RealGM / HoopsHype / NBA Transactions (Manual Collection)  
**Description**: Records of G-League players called up to NBA  
**Columns**:

- `player_name` (string): Full player name
- `season_year` (int): Season (e.g., 2022 for 2022-23)
- `callup_date` (date): Date of call-up (YYYY-MM-DD)
- `gleague_team` (string): G-League team name
- `nba_team` (string): NBA team that called up the player
- `contract_type` (string): "10-day", "two-way", "standard", or "hardship"
- `called_up` (binary): 1 if called up, 0 if not

**Records**: ~240-360 call-ups + ~900-1,200 non-call-ups = ~1,200-1,500 total  
**Status**: ❌ **CRITICAL - Must collect manually**

**Collection Instructions**:

1. Visit RealGM: https://basketball.realgm.com/nba/transactions
2. Filter by season and search for "G-League" or "G League"
3. Look for transactions containing:
   - "Activated from"
   - "Called up from"
   - "Signed from"
   - "Two-way contract"
4. **Important**: Also mark all G-League players NOT called up as `called_up = 0`
5. Cross-reference with HoopsHype and NBA official transactions

---

#### 4. `raw/gleague_rosters.csv`

**Source**: SportsRadar NBDL API  
**Description**: Team rosters by season  
**Columns**: season_year, season_id, gleague_team_id, gleague_team_name, nba_affiliate_team, player_id, player_name, position, jersey_number  
**Records**: ~1,500-2,000 player-team-season records  
**Status**: ⏸️ To be collected

---

### Priority 2 (Recommended - Enhances Model)

#### 5. `raw/nba_rosters.csv`

**Source**: SportsRadar NBA API  
**Description**: NBA team rosters to identify roster needs  
**Columns**: season_year, nba_team, player_name, position, jersey_number, roster_size, roster_spots_available  
**Records**: ~2,500-3,000  
**Status**: ⏸️ Optional

---

#### 6. `raw/gleague_game_logs.csv`

**Source**: SportsRadar NBDL API  
**Description**: Individual game performance (last 20-30 games per player)  
**Columns**: player_id, player_name, game_id, game_date, season, opponent, home_away, minutes, points, rebounds, assists, steals, blocks, turnovers, fg_made, fg_attempted, three_pt_made, three_pt_attempted, ft_made, ft_attempted, plus_minus, team_result  
**Records**: ~30,000-40,000 game logs  
**Status**: ⏸️ Optional

---

### Priority 3 (Advanced Features)

#### 7. `external/player_agents.csv`

**Source**: HoopsHype / Spotrac (Manual Collection)  
**Description**: Agent and agency representation  
**Columns**: player_name, agent_name, agency_name, agency_size, years_represented, num_nba_clients  
**Records**: ~300-500  
**Status**: ⏸️ Optional

---

#### 8. `external/nba_injuries.csv`

**Source**: Spotrac / RotowWorld (Manual or Scraped)  
**Description**: NBA injury reports that may create call-up opportunities  
**Columns**: date, nba_team, injured_player, position, injury_type, injury_status, expected_return_date  
**Records**: ~500-1,000  
**Status**: ⏸️ Optional

---

## 🎯 Target Dataset: `processed/master_dataset.csv`

After collecting all raw datasets, merge them into a single master dataset.

**Expected Structure**:

```
master_dataset.csv
├── Player Info: player_id, player_name, position, age, height, weight
├── Season Stats: points_per_game, rebounds_per_game, assists_per_game, etc.
├── Team Context: gleague_team, nba_affiliate
├── Draft Info: draft_year, draft_round, draft_pick
├── Agent Info: agent_name, agency_name (if available)
├── NBA Team Needs: roster_spots_available, injuries_at_position (if available)
└── TARGET: called_up (0 or 1)
```

**Records**: ~1,200-1,500 player-seasons  
**Timeframe**: 2018-2024 (6 seasons)

---

## ✅ Data Collection Checklist

Use this checklist to track your progress:

### Priority 1 (Must Have)

- [ ] `gleague_player_stats.csv` collected
- [ ] `gleague_players.csv` collected
- [ ] `gleague_rosters.csv` collected
- [ ] `callups.csv` collected ⭐ **MOST IMPORTANT**
- [ ] All datasets use consistent player IDs or names
- [ ] No duplicate player-season records
- [ ] All 6 seasons present (2018-2024)

### Priority 2 (Nice to Have)

- [ ] `nba_rosters.csv` collected
- [ ] `gleague_game_logs.csv` collected

### Priority 3 (Bonus)

- [ ] `player_agents.csv` collected
- [ ] `nba_injuries.csv` collected

### Data Quality

- [ ] Data dictionary created
- [ ] Missing values documented
- [ ] Column names standardized
- [ ] Test merge successful
- [ ] Positive class rate verified (should be 20-25%)

---

## 📝 Data Dictionary Template

Create a `DATA_DICTIONARY.md` file documenting each column:

```markdown
## gleague_player_stats.csv

| Column          | Type   | Description              | Example       | Missing Values |
| --------------- | ------ | ------------------------ | ------------- | -------------- |
| player_id       | string | Unique player identifier | "abc-123-def" | 0%             |
| player_name     | string | Full player name         | "John Doe"    | 0%             |
| season_year     | int    | Season year              | 2022          | 0%             |
| points_per_game | float  | Average points per game  | 18.5          | 2%             |
| ...             | ...    | ...                      | ...           | ...            |
```

---

## 🚨 Common Issues

### Issue 1: Player Names Don't Match Across Datasets

**Solution**: Create a mapping file or use player IDs consistently

### Issue 2: Call-up Data is Incomplete

**Solution**:

- Cross-reference multiple sources (RealGM, HoopsHype, NBA.com)
- Focus on quality over quantity
- It's okay to have fewer seasons if data is more complete

### Issue 3: Too Many API Calls

**Solution**:

- Use caching (see `raw_json/` directory)
- Prioritize recent seasons (2020-2024)
- Request production API access from SportsRadar

### Issue 4: Class Imbalance (Few Call-ups)

**Solution**:

- This is expected (~20-25% call-up rate)
- Use stratified sampling in model training
- Apply class weighting or SMOTE
- Evaluate model with F1 score, not just accuracy

---

## 📊 Data Validation

Before handing data to the team, run these checks:

```python
import pandas as pd

# Load datasets
stats = pd.read_csv('raw/gleague_player_stats.csv')
players = pd.read_csv('raw/gleague_players.csv')
callups = pd.read_csv('external/callups.csv')
rosters = pd.read_csv('raw/gleague_rosters.csv')

# Check 1: Duplicate records
assert stats.duplicated(subset=['player_id', 'season_year']).sum() == 0

# Check 2: Consistent player IDs
player_ids_stats = set(stats['player_id'].unique())
player_ids_profiles = set(players['player_id'].unique())
print(f"Players in stats but not profiles: {len(player_ids_stats - player_ids_profiles)}")

# Check 3: Call-up rate
callup_rate = callups['called_up'].mean()
print(f"Call-up rate: {callup_rate:.1%} (should be 20-25%)")

# Check 4: Missing values
print("\nMissing values:")
print(stats.isnull().sum())

# Check 5: Season coverage
print(f"\nSeasons covered: {sorted(stats['season_year'].unique())}")
```

---

## 📚 Additional Documentation

- **Detailed Collection Plan**: `../DATA_COLLECTION_PLAN.md`
- **API Reference**: `../SPORTSRADAR_API_ENDPOINTS.md`
- **Quick Summary**: `../DATASETS_SUMMARY.md`
- **Collection Notebook**: `data.ipynb`

---

## 🆘 Need Help?

1. Check the `DATA_COLLECTION_PLAN.md` for detailed instructions
2. Review `SPORTSRADAR_API_ENDPOINTS.md` for API-specific issues
3. Ask your DS3 mentor
4. Check SportsRadar developer forum

---

**Last Updated**: November 2025  
**Maintained By**: Anirudh Annabathula (Project Mentor)
