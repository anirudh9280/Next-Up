# How to Collect Call-up Data (TARGET VARIABLE)

## CRITICAL: This is Required Before Students Can Start

The call-up data is your **target variable** - without it, students cannot build a predictive model. This must be collected manually from NBA transaction records.

---

## What You're Creating

A CSV file with 442 rows (one per player in your dataset) that labels whether each player was called up to the NBA.

**File Location**: `data/external/callups.csv`

**Required Columns**:

```csv
player_name,nba_team,callup_date,contract_type,called_up
John Doe,Los Angeles Lakers,2024-12-15,10-day,1
Jane Smith,,,None,0
```

---

## Step 1: Get the Player List

Run this to create a template:

```bash
cd /Users/anirudhannabathula/Desktop/ds3/25-25_Projects/Next-Up
```

```python
import pandas as pd

players = pd.read_csv('raw/gleague_players.csv')
template = pd.DataFrame({
    'player_name': players['full_name'],
    'nba_team': '',
    'callup_date': '',
    'contract_type': '',
    'called_up': 0
})
template.to_csv('data/external/callups_TEMPLATE.csv', index=False)
print(f"Created template with {len(template)} players")
```

This creates `data/external/callups_TEMPLATE.csv` with all 442 players defaulted to `called_up=0`.

---

## 🔍 Step 2: Research Call-ups

### Primary Source: RealGM Transactions

**URL**: https://basketball.realgm.com/nba/transactions

**How to Use**:

1. Filter by date range (e.g., 2024-25 season: October 2024 - Present)
2. Use browser search (Ctrl+F / Cmd+F) for:

   - "G-League"
   - "G League"
   - "affiliate"
   - "Activated"
   - "Called up"
   - "Signed from"

3. Look for transaction types:
   - ✅ "Activated from G-League affiliate"
   - ✅ "Signed to NBA contract from [G-League Team]"
   - ✅ "Called up from [G-League Team]"
   - ❌ "Assigned to G-League" (wrong direction - ignore)

**Example Transactions**:

```
12/15/2024: Lakers activated Mac McClung from South Bay Lakers (G-League)
→ Record: Mac McClung, Los Angeles Lakers, 2024-12-15, 10-day, 1

11/20/2024: Knicks signed Landry Shamet to two-way contract
→ Record: Landry Shamet, New York Knicks, 2024-11-20, two-way, 1
```

---

### Secondary Sources

**HoopsHype Transactions**: https://hoopshype.com/

- Good for recent transactions
- Has transaction history by date

**NBA Official**: https://www.nba.com/transactions

- Official source but harder to search
- Most reliable

**Spotrac**: https://www.spotrac.com/nba/transactions/

- Good for contract details
- Shows contract type clearly

---

## Step 3: Fill in the Template

Open `data/external/callups_TEMPLATE.csv` in Excel or a text editor.

**For each player who was called up**:

1. Find their name in the list
2. Fill in:
   - `nba_team`: "Los Angeles Lakers" (full team name)
   - `callup_date`: "2024-12-15" (YYYY-MM-DD format)
   - `contract_type`: One of:
     - `"10-day"` - 10-day contract
     - `"two-way"` - Two-way contract
     - `"standard"` - Full NBA contract
     - `"hardship"` - Hardship exception
   - `called_up`: Change to `1`

**For players NOT called up**:

- Leave everything blank except `called_up=0` (already set)

---

## Step 4: Validation

Before saving as final, check:

- [ ] File has exactly 442 rows (one per player)
- [ ] No duplicate player names
- [ ] All `callup_date` fields are in YYYY-MM-DD format
- [ ] `contract_type` is one of: 10-day, two-way, standard, hardship, or empty
- [ ] `called_up` is either 0 or 1 (no empty values)
- [ ] Call-up rate is between 15-30% (typical range)
  ```python
  df = pd.read_csv('data/external/callups.csv')
  print(f"Call-up rate: {df['called_up'].mean():.1%}")  # Should be ~15-30%
  ```

---

## Step 5: Save Final File

1. **Review** your filled template
2. **Save as**: `data/external/callups.csv` (remove \_TEMPLATE from filename)
3. **Verify** it loads correctly:
   ```python
   import pandas as pd
   df = pd.read_csv('data/external/callups.csv')
   print(f"Total players: {len(df)}")
   print(f"Called up: {df['called_up'].sum()}")
   print(f"Not called up: {(df['called_up']==0).sum()}")
   print(f"Call-up rate: {df['called_up'].mean():.1%}")
   ```

---

## ⏱️ Time Estimate

- **Quick pass** (recent season only): 1-2 hours
- **Thorough** (verify each player): 3-5 hours
- **Complete** (multiple seasons, cross-reference): 5-8 hours

**Recommendation**: Start with recent season (2024-25) to get students working, then add historical data later.

---

## Quick Start (MVP Approach)

If short on time, collect just the 2024-25 season:

1. Create template (5 min)
2. Search RealGM for Oct 2024 - Present (30 min)
3. Fill in ~30-50 call-ups (1 hour)
4. Save and verify (5 min)

**Total: ~2 hours** for a working dataset

Students can start EDA with this!

---

## Expected Results

Based on typical G-League data:

- **Total players**: 442
- **Expected call-ups**: 70-130 (15-30%)
- **Most common contract**: Two-way
- **Peak call-up period**: November-January (NBA season start)

---

## If You Get Stuck

### Can't find a player's call-up?

- They likely weren't called up → leave as `called_up=0`
- Search: "[Player Name] NBA call up"
- Check team-specific G-League affiliate news

### Not sure about contract type?

- Default to "standard" if unclear
- Two-way contracts are most common for G-League call-ups
- 10-day contracts typically happen mid-season

### Want to speed this up?

- Focus on players with recent stats (they're more likely to be called up)
- Skip obviously inactive players (no stats, old birthdate)
- Use RealGM's search filters by team

---

## After Completing

Once you have `data/external/callups.csv` created:

1. **Merge with player data**:

   ```python
   players = pd.read_csv('raw/gleague_players.csv')
   callups = pd.read_csv('data/external/callups.csv')

   merged = players.merge(callups, left_on='full_name', right_on='player_name')
   merged.to_csv('processed/master_dataset.csv', index=False)
   ```

2. **Students can now**:
   - Analyze call-up patterns
   - Build predictive models
   - Identify key features
   - Validate predictions

---

## Other Solutions

- **RealGM not loading**: Try HoopsHype or NBA.com instead
- **Too many players**: Start with just the top 100 by stats
- **Ambiguous transactions**: When in doubt, mark as `called_up=0`

---

**Remember**: This is THE most important dataset for the project. Take your time and be thorough!
