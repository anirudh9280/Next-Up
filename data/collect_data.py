import requests
import pandas as pd
import time
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in .env file")

BASE_URL = "https://api.sportradar.us/nbdl/trial/v8/en"

os.makedirs("raw", exist_ok=True)
os.makedirs("raw_json", exist_ok=True)
os.makedirs("external", exist_ok=True)

def api_call(url, cache_file=None, max_retries=5):
    if cache_file and os.path.exists(cache_file):
        print(f"  [cached]")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if cache_file:
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
            
            time.sleep(2)  # Increased delay
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = (2 ** attempt) * 5  # Exponential backoff: 5, 10, 20, 40, 80 seconds
                print(f"  Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Error: {e}. Retrying...")
                time.sleep(2)
            else:
                raise
    
    return None

print("Fetching league hierarchy...")
url = f"{BASE_URL}/league/hierarchy.json?api_key={API_KEY}"
hierarchy = api_call(url, "raw_json/hierarchy.json")

teams_data = []
for conf in hierarchy['conferences']:
    for div in conf['divisions']:
        for team in div['teams']:
            teams_data.append({
                'team_id': team['id'],
                'team_name': team.get('market', '') + ' ' + team['name'],
                'alias': team['alias']
            })

df_teams = pd.DataFrame(teams_data)
print(f"\nFound {len(df_teams)} teams\n")

all_players = []
all_rosters = []
all_stats = []

for idx, row in df_teams.iterrows():
    team_id = row['team_id']
    team_name = row['team_name']
    
    print(f"[{idx+1}/{len(df_teams)}] {team_name}")
    
    url = f"{BASE_URL}/teams/{team_id}/profile.json?api_key={API_KEY}"
    team_data = api_call(url, f"raw_json/team_{team_id}.json")
    
    if 'players' not in team_data:
        continue
    
    for player in team_data['players']:
        player_id = player.get('id')
        player_name = player.get('full_name', '')
        
        all_players.append({
            'player_id': player_id,
            'full_name': player_name,
            'position': player.get('position', ''),
            'height': player.get('height', None),
            'weight': player.get('weight', None),
            'birthdate': player.get('birthdate', ''),
            'college': player.get('college', '')
        })
        
        all_rosters.append({
            'team_id': team_id,
            'team_name': team_name,
            'player_id': player_id,
            'player_name': player_name,
            'position': player.get('position', '')
        })
        
        if 'seasons' in player:
            for season in player['seasons']:
                if 'teams' in season:
                    for team in season['teams']:
                        if 'total' in team:
                            stats = team['total']
                            all_stats.append({
                                'player_id': player_id,
                                'player_name': player_name,
                                'team_id': team_id,
                                'team_name': team_name,
                                'position': player.get('position', ''),
                                'games_played': stats.get('games_played', 0),
                                'minutes': stats.get('minutes', 0),
                                'points': stats.get('points', 0),
                                'rebounds': stats.get('rebounds', 0),
                                'assists': stats.get('assists', 0),
                                'steals': stats.get('steals', 0),
                                'blocks': stats.get('blocks', 0),
                                'turnovers': stats.get('turnovers', 0),
                                'field_goals_made': stats.get('field_goals_made', 0),
                                'field_goals_att': stats.get('field_goals_att', 0),
                                'field_goals_pct': stats.get('field_goals_pct', 0),
                                'three_points_made': stats.get('three_points_made', 0),
                                'three_points_att': stats.get('three_points_att', 0),
                                'three_points_pct': stats.get('three_points_pct', 0),
                                'free_throws_made': stats.get('free_throws_made', 0),
                                'free_throws_att': stats.get('free_throws_att', 0),
                                'free_throws_pct': stats.get('free_throws_pct', 0)
                            })

print(f"\nCollected {len(all_stats)} stat records")
print(f"Collected {len(all_players)} players")
print(f"Collected {len(all_rosters)} roster entries\n")

df_stats = pd.DataFrame(all_stats)
df_players = pd.DataFrame(all_players).drop_duplicates('player_id')
df_rosters = pd.DataFrame(all_rosters)

if len(df_stats) > 0:
    df_stats['points_per_game'] = (df_stats['points'] / df_stats['games_played'].replace(0, 1)).round(2)
    df_stats['rebounds_per_game'] = (df_stats['rebounds'] / df_stats['games_played'].replace(0, 1)).round(2)
    df_stats['assists_per_game'] = (df_stats['assists'] / df_stats['games_played'].replace(0, 1)).round(2)

df_stats.to_csv('raw/gleague_player_stats.csv', index=False)
df_players.to_csv('raw/gleague_players.csv', index=False)
df_rosters.to_csv('raw/gleague_rosters.csv', index=False)
df_teams.to_csv('raw/gleague_teams.csv', index=False)

print("Saved:")
print(f"  - gleague_player_stats.csv: {len(df_stats)} records")
print(f"  - gleague_players.csv: {len(df_players)} records")
print(f"  - gleague_rosters.csv: {len(df_rosters)} records")
print(f"  - gleague_teams.csv: {len(df_teams)} records")

if len(df_stats) > 0:
    template = df_stats[['player_name']].drop_duplicates()
    template['callup_date'] = None
    template['nba_team'] = None
    template['contract_type'] = None
    template['called_up'] = 0
    template.to_csv('external/callups_TEMPLATE.csv', index=False)
    print(f"\nCreated callups_TEMPLATE.csv with {len(template)} players")
    print("Fill in call-up information and save as external/callups.csv")

