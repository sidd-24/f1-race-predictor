"""
F1 Data Collector using FastF1

Fetches historical race data including:
- Race results
- Qualifying results
- Driver/Constructor standings
- Session information
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import logging
import argparse

from .config import (
    FASTF1_CACHE_DIR,
    DATA_DIR,
    SEASONS,
    MIN_SEASON,
    DEFAULT_DATA_PREFIX,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress FastF1 warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Enable FastF1 cache
fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))


def get_event_schedule(year: int) -> pd.DataFrame:
    """Get the race schedule for a given year."""
    try:
        schedule = fastf1.get_event_schedule(year)
        # Filter only race events (exclude testing)
        schedule = schedule[schedule['EventFormat'] != 'testing']
        return schedule
    except Exception as e:
        logger.error(f"Error fetching schedule for {year}: {e}")
        return pd.DataFrame()


def get_race_results(year: int, round_number: int) -> pd.DataFrame:
    """Get race results for a specific race."""
    try:
        session = fastf1.get_session(year, round_number, 'R')
        session.load(telemetry=False, weather=False, messages=False)

        results = session.results.copy()

        # Add metadata
        results['Year'] = year
        results['Round'] = round_number
        results['EventName'] = session.event['EventName']
        results['CircuitName'] = session.event.get('Location', 'Unknown')

        # Clean up position data
        results['Position'] = pd.to_numeric(results['Position'], errors='coerce')
        results['GridPosition'] = pd.to_numeric(results['GridPosition'], errors='coerce')

        return results

    except Exception as e:
        logger.warning(f"Error fetching race {year} Round {round_number}: {e}")
        return pd.DataFrame()


def get_qualifying_results(year: int, round_number: int) -> pd.DataFrame:
    """Get qualifying results for a specific race."""
    try:
        session = fastf1.get_session(year, round_number, 'Q')
        session.load(telemetry=False, weather=False, messages=False)

        results = session.results.copy()

        # Add metadata
        results['Year'] = year
        results['Round'] = round_number

        # Get qualifying position and time gap to pole
        results['QualifyingPosition'] = pd.to_numeric(results['Position'], errors='coerce')

        # Calculate gap to pole (Q3 time difference)
        if 'Q3' in results.columns:
            pole_time = results['Q3'].min()
            if pd.notna(pole_time):
                results['GapToPole'] = (results['Q3'] - pole_time).dt.total_seconds()

        return results[['DriverNumber', 'Abbreviation', 'TeamName',
                       'QualifyingPosition', 'Year', 'Round', 'Q1', 'Q2', 'Q3']]

    except Exception as e:
        logger.warning(f"Error fetching qualifying {year} Round {round_number}: {e}")
        return pd.DataFrame()


def get_driver_standings(year: int, round_number: int = None) -> pd.DataFrame:
    """
    Get driver championship standings.
    If round_number is provided, gets standings after that round.
    """
    try:
        # FastF1 doesn't directly provide standings, so we'll calculate from results
        # For now, return empty - we'll compute this from collected race data
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Error fetching driver standings: {e}")
        return pd.DataFrame()


def collect_season_data(year: int) -> dict:
    """Collect all race and qualifying data for a season."""
    logger.info(f"Collecting data for {year} season...")

    schedule = get_event_schedule(year)
    if schedule.empty:
        return {'races': pd.DataFrame(), 'qualifying': pd.DataFrame()}

    races = []
    qualifying = []

    # Get number of completed races
    completed_rounds = schedule[schedule['EventDate'] < pd.Timestamp.now()]['RoundNumber'].tolist()

    for round_num in tqdm(completed_rounds, desc=f"Processing {year}"):
        # Get race results
        race_result = get_race_results(year, round_num)
        if not race_result.empty:
            races.append(race_result)

        # Get qualifying results
        quali_result = get_qualifying_results(year, round_num)
        if not quali_result.empty:
            qualifying.append(quali_result)

    return {
        'races': pd.concat(races, ignore_index=True) if races else pd.DataFrame(),
        'qualifying': pd.concat(qualifying, ignore_index=True) if qualifying else pd.DataFrame()
    }


def collect_all_data(seasons: list = None) -> dict:
    """
    Collect data for multiple seasons.

    Returns:
        dict with 'races' and 'qualifying' DataFrames
    """
    if seasons is None:
        seasons = SEASONS

    all_races = []
    all_qualifying = []

    for year in seasons:
        season_data = collect_season_data(year)

        if not season_data['races'].empty:
            all_races.append(season_data['races'])
        if not season_data['qualifying'].empty:
            all_qualifying.append(season_data['qualifying'])

    result = {
        'races': pd.concat(all_races, ignore_index=True) if all_races else pd.DataFrame(),
        'qualifying': pd.concat(all_qualifying, ignore_index=True) if all_qualifying else pd.DataFrame()
    }

    logger.info(f"Collected {len(result['races'])} race results")
    logger.info(f"Collected {len(result['qualifying'])} qualifying results")

    return result


def build_seasons(start_year: int, end_year: int) -> list:
    """Build an inclusive season list with validation."""
    if start_year < MIN_SEASON:
        raise ValueError(f"Start year must be >= {MIN_SEASON}")
    if end_year < start_year:
        raise ValueError("End year must be greater than or equal to start year")
    return list(range(start_year, end_year + 1))


def compute_standings(race_results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute driver and constructor standings before each race.

    This adds columns for championship position and points going into each race.
    """
    if race_results.empty:
        return race_results

    # Points system (2010 onwards)
    points_map = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
        6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    }

    # Sort by year and round
    df = race_results.sort_values(['Year', 'Round']).copy()

    # Calculate points earned in each race
    df['PointsEarned'] = df['Position'].map(points_map).fillna(0)

    # Add fastest lap point (simplified - would need actual data)
    # For now, skip this

    # Calculate cumulative points for each driver within a season
    driver_standings = []
    constructor_standings = []

    for (year,), group in df.groupby(['Year']):
        rounds = sorted(group['Round'].unique())

        driver_cumulative = {}
        constructor_cumulative = {}

        for round_num in rounds:
            round_data = group[group['Round'] == round_num]

            for _, row in round_data.iterrows():
                driver = row.get('Abbreviation', row.get('DriverId', 'Unknown'))
                team = row.get('TeamName', 'Unknown')

                # Points BEFORE this race (standings going into the race)
                driver_points_before = driver_cumulative.get(driver, 0)
                constructor_points_before = constructor_cumulative.get(team, 0)

                driver_standings.append({
                    'Year': year,
                    'Round': round_num,
                    'Driver': driver,
                    'DriverPointsBefore': driver_points_before,
                    'TeamName': team,
                    'ConstructorPointsBefore': constructor_points_before
                })

                # Update cumulative points
                points = row.get('PointsEarned', 0)
                driver_cumulative[driver] = driver_cumulative.get(driver, 0) + points
                constructor_cumulative[team] = constructor_cumulative.get(team, 0) + points

    standings_df = pd.DataFrame(driver_standings)

    # Apply ranking within each race
    result_dfs = []
    for (year, round_num), group in standings_df.groupby(['Year', 'Round']):
        group = group.copy()
        group['DriverChampionshipPosition'] = group['DriverPointsBefore'].rank(
            ascending=False, method='min'
        )
        group['ConstructorChampionshipPosition'] = group.groupby('TeamName')[
            'ConstructorPointsBefore'
        ].transform('first').rank(ascending=False, method='dense')
        result_dfs.append(group)

    return pd.concat(result_dfs, ignore_index=True)


def save_data(data: dict, filename_prefix: str = "f1_data"):
    """Save collected data to CSV files."""
    for key, df in data.items():
        if not df.empty:
            filepath = DATA_DIR / f"{filename_prefix}_{key}.csv"
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {key} data to {filepath}")


def load_data(filename_prefix: str = "f1_data") -> dict:
    """Load previously saved data."""
    data = {}

    for key in ['races', 'qualifying', 'standings']:
        filepath = DATA_DIR / f"{filename_prefix}_{key}.csv"
        if filepath.exists():
            data[key] = pd.read_csv(filepath)
            logger.info(f"Loaded {key} data from {filepath}")
        else:
            data[key] = pd.DataFrame()

    return data


def main():
    """Main function to collect and save all F1 data."""
    parser = argparse.ArgumentParser(description="Collect F1 race and qualifying data")
    parser.add_argument('--start-year', type=int, default=MIN_SEASON, help='Start season year')
    parser.add_argument(
        '--end-year',
        type=int,
        default=pd.Timestamp.now().year,
        help='End season year (inclusive)'
    )
    parser.add_argument('--prefix', type=str, default=DEFAULT_DATA_PREFIX, help='Output filename prefix')
    args = parser.parse_args()

    seasons = build_seasons(args.start_year, args.end_year)

    logger.info("Starting F1 data collection...")

    # Collect race and qualifying data
    data = collect_all_data(seasons)

    # Compute standings
    if not data['races'].empty:
        standings = compute_standings(data['races'])
        data['standings'] = standings

    # Save to CSV
    save_data(data, args.prefix)

    logger.info("Data collection complete!")

    # Print summary
    print("\n" + "="*50)
    print("DATA COLLECTION SUMMARY")
    print("="*50)
    print(f"Seasons collected: {seasons}")
    print(f"Data prefix: {args.prefix}")
    print(f"Total race results: {len(data['races'])}")
    print(f"Total qualifying results: {len(data['qualifying'])}")
    if 'standings' in data:
        print(f"Standings entries: {len(data['standings'])}")
    print("="*50)


if __name__ == "__main__":
    main()
