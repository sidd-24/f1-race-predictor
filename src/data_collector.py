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

RACE_COLUMNS_TO_DROP = ['HeadshotUrl', 'Q1', 'Q2', 'Q3']


def configure_logging(debug: bool = False) -> None:
    """Reduce FastF1's repeated internal request logs unless debugging."""
    level = logging.DEBUG if debug else logging.WARNING
    for logger_name in ('fastf1', 'fastf1.api', 'fastf1.req', 'requests_cache'):
        logging.getLogger(logger_name).setLevel(level)


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
        results = results.drop(columns=RACE_COLUMNS_TO_DROP, errors='ignore')

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


def _existing_event_keys(data: pd.DataFrame) -> set:
    """Return (year, round) keys already present in a collected dataset."""
    if data.empty or not {'Year', 'Round'}.issubset(data.columns):
        return set()

    normalized = data[['Year', 'Round']].copy()
    normalized['Year'] = pd.to_numeric(normalized['Year'], errors='coerce')
    normalized['Round'] = pd.to_numeric(normalized['Round'], errors='coerce')
    normalized = normalized.dropna().astype(int).drop_duplicates()
    return set(zip(normalized['Year'], normalized['Round']))


def _clean_collected_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated CSV headers and invalid event keys from existing data."""
    if data.empty or not {'Year', 'Round'}.issubset(data.columns):
        return data

    cleaned = data.copy()
    cleaned['Year'] = pd.to_numeric(cleaned['Year'], errors='coerce')
    cleaned['Round'] = pd.to_numeric(cleaned['Round'], errors='coerce')
    cleaned = cleaned.dropna(subset=['Year', 'Round'])
    cleaned['Year'] = cleaned['Year'].astype(int)
    cleaned['Round'] = cleaned['Round'].astype(int)
    return cleaned.reset_index(drop=True)


def _load_unavailable_sessions(filename_prefix: str) -> set:
    """Load sessions that previously returned no data."""
    path = DATA_DIR / f"{filename_prefix}_unavailable_sessions.csv"
    if not path.exists():
        return set()
    data = pd.read_csv(path)
    if not {'Year', 'Round', 'Session'}.issubset(data.columns):
        return set()
    return {
        (int(row.Year), int(row.Round), row.Session)
        for row in data.itertuples(index=False)
        if pd.notna(row.Year) and pd.notna(row.Round)
    }


def collect_season_data(
    year: int,
    existing_races: pd.DataFrame = None,
    existing_qualifying: pd.DataFrame = None,
    unavailable_sessions: set = None,
    retry_failed: bool = False
) -> dict:
    """Collect all race and qualifying data for a season."""
    logger.info(f"Collecting data for {year} season...")

    schedule = get_event_schedule(year)
    if schedule.empty:
        return {'races': pd.DataFrame(), 'qualifying': pd.DataFrame()}

    existing_races = existing_races if existing_races is not None else pd.DataFrame()
    existing_qualifying = existing_qualifying if existing_qualifying is not None else pd.DataFrame()
    if unavailable_sessions is None:
        unavailable_sessions = set()
    existing_race_keys = _existing_event_keys(existing_races)
    existing_qualifying_keys = _existing_event_keys(existing_qualifying)

    races = []
    qualifying = []

    # Get number of completed races
    completed_rounds = schedule[schedule['EventDate'] < pd.Timestamp.now()]['RoundNumber'].tolist()
    skipped_races = 0
    skipped_qualifying = 0
    missing_races = 0
    missing_qualifying = 0

    for round_num in tqdm(completed_rounds, desc=f"Processing {year}"):
        event_key = (year, int(round_num))

        race_unavailable = (year, int(round_num), 'race') in unavailable_sessions
        qualifying_unavailable = (year, int(round_num), 'qualifying') in unavailable_sessions

        if event_key in existing_race_keys or (race_unavailable and not retry_failed):
            skipped_races += 1
        else:
            missing_races += 1
            race_result = get_race_results(year, round_num)
            if not race_result.empty:
                races.append(race_result)
            else:
                unavailable_sessions.add((year, int(round_num), 'race'))

        if event_key in existing_qualifying_keys or (qualifying_unavailable and not retry_failed):
            skipped_qualifying += 1
        else:
            missing_qualifying += 1
            quali_result = get_qualifying_results(year, round_num)
            if not quali_result.empty:
                qualifying.append(quali_result)
            else:
                unavailable_sessions.add((year, int(round_num), 'qualifying'))

    logger.info(
        "%s: skipped %d existing race sessions and %d existing qualifying sessions; "
        "fetching %d missing race sessions and %d missing qualifying sessions",
        year,
        skipped_races,
        skipped_qualifying,
        missing_races,
        missing_qualifying,
    )

    return {
        'races': pd.concat(races, ignore_index=True) if races else pd.DataFrame(),
        'qualifying': pd.concat(qualifying, ignore_index=True) if qualifying else pd.DataFrame(),
        'unavailable_sessions': unavailable_sessions
    }


def collect_all_data(
    seasons: list = None,
    filename_prefix: str = DEFAULT_DATA_PREFIX,
    retry_failed: bool = False
) -> dict:
    """
    Collect data for multiple seasons.

    Returns:
        dict with 'races' and 'qualifying' DataFrames
    """
    if seasons is None:
        seasons = SEASONS

    existing_data = load_data(filename_prefix)
    unavailable_sessions = _load_unavailable_sessions(filename_prefix)
    existing_data['races'] = _clean_collected_data(existing_data['races'])
    existing_data['races'] = existing_data['races'].drop(
        columns=RACE_COLUMNS_TO_DROP,
        errors='ignore'
    )
    existing_data['qualifying'] = _clean_collected_data(existing_data['qualifying'])
    existing_data['standings'] = _clean_collected_data(existing_data['standings'])
    all_races = [existing_data['races']] if not existing_data['races'].empty else []
    all_qualifying = [existing_data['qualifying']] if not existing_data['qualifying'].empty else []

    for year in seasons:
        season_data = collect_season_data(
            year,
            existing_races=existing_data['races'],
            existing_qualifying=existing_data['qualifying'],
            unavailable_sessions=unavailable_sessions,
            retry_failed=retry_failed
        )

        if not season_data['races'].empty:
            all_races.append(season_data['races'])
        if not season_data['qualifying'].empty:
            all_qualifying.append(season_data['qualifying'])

    result = {
        'races': pd.concat(all_races, ignore_index=True) if all_races else pd.DataFrame(),
        'qualifying': pd.concat(all_qualifying, ignore_index=True) if all_qualifying else pd.DataFrame(),
        'unavailable_sessions': pd.DataFrame(
            list(unavailable_sessions),
            columns=['Year', 'Round', 'Session']
        )
    }

    for key in ('races', 'qualifying'):
        if not result[key].empty and {'Year', 'Round'}.issubset(result[key].columns):
            result[key] = result[key].drop_duplicates(
                subset=['Year', 'Round', 'Abbreviation']
                if 'Abbreviation' in result[key].columns
                else ['Year', 'Round'],
                keep='last'
            ).reset_index(drop=True)

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
    """Save the complete merged datasets to CSV files."""
    for key, df in data.items():
        if df.empty:
            continue

        filepath = DATA_DIR / f"{filename_prefix}_{key}.csv"
        if key == 'races':
            df = df.drop(columns=RACE_COLUMNS_TO_DROP, errors='ignore')
        sort_columns = [
            column for column in ('Year', 'Round', 'Position', 'QualifyingPosition')
            if column in df.columns
        ]
        if sort_columns:
            df = df.copy()
            for column in ('Year', 'Round', 'Position', 'QualifyingPosition'):
                if column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
            df = df.sort_values(sort_columns, na_position='last').reset_index(drop=True)
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
    parser.add_argument(
        '--retry-failed',
        action='store_true',
        help='Retry sessions previously marked unavailable by FastF1'
    )
    parser.add_argument('--debug', action='store_true', help='Show detailed FastF1 request logs')
    args = parser.parse_args()
    configure_logging(args.debug)

    seasons = build_seasons(args.start_year, args.end_year)

    logger.info("Starting F1 data collection...")

    # Collect race and qualifying data
    data = collect_all_data(seasons, args.prefix, args.retry_failed)

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
