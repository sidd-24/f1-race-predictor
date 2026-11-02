"""
Feature Engineering for F1 Race Prediction

Transforms raw F1 data into ML-ready features including:
- Grid position
- Championship standings
- Historical track performance
- Recent form (last N races)
- Qualifying gaps
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse

from .config import DATA_DIR, DEFAULT_DATA_PREFIX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared by training and prediction so both build features the same way
BACK_OF_GRID = 20             # pit-lane starts (grid 0 in FastF1) and missing grid slots
QUALIFYING_GAP_FILL = 5.0     # seconds; used when a driver has no Q3 time
RECENT_FORM_WEIGHTS = np.array([0.5, 0.3, 0.2])   # most recent race first


def clean_grid_position(grid: pd.Series) -> pd.Series:
    """FastF1 reports a pit-lane start as grid 0, which would look better than pole."""
    grid = pd.to_numeric(grid, errors='coerce')
    return grid.mask(grid <= 0).fillna(BACK_OF_GRID)


def gap_to_pole(q3_times: pd.Series) -> pd.Series:
    """Seconds behind the fastest Q3 time of the same session; NaN for drivers without a Q3 time."""
    q3 = pd.to_timedelta(q3_times, errors='coerce')
    return (q3 - q3.min()).dt.total_seconds()


def weighted_recent_form(positions_most_recent_first) -> float:
    """Weighted average finishing position over up to the last 3 races (NaN if none)."""
    positions = np.asarray(positions_most_recent_first, dtype=float)[:len(RECENT_FORM_WEIGHTS)]
    if len(positions) == 0:
        return np.nan
    weights = RECENT_FORM_WEIGHTS[:len(positions)]
    return float(np.average(positions, weights=weights / weights.sum()))


def previous_races(df: pd.DataFrame, driver: str, year: int, round_num: int, n_races: int = 3) -> pd.DataFrame:
    """
    A driver's last n races before (year, round), in chronological order.
    Falls back to the end of the previous season at the start of a season.
    """
    driver_rows = df[df['Abbreviation'] == driver].sort_values(['Year', 'Round'])
    prev = driver_rows[(driver_rows['Year'] == year) & (driver_rows['Round'] < round_num)].tail(n_races)
    if prev.empty:
        prev = driver_rows[driver_rows['Year'] == year - 1].tail(n_races)
    return prev


def load_raw_data(prefix: str = DEFAULT_DATA_PREFIX) -> dict:
    """Load raw CSV data files."""
    data = {}
    for key in ['races', 'qualifying', 'standings']:
        filepath = DATA_DIR / f"{prefix}_{key}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            
            # Force critical columns to numeric to prevent TypeError during calculations
            # 'Position' is used in track history and recent form
            if 'Position' in df.columns:
                df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
            
            # 'Year' and 'Round' are used for filtering and merging
            for col in ['Year', 'Round']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            data[key] = df
            logger.info(f"Loaded {len(df)} rows from {filepath.name}")
        else:
            logger.warning(f"File not found: {filepath}")
            data[key] = pd.DataFrame()
    return data


def merge_race_qualifying(races: pd.DataFrame, qualifying: pd.DataFrame) -> pd.DataFrame:
    """Merge race results with qualifying data."""
    if races.empty or qualifying.empty:
        return races

    # Ensure merge keys are the same type to avoid ValueError
    races = races.copy()
    qualifying = qualifying.copy()
    for col in ['Year', 'Round']:
        races[col] = pd.to_numeric(races[col], errors='coerce')
        qualifying[col] = pd.to_numeric(qualifying[col], errors='coerce')

    # Standardize column names for merging
    qualifying_clean = qualifying.rename(columns={
        'Abbreviation': 'Driver',
        'QualifyingPosition': 'QualifyingPos'
    })

    races_clean = races.copy()
    races_clean['Driver'] = races_clean['Abbreviation']

    # Merge on Year, Round, and Driver
    merged = races_clean.merge(
        qualifying_clean[['Year', 'Round', 'Driver', 'QualifyingPos', 'Q1', 'Q2', 'Q3']],
        on=['Year', 'Round', 'Driver'],
        how='left'
    )

    return merged


def add_standings(df: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    """Add championship standings information."""
    if standings.empty:
        return df

    standings_clean = standings.rename(columns={'Driver': 'DriverAbbr'})

    merged = df.merge(
        standings_clean[[
            'Year', 'Round', 'DriverAbbr',
            'DriverPointsBefore', 'DriverChampionshipPosition',
            'ConstructorPointsBefore', 'ConstructorChampionshipPosition'
        ]],
        left_on=['Year', 'Round', 'Abbreviation'],
        right_on=['Year', 'Round', 'DriverAbbr'],
        how='left'
    )

    return merged


def calculate_track_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate historical performance at each track.
    Average finishing position at this circuit in previous races.
    """
    df = df.sort_values(['Year', 'Round']).copy()

    # Group by driver and circuit
    track_avg = []

    for idx, row in df.iterrows():
        driver = row['Abbreviation']
        circuit = row.get('CircuitName', row.get('EventName', 'Unknown'))
        year = row['Year']
        round_num = row['Round']

        # Get all previous races at this circuit for this driver
        prev_races = df[
            (df['Abbreviation'] == driver) &
            (df.get('CircuitName', df.get('EventName', 'Unknown')) == circuit) &
            ((df['Year'] < year) | ((df['Year'] == year) & (df['Round'] < round_num)))
        ]

        if len(prev_races) > 0:
            avg_pos = prev_races['Position'].mean()
        else:
            avg_pos = np.nan  # No history

        track_avg.append(avg_pos)

    df['TrackHistoryAvgPosition'] = track_avg

    return df


def calculate_recent_form(df: pd.DataFrame, n_races: int = 3) -> pd.DataFrame:
    """
    Calculate recent form based on last N races using a weighted average.
    More recent races have a higher impact on the form score.
    """
    df = df.sort_values(['Year', 'Round']).copy()

    recent_form = []
    for _, row in df.iterrows():
        prev = previous_races(df, row['Abbreviation'], row['Year'], row['Round'], n_races)
        recent_form.append(weighted_recent_form(prev['Position'].values[::-1]))

    df['RecentForm'] = recent_form

    return df


def calculate_qualifying_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate gap to pole position in qualifying (in seconds).
    """
    df = df.copy()

    # If Q3 times exist, calculate gap. transform() keeps every gap on its own row;
    # the previous version appended gaps race by race, which misaligned them when
    # the rows were not already sorted by (Year, Round).
    if 'Q3' in df.columns:
        df['QualifyingGapToPole'] = df.groupby(['Year', 'Round'])['Q3'].transform(gap_to_pole)
    else:
        df['QualifyingGapToPole'] = np.nan

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create final feature set for ML model.
    """
    features = pd.DataFrame()

    # Target variable
    features['FinishingPosition'] = df['Position']

    # Grid position (most important feature)
    features['GridPosition'] = clean_grid_position(df['GridPosition'])

    # Championship standings
    features['DriverChampPosition'] = df.get('DriverChampionshipPosition', np.nan)
    features['DriverPoints'] = df.get('DriverPointsBefore', 0).fillna(0)
    features['ConstructorChampPosition'] = df.get('ConstructorChampionshipPosition', np.nan)
    features['ConstructorPoints'] = df.get('ConstructorPointsBefore', 0).fillna(0)

    # Track history
    features['TrackHistoryAvgPos'] = df.get('TrackHistoryAvgPosition', np.nan)

    # Recent form
    features['RecentForm'] = df.get('RecentForm', np.nan)

    # Qualifying gap
    features['QualifyingGap'] = df.get('QualifyingGapToPole', np.nan)

    # Metadata (for reference, not features)
    features['Year'] = df['Year']
    features['Round'] = df['Round']
    features['Driver'] = df['Abbreviation']
    features['Team'] = df['TeamName']
    features['EventName'] = df.get('EventName', 'Unknown')

    return features


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in features.
    """
    df = df.copy()

    # Fill championship positions with worst case (20)
    df['DriverChampPosition'] = df['DriverChampPosition'].fillna(20)
    df['ConstructorChampPosition'] = df['ConstructorChampPosition'].fillna(10)

    # Fill track history with median grid position
    df['TrackHistoryAvgPos'] = df['TrackHistoryAvgPos'].fillna(df['GridPosition'])

    # Fill recent form with grid position
    df['RecentForm'] = df['RecentForm'].fillna(df['GridPosition'])

    # Fill qualifying gap with large value for those not in Q3
    df['QualifyingGap'] = df['QualifyingGap'].fillna(QUALIFYING_GAP_FILL)

    return df


def prepare_training_data(prefix: str = DEFAULT_DATA_PREFIX) -> pd.DataFrame:
    """
    Full pipeline to prepare training data.
    """
    logger.info("Loading raw data...")
    data = load_raw_data(prefix)

    if data['races'].empty:
        raise ValueError("No race data found!")

    logger.info("Merging race and qualifying data...")
    df = merge_race_qualifying(data['races'], data['qualifying'])

    logger.info("Adding standings information...")
    df = add_standings(df, data['standings'])

    logger.info("Calculating track history...")
    df = calculate_track_history(df)

    logger.info("Calculating recent form...")
    df = calculate_recent_form(df)

    logger.info("Calculating qualifying gaps...")
    df = calculate_qualifying_gap(df)

    logger.info("Creating feature set...")
    features = create_features(df)

    logger.info("Filling missing values...")
    features = fill_missing_values(features)

    # Remove rows with missing target
    features = features.dropna(subset=['FinishingPosition'])

    # Save processed features
    output_path = DATA_DIR / f"{prefix}_features.csv"
    features.to_csv(output_path, index=False)
    logger.info(f"Saved features to {output_path}")

    return features


def get_feature_columns() -> list:
    """Return list of feature columns for model training."""
    return [
        'GridPosition',
        'DriverChampPosition',
        'DriverPoints',
        'ConstructorChampPosition',
        'ConstructorPoints',
        'TrackHistoryAvgPos',
        'RecentForm',
        'QualifyingGap'
    ]


def get_metadata_columns() -> list:
    """Return list of metadata columns (not used for training)."""
    return ['Year', 'Round', 'Driver', 'Team', 'EventName']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ML features from collected F1 data")
    parser.add_argument('--prefix', type=str, default=DEFAULT_DATA_PREFIX, help='Input/output filename prefix')
    args = parser.parse_args()

    features = prepare_training_data(args.prefix)

    print("\n" + "="*50)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*50)
    print(f"Total samples: {len(features)}")
    print(f"\nFeature columns: {get_feature_columns()}")
    print(f"\nSample data:")
    print(features[get_feature_columns() + ['FinishingPosition']].head(10))
    print("\nFeature statistics:")
    print(features[get_feature_columns()].describe())
