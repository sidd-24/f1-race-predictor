"""Tests for the feature and prediction fixes. Synthetic data only, no FastF1 download."""
import sys

import numpy as np
import pandas as pd
import pytest

from src import predictor
from src.feature_engineering import (
    BACK_OF_GRID,
    calculate_qualifying_gap,
    calculate_recent_form,
    clean_grid_position,
    gap_to_pole,
    previous_races,
    weighted_recent_form,
)
from src.model import predict_positions, prepare_xy


def races(rows):
    """rows: (Abbreviation, Year, Round, Position, CircuitName)"""
    return pd.DataFrame(rows, columns=['Abbreviation', 'Year', 'Round', 'Position', 'CircuitName'])


# ─── Grid position ─────────────────────────────────────────────────────────────

def test_pit_lane_start_goes_to_back_of_grid():
    grid = clean_grid_position(pd.Series([1, 0, None, 7]))
    assert grid.tolist() == [1, BACK_OF_GRID, BACK_OF_GRID, 7]


# ─── Qualifying gap ────────────────────────────────────────────────────────────

def test_gap_to_pole():
    gaps = gap_to_pole(pd.Series(["0 days 00:01:20.000", "0 days 00:01:20.500", None]))
    assert gaps.iloc[0] == 0 and gaps.iloc[1] == pytest.approx(0.5) and np.isnan(gaps.iloc[2])


def test_qualifying_gap_stays_on_the_right_row_when_unsorted():
    df = pd.DataFrame({
        'Year': [2024, 2023, 2024, 2023],
        'Round': [1, 1, 1, 1],
        'Abbreviation': ['A', 'B', 'C', 'D'],
        'Q3': ["0 days 00:01:30", "0 days 00:01:10", "0 days 00:01:31", "0 days 00:01:12"],
    })
    gaps = calculate_qualifying_gap(df).set_index('Abbreviation')['QualifyingGapToPole']
    assert gaps.to_dict() == {'A': 0.0, 'B': 0.0, 'C': 1.0, 'D': 2.0}


# ─── Recent form ───────────────────────────────────────────────────────────────

def test_weighted_recent_form():
    assert weighted_recent_form([1, 4, 10]) == pytest.approx(0.5 * 1 + 0.3 * 4 + 0.2 * 10)
    assert weighted_recent_form([2, 6]) == pytest.approx((0.5 * 2 + 0.3 * 6) / 0.8)
    assert np.isnan(weighted_recent_form([]))


def test_previous_races_falls_back_to_last_season():
    df = races([('VER', 2023, 21, 1, 'X'), ('VER', 2023, 22, 2, 'Y'), ('VER', 2024, 1, 5, 'Z')])
    assert previous_races(df, 'VER', 2024, 1)['Position'].tolist() == [1, 2]
    assert previous_races(df, 'VER', 2024, 2)['Position'].tolist() == [5]


def test_prediction_recent_form_matches_training():
    df = races([
        ('VER', 2024, 1, 3, 'A'), ('VER', 2024, 2, 1, 'B'), ('VER', 2024, 3, 8, 'C'), ('VER', 2024, 4, 2, 'D'),
    ])
    training = calculate_recent_form(df).set_index('Round')['RecentForm']
    for round_number in (2, 3, 4):
        assert predictor.calculate_recent_form('VER', df, 2024, round_number) == pytest.approx(training[round_number])


# ─── Track history ─────────────────────────────────────────────────────────────

def test_track_history_uses_circuit_and_only_earlier_races():
    df = races([
        ('HAM', 2022, 5, 2, 'Monaco'),
        ('HAM', 2023, 6, 4, 'Monaco'),
        ('HAM', 2023, 7, 15, 'Barcelona'),
        ('HAM', 2024, 8, 1, 'Monaco'),     # the race being predicted
    ])
    assert predictor.calculate_driver_track_history('HAM', 'Monaco', df, 2024, 8) == pytest.approx(3.0)
    assert np.isnan(predictor.calculate_driver_track_history('HAM', 'Suzuka', df, 2024, 8))


# ─── Predicted positions ───────────────────────────────────────────────────────

class FixedScores:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def predict(self, X):
        return self.scores


def test_predicted_positions_are_distinct_and_ties_follow_grid():
    X = pd.DataFrame({'GridPosition': [5, 2, 9, 1]})
    positions = predict_positions(FixedScores([3.2, 3.4, 1.1, 3.2]), X)
    # Scores 3.2 and 3.4 used to round to the same position; 3.2 vs 3.2 is broken by grid (1 before 5)
    assert positions.tolist() == [3, 4, 1, 2]
    assert sorted(positions) == [1, 2, 3, 4]


def test_latest_season_gets_more_weight_without_hardcoded_year():
    df = pd.DataFrame({col: [1, 2, 3] for col in [
        'GridPosition', 'DriverChampPosition', 'DriverPoints', 'ConstructorChampPosition',
        'ConstructorPoints', 'TrackHistoryAvgPos', 'RecentForm', 'QualifyingGap', 'FinishingPosition', 'Round']})
    df['Year'] = [2030, 2031, 2031]
    df['Driver'], df['Team'], df['EventName'] = 'X', 'T', 'E'
    *_, weights = prepare_xy(df)
    assert weights[0] == 1.0 and weights[1] == weights[2] > 1.0


# ─── CLI: --race ───────────────────────────────────────────────────────────────

def run_cli(monkeypatch, argv):
    calls = []
    monkeypatch.setattr(sys, 'argv', ['predictor'] + argv)
    monkeypatch.setattr(predictor, 'resolve_race_query',
                        lambda q, y: {'year': 2026, 'round': 20, 'event_name': 'Test Grand Prix'})
    monkeypatch.setattr(predictor, 'compare_predictions', lambda r, e: calls.append('compare'))
    monkeypatch.setattr(predictor, 'display_predictions', lambda r, e: calls.append('display'))
    monkeypatch.setattr(predictor, 'predict_race',
                        lambda *a, **k: calls.append(('predict_race', a)) or pd.DataFrame({'x': [1]}))
    return calls


def test_race_without_results_is_predicted(monkeypatch):
    calls = run_cli(monkeypatch, ['--race', 'Test Grand Prix', '--year', '2026', '--model', 'xgboost'])

    def no_data(*a, **k):
        raise ValueError("No data found")

    monkeypatch.setattr(predictor, 'predict_from_historical', no_data)
    predictor.main()
    assert calls[0][0] == 'predict_race' and calls[0][1][2] == 'xgboost'
    assert calls[1] == 'display'


def test_completed_race_is_compared_with_chosen_model(monkeypatch):
    calls = run_cli(monkeypatch, ['--race', 'Test Grand Prix', '--year', '2026', '--model', 'xgboost'])
    used = {}

    def historical(year, round_number, data_prefix, model_name):
        used['model'] = model_name
        return pd.DataFrame({'x': [1]})

    monkeypatch.setattr(predictor, 'predict_from_historical', historical)
    predictor.main()
    assert calls == ['compare'] and used['model'] == 'xgboost'
