import numpy as np
import pandas as pd
import pytest

from energy_net.gym_envs.alternating_env import AlternatingISOEnv
from energy_net.grid_entities.management.price_curve import (
    ActionBasedPriceStrategy,
)
from for_students.competitive_alternating import (
    CompetitiveAlternatingISOEnv,
    RecordingPCSEnv,
    build_combined_market_action_seen_by_pcs,
    calculate_controlled_iso_profit,
)


def values(value):
    return np.full(48, value)


def test_pcs_sees_best_prices_and_controlled_iso_dispatch():
    controlled_iso_action = np.concatenate(
        [values(0.50), values(0.40), values(2.0 / 6.0)]
    )
    opponent_iso_action = np.concatenate(
        [values(0.35), values(0.55), values(3.0 / 6.0)]
    )
    action = build_combined_market_action_seen_by_pcs(
        controlled_iso_action,
        opponent_iso_action,
        steps_per_day=48,
    )

    np.testing.assert_allclose(action[:48], 0.50)
    np.testing.assert_allclose(action[48:96], 0.40)
    np.testing.assert_allclose(action[96:], 2.0 / 6.0)


def test_lower_sell_price_wins_pcs_imports():
    steps = [{"market_energy_units": 2.0}] * 48

    profit, details = calculate_controlled_iso_profit(
        values(0.05),
        values(0.08),
        values(2.0),
        values(0.06),
        values(0.10),
        steps,
        generation_cost=0.06,
        emergency_cost=0.30,
    )

    assert details["retail_energy"] == pytest.approx(96.0)
    assert profit > 0


def test_high_sell_price_loses_pcs_imports_but_still_pays_dispatch():
    steps = [{"market_energy_units": 2.0}] * 48

    profit, details = calculate_controlled_iso_profit(
        values(0.05),
        values(0.20),
        values(2.0),
        values(0.06),
        values(0.10),
        steps,
        generation_cost=0.06,
        emergency_cost=0.30,
    )

    assert details["retail_energy"] == 0
    assert details["dispatch_cost"] > 0
    assert profit < 0


def test_insufficient_dispatch_pays_emergency_cost():
    steps = [{"market_energy_units": 2.0}] * 48

    _, details = calculate_controlled_iso_profit(
        values(0.05),
        values(0.08),
        values(1.0),
        values(0.06),
        values(0.10),
        steps,
        generation_cost=0.06,
        emergency_cost=0.30,
    )

    assert details["emergency_energy"] == pytest.approx(48.0)
    assert details["emergency_dispatch_cost"] == pytest.approx(14.4)


def test_higher_buyback_price_wins_pcs_exports():
    steps = [{"market_energy_units": -2.0}] * 48

    _, details = calculate_controlled_iso_profit(
        values(0.10),
        values(0.08),
        values(0.0),
        values(0.07),
        values(0.10),
        steps,
        generation_cost=0.06,
        emergency_cost=0.30,
    )

    assert details["export_energy"] == pytest.approx(96.0)
    assert details["export_margin"] == pytest.approx(1.92)


def test_pcs_exports_are_not_sold_below_reservation_price():
    steps = [{"market_energy_units": -2.0}] * 48

    _, details = calculate_controlled_iso_profit(
        values(0.05),
        values(0.08),
        values(0.0),
        values(0.04),
        values(0.10),
        steps,
        generation_cost=0.06,
        emergency_cost=0.30,
    )

    assert details["export_energy"] == 0.0
    assert details["export_margin"] == 0.0


def test_pcs_receives_no_payment_below_export_reservation_price(tmp_path):
    timestamps = pd.date_range("2025-01-01", periods=96, freq="30min")
    actual = tmp_path / "actual.csv"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        {"Datetime": timestamps, "Consumption": 0.0}
    ).set_index("Datetime").to_csv(actual)
    pd.DataFrame(
        {"timestamp": timestamps, "predicted_consumption": 0.0}
    ).to_csv(predictions, index=False)
    pcs_environment = RecordingPCSEnv(
        test_data_file=str(actual),
        predictions_file=str(predictions),
        verbosity=0,
    )
    pcs_environment.set_price_strategy(
        ActionBasedPriceStrategy(
            np.concatenate(
                [
                    values(0.25),
                    values(0.50),
                    values(0.0),
                ]
            ),
            use_asymmetric_pricing=True,
            steps_per_day=48,
        )
    )
    pcs_environment.reset()
    pcs_environment.pcs.storage_units[0].reset(10.0)

    _, _, _, _, information = pcs_environment.step(
        np.asarray([-1.0], dtype=np.float32)
    )

    assert information["market_energy_units"] < 0
    assert information["settled_market_energy_units"] == 0.0
    assert information["export_sold"] is False
    assert information["step_money"] == 0.0


def test_market_flow_includes_household_consumption(tmp_path):
    timestamps = pd.date_range("2025-01-01", periods=96, freq="30min")
    actual = tmp_path / "actual.csv"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        {"Datetime": timestamps, "Consumption": 2.0}
    ).set_index("Datetime").to_csv(actual)
    pd.DataFrame(
        {"timestamp": timestamps, "predicted_consumption": 2.0}
    ).to_csv(predictions, index=False)

    pcs_environment = RecordingPCSEnv(
        test_data_file=str(actual),
        predictions_file=str(predictions),
        verbosity=0,
    )
    market_action = np.concatenate(
        [
            values(0.25),
            values(0.50),
            values(0.0),
        ]
    )
    pcs_environment.set_price_strategy(
        ActionBasedPriceStrategy(
            market_action,
            use_asymmetric_pricing=True,
            steps_per_day=48,
        )
    )
    pcs_environment.reset()

    _, reward, _, _, information = pcs_environment.step(
        np.asarray([0.0], dtype=np.float32)
    )

    assert information["battery_action_energy_units"] == 0.0
    assert information["market_energy_units"] == pytest.approx(2.0)
    assert information["current_price"] == pytest.approx(0.10)
    assert information["step_money"] == pytest.approx(-0.20)
    assert reward == pytest.approx(-1.20)


def test_consumption_advances_to_the_requested_day(tmp_path):
    timestamps = pd.date_range("2025-01-01", periods=96, freq="30min")
    actual = tmp_path / "actual.csv"
    predictions = tmp_path / "predictions.csv"
    daily_consumption = np.concatenate(
        [
            np.full(48, 1.0),
            np.full(48, 2.0),
        ]
    )
    pd.DataFrame(
        {"Datetime": timestamps, "Consumption": daily_consumption}
    ).set_index("Datetime").to_csv(actual)
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "predicted_consumption": daily_consumption,
        }
    ).to_csv(predictions, index=False)

    pcs_environment = RecordingPCSEnv(
        test_data_file=str(actual),
        predictions_file=str(predictions),
        verbosity=0,
    )
    pcs_environment.reset(start_date=timestamps[0])
    _, _, _, _, first_day_information = pcs_environment.step(
        np.asarray([0.0], dtype=np.float32)
    )
    pcs_environment.reset(start_date=timestamps[48])
    _, _, _, _, second_day_information = pcs_environment.step(
        np.asarray([0.0], dtype=np.float32)
    )

    assert first_day_information["consumption_units"] == pytest.approx(1.0)
    assert second_day_information["consumption_units"] == pytest.approx(2.0)


def test_pcs_observes_retail_and_household_export_prices(tmp_path):
    timestamps = pd.date_range("2025-01-01", periods=96, freq="30min")
    actual = tmp_path / "actual.csv"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        {"Datetime": timestamps, "Consumption": 2.0}
    ).set_index("Datetime").to_csv(actual)
    pd.DataFrame(
        {"timestamp": timestamps, "predicted_consumption": 2.0}
    ).to_csv(predictions, index=False)
    pcs_environment = RecordingPCSEnv(
        test_data_file=str(actual),
        predictions_file=str(predictions),
        verbosity=0,
    )
    pcs_environment.set_price_strategy(
        ActionBasedPriceStrategy(
            np.concatenate(
                [
                    values(1.0),
                    values(0.0),
                    values(0.0),
                ]
            ),
            use_asymmetric_pricing=True,
            steps_per_day=48,
        )
    )

    observation, _ = pcs_environment.reset()

    assert observation[2] == pytest.approx(0.0)
    assert observation[3] == pytest.approx(0.20)


def test_new_daily_price_strategy_replaces_cached_prices(tmp_path):
    timestamps = pd.date_range("2025-01-01", periods=96, freq="30min")
    actual = tmp_path / "actual.csv"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        {"Datetime": timestamps, "Consumption": 2.0}
    ).set_index("Datetime").to_csv(actual)
    pd.DataFrame(
        {"timestamp": timestamps, "predicted_consumption": 2.0}
    ).to_csv(predictions, index=False)
    pcs_environment = RecordingPCSEnv(
        test_data_file=str(actual),
        predictions_file=str(predictions),
        verbosity=0,
    )
    first_day_action = np.concatenate(
        [
            values(0.10),
            values(0.20),
            values(0.0),
        ]
    )
    second_day_action = np.concatenate(
        [
            values(0.50),
            values(0.90),
            values(0.0),
        ]
    )
    pcs_environment.set_price_strategy(
        ActionBasedPriceStrategy(
            first_day_action,
            use_asymmetric_pricing=True,
            steps_per_day=48,
        )
    )
    pcs_environment.reset(start_date=timestamps[0])
    for _ in range(48):
        pcs_environment.step(np.asarray([0.0], dtype=np.float32))

    pcs_environment.set_price_strategy(
        ActionBasedPriceStrategy(
            second_day_action,
            use_asymmetric_pricing=True,
            steps_per_day=48,
        )
    )
    pcs_environment.reset(start_date=timestamps[48])
    _, _, _, _, information = pcs_environment.step(
        np.asarray([0.0], dtype=np.float32)
    )

    assert information["current_price"] == pytest.approx(0.18)


def test_market_money_uses_the_price_for_the_current_half_hour(tmp_path):
    timestamps = pd.date_range("2025-01-01", periods=96, freq="30min")
    actual = tmp_path / "actual.csv"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        {"Datetime": timestamps, "Consumption": 2.0}
    ).set_index("Datetime").to_csv(actual)
    pd.DataFrame(
        {"timestamp": timestamps, "predicted_consumption": 2.0}
    ).to_csv(predictions, index=False)
    retail_actions = np.linspace(0.10, 0.90, 48, dtype=np.float32)
    pcs_environment = RecordingPCSEnv(
        test_data_file=str(actual),
        predictions_file=str(predictions),
        verbosity=0,
    )
    pcs_environment.set_price_strategy(
        ActionBasedPriceStrategy(
            np.concatenate(
                [
                    values(0.0),
                    retail_actions,
                    values(0.0),
                ]
            ),
            use_asymmetric_pricing=True,
            steps_per_day=48,
        )
    )
    pcs_environment.reset()

    _, _, _, _, information = pcs_environment.step(
        np.asarray([0.0], dtype=np.float32)
    )

    assert information["current_price"] == pytest.approx(0.02)
    assert information["step_money"] == pytest.approx(-0.04)


def test_battery_covered_consumption_is_not_grid_reliance(tmp_path):
    timestamps = pd.date_range("2025-01-01", periods=96, freq="30min")
    actual = tmp_path / "actual.csv"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        {"Datetime": timestamps, "Consumption": 2.0}
    ).set_index("Datetime").to_csv(actual)
    pd.DataFrame(
        {"timestamp": timestamps, "predicted_consumption": 2.0}
    ).to_csv(predictions, index=False)
    pcs_environment = RecordingPCSEnv(
        test_data_file=str(actual),
        predictions_file=str(predictions),
        verbosity=0,
    )
    pcs_environment.reset()
    pcs_environment.pcs.storage_units[0].reset(4.0)

    _, reward, _, _, information = pcs_environment.step(
        np.asarray([0.0], dtype=np.float32)
    )

    assert information["market_energy_units"] == pytest.approx(0.0)
    assert information["grid_energy_used_for_consumption_units"] == 0.0
    assert information["shortage"] is False
    assert reward == pytest.approx(0.0)


def test_battery_storage_continues_between_chronological_days(tmp_path):
    timestamps = pd.date_range("2025-01-01", periods=96, freq="30min")
    actual = tmp_path / "actual.csv"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        {"Datetime": timestamps, "Consumption": 0.0}
    ).set_index("Datetime").to_csv(actual)
    pd.DataFrame(
        {"timestamp": timestamps, "predicted_consumption": 0.0}
    ).to_csv(predictions, index=False)
    pcs_environment = RecordingPCSEnv(
        test_data_file=str(actual),
        predictions_file=str(predictions),
        verbosity=0,
    )
    pcs_environment.reset(start_date=timestamps[0])
    pcs_environment.pcs.storage_units[0].reset(10.0)
    for _ in range(48):
        pcs_environment.step(np.asarray([0.0], dtype=np.float32))
    ending_storage = pcs_environment.pcs.get_total_storage()

    pcs_environment.reset(start_date=timestamps[48])

    assert pcs_environment.pcs.get_total_storage() == pytest.approx(
        ending_storage
    )


def test_battery_storage_survives_repeated_reset_for_same_next_day(tmp_path):
    timestamps = pd.date_range("2025-01-01", periods=96, freq="30min")
    actual = tmp_path / "actual.csv"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        {"Datetime": timestamps, "Consumption": 0.0}
    ).set_index("Datetime").to_csv(actual)
    pd.DataFrame(
        {"timestamp": timestamps, "predicted_consumption": 0.0}
    ).to_csv(predictions, index=False)
    pcs_environment = RecordingPCSEnv(
        test_data_file=str(actual),
        predictions_file=str(predictions),
        verbosity=0,
    )
    pcs_environment.reset(start_date=timestamps[0])
    pcs_environment.pcs.storage_units[0].reset(10.0)
    for _ in range(48):
        pcs_environment.step(np.asarray([0.0], dtype=np.float32))
    ending_storage = pcs_environment.pcs.get_total_storage()

    pcs_environment.reset()
    pcs_environment.reset(start_date=timestamps[48])

    assert pcs_environment.pcs.get_total_storage() == pytest.approx(
        ending_storage
    )


def test_dataset_wrap_resets_battery_and_consumption_day(tmp_path):
    timestamps = pd.date_range("2025-01-01", periods=96, freq="30min")
    actual = tmp_path / "actual.csv"
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        {"Datetime": timestamps, "Consumption": 0.0}
    ).set_index("Datetime").to_csv(actual)
    pd.DataFrame(
        {"timestamp": timestamps, "predicted_consumption": 0.0}
    ).to_csv(predictions, index=False)
    pcs_environment = RecordingPCSEnv(
        test_data_file=str(actual),
        predictions_file=str(predictions),
        verbosity=0,
    )
    pcs_environment.reset()
    pcs_environment.pcs.storage_units[0].reset(10.0)
    for _ in range(48):
        pcs_environment.step(np.asarray([0.0], dtype=np.float32))
    pcs_environment.reset()
    assert pcs_environment.pcs.get_total_storage() == pytest.approx(10.0)
    for _ in range(48):
        pcs_environment.step(np.asarray([0.0], dtype=np.float32))

    pcs_environment.reset()

    assert pcs_environment.current_datetime == pcs_environment.test_start_date
    assert pcs_environment.pcs.get_total_storage() == pytest.approx(0.0)


def test_competitive_environment_is_an_energy_net_alternating_env():
    assert issubclass(CompetitiveAlternatingISOEnv, AlternatingISOEnv)
