"""Accounting and PCS plumbing for the competitive student example."""

from datetime import timedelta

from gymnasium import spaces
import numpy as np

from energy_net.gym_envs.pcs_env import PCSEnv


# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
# Number of half-hour market periods in one simulated day.
STEPS_PER_DAY = 48

# Fixed amount the ISO receives per unit when it resells PCS-exported energy to
# the assumed larger external electricity market.
EXTERNAL_RESALE_PRICE = 0.12

# The PCS keeps or curtails excess energy unless an ISO offers at least this
# amount. This prevents the household from being forced to sell for nothing.
MINIMUM_PCS_EXPORT_PRICE = 0.06

# Cost paid by the learning ISO for every unit it dispatches ahead of time,
# whether or not that dispatched energy is ultimately sold.
DEFAULT_GENERATION_COST = 0.0375

# Cost paid per unit when won PCS demand exceeds the learning ISO's dispatch.
DEFAULT_EMERGENCY_COST = 0.30

# Fraction of a transaction assigned to the learning ISO when both ISOs offer
# exactly the same winning price.
TIED_MARKET_SHARE = 0.50

# Numerical tolerance used when deciding whether grid supply was required.
ENERGY_FLOW_TOLERANCE = 0.000001


class ChronologicalConsumptionDynamics:
    """Apply PCSEnv's episode-local time to the correct chronological day."""

    def __init__(self, wrapped_dynamics):
        self.wrapped_dynamics = wrapped_dynamics
        self.day_offset = 0.0

    def get_value(self, **kwargs):
        adjusted_arguments = dict(kwargs)
        adjusted_arguments["time"] = (
            adjusted_arguments.get("time", 0.0) + self.day_offset
        )
        return self.wrapped_dynamics.get_value(**adjusted_arguments)


# AlternatingISOEnv normally keeps only the total PCS money for the day. The
# competitive reward needs each half-hour battery flow so it can determine
# which ISO won that transaction.
class RecordingPCSEnv(PCSEnv):
    """PCSEnv with actual market flow retained for competitive settlement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._has_completed_market_step = False
        self._battery_preserved_for_date = None
        self._chronological_consumption_dynamics = []
        for consumption_unit in self.pcs.consumption_units:
            chronological_dynamics = ChronologicalConsumptionDynamics(
                consumption_unit.dynamics
            )
            consumption_unit.dynamics = chronological_dynamics
            self._chronological_consumption_dynamics.append(
                chronological_dynamics
            )

        observation_low = np.insert(
            self.observation_space.low,
            3,
            -np.inf,
        )
        observation_high = np.insert(
            self.observation_space.high,
            3,
            np.inf,
        )
        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high,
            dtype=np.float32,
        )

    def _set_consumption_day(self, start_date):
        day_offset = (
            start_date - self.test_start_date
        ).total_seconds() / timedelta(days=1).total_seconds()
        for dynamics in self._chronological_consumption_dynamics:
            dynamics.day_offset = day_offset

    def _clear_cached_prices(self):
        self.cached_day_prices = None
        self.cached_buy_prices = None
        self.cached_sell_prices = None
        self.last_price_update_date = None

    def set_price_strategy(self, strategy):
        super().set_price_strategy(strategy)
        self._clear_cached_prices()

    def _get_obs(self):
        observation = super()._get_obs()
        retail_price = self._get_current_price("buy")
        household_export_price = self._get_current_price("sell")
        observation[2] = retail_price
        return np.insert(
            observation,
            3,
            household_export_price,
        ).astype(np.float32)

    def reset(self, *args, **kwargs):
        requested_start_date = kwargs.get("start_date")
        if requested_start_date is None and args:
            requested_start_date = args[0]
        if requested_start_date is None:
            requested_start_date = self.current_datetime
            if requested_start_date >= self.test_end_date:
                requested_start_date = self.test_start_date

        preserve_battery = (
            (
                self._has_completed_market_step
                or self._battery_preserved_for_date == requested_start_date
            )
            and requested_start_date == self.current_datetime
            and requested_start_date != self.test_start_date
        )
        previous_battery_levels = [
            storage_unit.get_state()
            for storage_unit in self.pcs.storage_units
        ]

        self._set_consumption_day(requested_start_date)
        self._clear_cached_prices()
        self.day_steps = []
        _, reset_information = super().reset(*args, **kwargs)

        if preserve_battery:
            for storage_unit, previous_level in zip(
                self.pcs.storage_units,
                previous_battery_levels,
            ):
                storage_unit.reset(previous_level)

        self._battery_preserved_for_date = (
            requested_start_date if preserve_battery else None
        )
        self._has_completed_market_step = False
        return self._get_obs(), reset_information

    def step(self, action):
        self._battery_preserved_for_date = None
        retail_price = self._get_current_price("buy")
        household_export_price = self._get_current_price("sell")
        (
            observation,
            original_reward,
            terminated,
            truncated,
            information,
        ) = super().step(action)

        # PCSUnit returns only the battery change requested by the action. It
        # intentionally excludes household consumption. Reconstruct the actual
        # net grid flow from the complete household energy balance instead.
        battery_action_energy = information[
            "energy_sold_or_bought_units"
        ]
        local_production = float(self.pcs.get_production())
        battery_level_after_action = (
            information["storage_before_units"]
            + battery_action_energy
        )
        battery_energy_used_for_consumption = max(
            battery_level_after_action
            - information["storage_after_units"],
            0.0,
        )
        grid_energy_used_for_consumption = max(
            information["consumption_units"]
            - local_production
            - battery_energy_used_for_consumption,
            0.0,
        )
        market_energy = (
            information["consumption_units"]
            + information["storage_after_units"]
            - information["storage_before_units"]
            - local_production
        )
        export_is_sold = bool(
            market_energy < 0
            and household_export_price >= MINIMUM_PCS_EXPORT_PRICE
        )
        if market_energy > 0:
            market_price = retail_price
            settled_market_energy = market_energy
        elif export_is_sold:
            market_price = household_export_price
            settled_market_energy = market_energy
        else:
            market_price = 0.0
            settled_market_energy = 0.0

        original_money = information["step_money"]
        original_penalty = original_money - original_reward
        original_grid_reliance = bool(information["shortage"])
        grid_reliance = bool(
            grid_energy_used_for_consumption > ENERGY_FLOW_TOLERANCE
        )
        corrected_penalty = (
            self.shortage_penalty if grid_reliance else 0.0
        )
        corrected_money = -settled_market_energy * market_price
        corrected_reward = corrected_money - corrected_penalty

        # PCSEnv already updated these totals using only battery-action energy.
        # Replace that contribution with the complete market transaction.
        self.total_money_earned += corrected_money - original_money
        self.total_shortage_penalty += (
            corrected_penalty - original_penalty
        )
        self.shortage_count += (
            int(grid_reliance) - int(original_grid_reliance)
        )
        self.total_reward += corrected_reward - original_reward

        information.update(
            {
                "battery_action_energy_units": battery_action_energy,
                "battery_energy_used_for_consumption_units": (
                    battery_energy_used_for_consumption
                ),
                "grid_energy_used_for_consumption_units": (
                    grid_energy_used_for_consumption
                ),
                "local_production_units": local_production,
                "market_energy_units": market_energy,
                "settled_market_energy_units": settled_market_energy,
                "export_sold": export_is_sold,
                "current_price": market_price,
                "step_money": corrected_money,
                "shortage": grid_reliance,
                "total_money_so_far": self.total_money_earned,
            }
        )
        self.day_steps.append(dict(information))
        self._has_completed_market_step = terminated
        return (
            observation,
            corrected_reward,
            terminated,
            truncated,
            information,
        )


def build_combined_market_action_seen_by_pcs(
    controlled_iso_action,
    opponent_iso_action,
    steps_per_day,
):
    """Combine two ISO offers into the single action accepted by EnergyNet."""

    controlled_iso_buyback_actions = controlled_iso_action[:steps_per_day]
    controlled_iso_retail_actions = controlled_iso_action[
        steps_per_day : 2 * steps_per_day
    ]
    controlled_iso_dispatch_actions = controlled_iso_action[
        2 * steps_per_day :
    ]
    opponent_iso_buyback_actions = opponent_iso_action[:steps_per_day]
    opponent_iso_retail_actions = opponent_iso_action[
        steps_per_day : 2 * steps_per_day
    ]

    # These are the only competition rules in the example:
    #   - households buy from the cheaper ISO;
    #   - households sell to the ISO offering the higher buyback price.
    return np.concatenate(
        [
            np.maximum(
                controlled_iso_buyback_actions,
                opponent_iso_buyback_actions,
            ),
            np.minimum(
                controlled_iso_retail_actions,
                opponent_iso_retail_actions,
            ),
            controlled_iso_dispatch_actions,
        ]
    ).astype(np.float32)


def calculate_controlled_iso_profit(
    controlled_iso_buyback_prices,
    controlled_iso_retail_prices,
    controlled_iso_dispatch,
    opponent_iso_buyback_prices,
    opponent_iso_retail_prices,
    pcs_steps,
    generation_cost,
    emergency_cost,
):
    """Settle the ISO currently being trained against its learning opponent."""

    # Positive market flow means the household bought from the market.
    # Negative market flow means the household exported genuine excess energy.
    market_energy = np.asarray(
        [step["market_energy_units"] for step in pcs_steps]
    )
    pcs_imports = np.maximum(market_energy, 0.0)
    pcs_exports = np.maximum(-market_energy, 0.0)

    # Convert the winning-price comparison into the learning ISO's market
    # share. Exact ties are split evenly rather than favoring either ISO.
    controlled_iso_retail_market_share = np.where(
        controlled_iso_retail_prices < opponent_iso_retail_prices,
        1.0,
        np.where(
            controlled_iso_retail_prices > opponent_iso_retail_prices,
            0.0,
            TIED_MARKET_SHARE,
        ),
    )
    controlled_iso_export_market_share = np.where(
        controlled_iso_buyback_prices > opponent_iso_buyback_prices,
        1.0,
        np.where(
            controlled_iso_buyback_prices < opponent_iso_buyback_prices,
            0.0,
            TIED_MARKET_SHARE,
        ),
    )
    export_has_acceptable_offer = (
        np.maximum(
            controlled_iso_buyback_prices,
            opponent_iso_buyback_prices,
        )
        >= MINIMUM_PCS_EXPORT_PRICE
    )

    controlled_iso_retail_energy = (
        pcs_imports * controlled_iso_retail_market_share
    )
    controlled_iso_export_energy = (
        pcs_exports
        * controlled_iso_export_market_share
        * export_has_acceptable_offer
    )

    # Dispatch is energy prepared ahead of time. Demand above dispatch must be
    # covered with expensive emergency energy. Dispatch that is not used still
    # incurs its generation cost, so no extra oversupply penalty is needed.
    controlled_iso_emergency_energy = np.maximum(
        controlled_iso_retail_energy - controlled_iso_dispatch,
        0.0,
    )

    # Profit has the four terms described in the student concept document.
    controlled_iso_retail_revenue = np.sum(
        controlled_iso_retail_energy * controlled_iso_retail_prices
    )
    controlled_iso_export_margin = np.sum(
        controlled_iso_export_energy
        * (EXTERNAL_RESALE_PRICE - controlled_iso_buyback_prices)
    )
    controlled_iso_dispatch_cost = (
        np.sum(controlled_iso_dispatch) * generation_cost
    )
    controlled_iso_emergency_energy_cost = (
        np.sum(controlled_iso_emergency_energy) * emergency_cost
    )
    controlled_iso_profit = (
        controlled_iso_retail_revenue
        + controlled_iso_export_margin
        - controlled_iso_dispatch_cost
        - controlled_iso_emergency_energy_cost
    )

    return float(controlled_iso_profit), {
        "profit": float(controlled_iso_profit),
        "retail_energy": float(np.sum(controlled_iso_retail_energy)),
        "export_energy": float(np.sum(controlled_iso_export_energy)),
        "dispatch_energy": float(np.sum(controlled_iso_dispatch)),
        "emergency_energy": float(
            np.sum(controlled_iso_emergency_energy)
        ),
        "retail_revenue": float(controlled_iso_retail_revenue),
        "export_margin": float(controlled_iso_export_margin),
        "dispatch_cost": float(controlled_iso_dispatch_cost),
        "emergency_dispatch_cost": float(
            controlled_iso_emergency_energy_cost
        ),
    }
