"""The student-facing competitive extension of EnergyNet."""

from energy_net.gym_envs.alternating_env import (
    AlternatingISOEnv,
    get_pred_window,
)
from energy_net.grid_entities.management.price_curve import (
    ActionBasedPriceStrategy,
)
from for_students.internal.market_support import (
    DEFAULT_EMERGENCY_COST,
    DEFAULT_GENERATION_COST,
    RecordingPCSEnv,
    STEPS_PER_DAY,
    build_combined_market_action_seen_by_pcs,
    calculate_controlled_iso_profit,
)


class CompetitiveAlternatingISOEnv(AlternatingISOEnv):
    """Train one ISO against another ISO's current learning policy."""

    def __init__(
        self,
        *args,
        opponent_iso_model=None,
        generation_cost=DEFAULT_GENERATION_COST,
        emergency_cost=DEFAULT_EMERGENCY_COST,
        **kwargs,
    ):
        kwargs["use_asymmetric_pricing"] = True
        super().__init__(*args, **kwargs)
        if not isinstance(self.pcs_env, RecordingPCSEnv):
            raise TypeError("pcs_env must be a RecordingPCSEnv")
        self.opponent_iso_model = opponent_iso_model
        self.generation_cost = generation_cost
        self.emergency_cost = emergency_cost

    def step(self, action):
        if self.opponent_iso_model is None:
            raise RuntimeError(
                "opponent_iso_model must be assigned before training"
            )

        iso_observation = get_pred_window(
            self,
            self._next_start_idx,
            self.T,
        )
        controlled_action, _, controlled_dispatch_action = (
            self._split_action(action)
        )
        opponent_action, _ = self.opponent_iso_model.predict(
            iso_observation,
            deterministic=True,
        )
        opponent_action, _, opponent_dispatch_action = self._split_action(
            opponent_action
        )

        controlled_prices = ActionBasedPriceStrategy(
            controlled_action,
            use_asymmetric_pricing=True,
            steps_per_day=self.T,
        )
        opponent_prices = ActionBasedPriceStrategy(
            opponent_action,
            use_asymmetric_pricing=True,
            steps_per_day=self.T,
        )
        controlled_export_prices = controlled_prices.calculate_buy_price()
        controlled_retail_prices = controlled_prices.calculate_sell_price()
        controlled_dispatch = (
            controlled_dispatch_action * self.dispatch_scale
        )
        opponent_export_prices = opponent_prices.calculate_buy_price()
        opponent_retail_prices = opponent_prices.calculate_sell_price()
        opponent_dispatch = opponent_dispatch_action * self.dispatch_scale

        market_action = build_combined_market_action_seen_by_pcs(
            controlled_action,
            opponent_action,
            self.T,
        )
        next_observation, _, terminated, truncated, information = (
            super().step(market_action)
        )

        controlled_profit, controlled_details = (
            calculate_controlled_iso_profit(
                controlled_export_prices,
                controlled_retail_prices,
                controlled_dispatch,
                opponent_export_prices,
                opponent_retail_prices,
                self.pcs_env.day_steps,
                self.generation_cost,
                self.emergency_cost,
            )
        )
        _, opponent_details = calculate_controlled_iso_profit(
            opponent_export_prices,
            opponent_retail_prices,
            opponent_dispatch,
            controlled_export_prices,
            controlled_retail_prices,
            self.pcs_env.day_steps,
            self.generation_cost,
            self.emergency_cost,
        )
        information.update(
            {
                f"controlled_iso_{name}": value
                for name, value in controlled_details.items()
            }
        )
        information.update(
            {
                f"opponent_iso_{name}": value
                for name, value in opponent_details.items()
            }
        )
        self._last_reward = controlled_profit
        self._last_step_info.update(information)
        return (
            next_observation,
            controlled_profit,
            terminated,
            truncated,
            information,
        )
