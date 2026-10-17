"""Print experiment diagnostics without running another simulation."""


# Width of the longest bar in the terminal comparison charts.
TERMINAL_BAR_WIDTH = 32

# Number of half-hour opportunities for a PCS shortage during each day.
HALF_HOURS_PER_DAY = 48


def _print_metric_table(title, rows):
    print(f"\n{title}")
    print("-" * len(title))
    for label, total, daily_average in rows:
        print(f"{label:<28} {total:>12.2f}  {daily_average:>10.3f}/day")


def _print_signed_bar_chart(title, values):
    print(f"\n{title}")
    print("-" * len(title))
    largest_magnitude = max(abs(value) for value in values.values()) or 1.0

    for label, value in values.items():
        bar_length = round(
            abs(value) / largest_magnitude * TERMINAL_BAR_WIDTH
        )
        direction = "+" if value >= 0 else "-"
        print(
            f"{label:<18} {direction}{'#' * bar_length:<{TERMINAL_BAR_WIDTH}}"
            f" {value:>10.2f}"
        )


def print_experiment_report(results):
    """Print financial, energy, and reliability results from one evaluation."""

    test_days = results["test_days"]
    main_iso = results["main_iso"]
    competitor_iso = results["competitor_iso"]
    pcs = results["pcs"]

    print("\nCOMPETITIVE ENERGY MARKET TEST RESULTS")
    print("=" * 38)
    if "training_budget" in results:
        training_budget = results["training_budget"]
        print(
            "Training budget used: "
            f"{training_budget['cpu_seconds']:.1f} CPU seconds over "
            f"{training_budget['wall_seconds']:.1f} wall seconds"
        )
    print(f"Held-out test days: {test_days}")

    _print_metric_table(
        "Financial outcomes",
        [
            (
                "Main ISO profit",
                main_iso["profit"],
                main_iso["profit"] / test_days,
            ),
            (
                "Competitor ISO profit",
                competitor_iso["profit"],
                competitor_iso["profit"] / test_days,
            ),
            (
                "PCS transaction money",
                pcs["money"],
                pcs["money"] / test_days,
            ),
            (
                "PCS net reward",
                pcs["net_reward"],
                pcs["net_reward"] / test_days,
            ),
        ],
    )
    _print_signed_bar_chart(
        "Average daily financial comparison",
        {
            "Main ISO": main_iso["profit"] / test_days,
            "Competitor ISO": competitor_iso["profit"] / test_days,
            "PCS money": pcs["money"] / test_days,
            "PCS net reward": pcs["net_reward"] / test_days,
        },
    )
    _print_metric_table(
        "ISO profit breakdown",
        [
            (
                "Main ISO retail revenue",
                main_iso["retail_revenue"],
                main_iso["retail_revenue"] / test_days,
            ),
            (
                "Main ISO export margin",
                main_iso["export_margin"],
                main_iso["export_margin"] / test_days,
            ),
            (
                "Main ISO dispatch cost",
                -main_iso["dispatch_cost"],
                -main_iso["dispatch_cost"] / test_days,
            ),
            (
                "Main ISO emergency cost",
                -main_iso["emergency_energy_cost"],
                -main_iso["emergency_energy_cost"] / test_days,
            ),
            (
                "Competitor retail revenue",
                competitor_iso["retail_revenue"],
                competitor_iso["retail_revenue"] / test_days,
            ),
            (
                "Competitor export margin",
                competitor_iso["export_margin"],
                competitor_iso["export_margin"] / test_days,
            ),
            (
                "Competitor dispatch cost",
                -competitor_iso["dispatch_cost"],
                -competitor_iso["dispatch_cost"] / test_days,
            ),
            (
                "Competitor emergency cost",
                -competitor_iso["emergency_energy_cost"],
                -competitor_iso["emergency_energy_cost"] / test_days,
            ),
        ],
    )

    _print_metric_table(
        "Energy and cost diagnostics",
        [
            (
                "PCS consumption",
                pcs["consumption"],
                pcs["consumption"] / test_days,
            ),
            (
                "Main ISO retail energy",
                main_iso["retail_energy"],
                main_iso["retail_energy"] / test_days,
            ),
            (
                "Competitor retail energy",
                competitor_iso["retail_energy"],
                competitor_iso["retail_energy"] / test_days,
            ),
            (
                "Main ISO dispatch energy",
                main_iso["dispatch_energy"],
                main_iso["dispatch_energy"] / test_days,
            ),
            (
                "Competitor dispatch energy",
                competitor_iso["dispatch_energy"],
                competitor_iso["dispatch_energy"] / test_days,
            ),
            (
                "Main ISO emergency energy",
                main_iso["emergency_energy"],
                main_iso["emergency_energy"] / test_days,
            ),
            (
                "Competitor emergency energy",
                competitor_iso["emergency_energy"],
                competitor_iso["emergency_energy"] / test_days,
            ),
            (
                "Main ISO dispatch cost",
                main_iso["dispatch_cost"],
                main_iso["dispatch_cost"] / test_days,
            ),
            (
                "Competitor dispatch cost",
                competitor_iso["dispatch_cost"],
                competitor_iso["dispatch_cost"] / test_days,
            ),
        ],
    )

    total_half_hours = test_days * HALF_HOURS_PER_DAY
    grid_reliance_percentage = (
        pcs["grid_reliance_periods"] / total_half_hours * 100
    )
    print("\nHousehold grid reliance")
    print("-----------------------")
    print(
        "Battery could not cover consumption: "
        f"{pcs['grid_reliance_periods']} of {total_half_hours} half-hours"
    )
    print(f"Grid-reliance rate: {grid_reliance_percentage:.2f}%")
