# views/charts/__init__.py
# -----------------------------------------------------------
# Central export hub for all chart rendering modules
# -----------------------------------------------------------

# --- Generic charts
from .generic import (
    render_bar_chart,
    render_line_chart,
    render_grouped_bar_and_line,
    render_sankey,
)

# --- Ships (Maritime)
# --- Ships (Maritime)
from .ships import (
    render_ships_stock,
    render_ships_new,
    render_ships_investment_cost,
    render_ships_operational_cost,
    render_ships_fuel_demand,
    render_ships_fuel_cost,
    render_ships_emissions_and_cap,
    render_ships_ets_penalty,
    render_ships_interactive_controls,
)

# --- Water & FABLE (Food–Land)
from .water_food import (
    render_water_band,
    render_water_monthly_band,
    render_fable_interactive_controls,
    load_water_requirements,
)

# --- Energy
from .energy import (
    render_energy_interactive_controls,
)

# --- Biofuels
from .biofuels import (
    load_biofuels_data,
    get_biofuels_option_sets,
    render_biofuels_interactive_controls,
    render_biofuels_base_charts,
)