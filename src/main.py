import sys

import fastf1
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt

from src.elo.elo import elo_execution
from src.general_analysis.ergast import *
from src.general_analysis.qualy import *
from src.general_analysis.race_plots import *
from src.general_analysis.race_videos import bar_race
from src.general_analysis.wdc import *
from src.onetime_analysis.onetime_analysis import *
from src.utils.utils import parse_args, load_session

fastf1.plotting.setup_mpl(misc_mpl_mods=False)
fastf1.ergast.interface.BASE_URL = 'http://ergast.com/api/f1'
plt.rcParams["font.family"] = "Fira Sans"
fastf1.Cache.enable_cache('../cache')

FUNCTION_MAP = {
    'load_session': load_session,
    'elo_execution': elo_execution,
    'plot_upgrades': plot_upgrades,
    'dhl_pitstops': dhl_pitstops,
    'cluster_circuits': cluster_circuits,
    'lucky_drivers': lucky_drivers,
    'qualy_diff_last_year': qualy_diff_last_year,
    'team_performance_vs_qualy_last_year': team_performance_vs_qualy_last_year,
    'qualy_diff_teammates': qualy_diff_teammates,
    'race_pace_teammates': race_pace_teammates,
    'driver_race_times_per_tyre': driver_race_times_per_tyre,
    'get_fastest_data': get_fastest_data,
    'win_wdc': win_wdc,
    'race_pace_top_10': race_pace_top_10,
    'qualy_diff': qualy_diff,
    'race_diff': race_diff,
    'position_changes': position_changes,
    'telemetry_lap': telemetry_lap,
    'overlying_laps': overlying_laps,
    'race_distance': race_distance,
    'long_runs_FP2': long_runs_FP2,
    'fastest_by_point': fastest_by_point,
    'track_dominance': track_dominance,
    'plot_circuit_with_data': plot_circuit_with_data,
    'tyre_strategies': tyre_strategies,
    'qualy_results': qualy_results,
    'get_fastest_punctuable_lap': get_fastest_punctuable_lap,
    'races_by_driver_dorsal': races_by_driver_dorsal,
    'plot_overtakes': plot_overtakes,
    'get_historical_race_days': get_historical_race_days,
    'wins_and_poles_circuit': wins_and_poles_circuit,
    'get_pit_stops_ergast': get_pit_stops_ergast,
    'compare_drivers_season': compare_drivers_season,
    'qualy_results_ergast': qualy_results_ergast,
    'get_position_changes': get_position_changes,
    'get_circuitos': get_circuitos,
    'get_retirements_per_driver': get_retirements_per_driver,
    'team_wdc_history': team_wdc_history,
    'bar_race': bar_race,
    'get_retirements': get_retirements,
    'wdc_comparation': wdc_comparation,
    'get_topspeed': get_topspeed,
    'get_driver_results_circuit': get_driver_results_circuit,
    'race_qualy_avg_metrics': race_qualy_avg_metrics,
    'qualy_margin': qualy_margin,
    'compare_amount_points': compare_amount_points,
    'compare_qualy_results': compare_qualy_results,
    'avg_driver_position': avg_driver_position,
    'full_compare_drivers_season': full_compare_drivers_season,
    'simulate_season_different_psystem': simulate_season_different_psystem,
    'get_DNFs_team': get_DNFs_team,
    'simulate_qualy_championship': simulate_qualy_championship,
    'help': help

}

session = None

if __name__ == '__main__':
    while True:
        func_name = input("Enter the function name (or 'exit' to quit): ")
        if func_name.lower() == 'exit':
            print("Exiting...")
            sys.exit()
        args_input = input(f"Enter arguments for {func_name} separated by commas: ")
        args, kwargs = parse_args(args_input, FUNCTION_MAP, session)
        try:
            result = FUNCTION_MAP[func_name](*args, **kwargs)
            if func_name.lower() == 'load_session':
                session = result
            if result is not None:
                print(f"Result: {result}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

