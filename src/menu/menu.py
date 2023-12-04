from src.analysis.data_analysis import *
from src.analysis.drivers import *
from src.analysis.history import *
from src.analysis.overtakes import *
from src.analysis.pit_stops import *
from src.analysis.teams import *
from src.analysis.upgrades import *
from src.elo.elo import elo_execution
from src.analysis.ergast import *
from src.sessions.qualy import *
from src.sessions.race import *
from src.analysis.bar_race import *
from src.sessions.session import *
from src.analysis.wdc import *
from src.utils.utils import *


def get_funcs():
    FUNCTION_MAP = {
        'load_session': load_session,
        'elo_execution': elo_execution,
        'plot_upgrades': plot_upgrades,
        'dhl_pitstops': dhl_pitstops,
        'cluster_circuits': cluster_circuits,
        'lucky_drivers': lucky_drivers,
        'session_diff_last_year': session_diff_last_year,
        'team_performance_vs_qualy_last_year': team_performance_vs_qualy_last_year,
        'qualy_diff_teammates': qualy_diff_teammates,
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
        'long_runs_driver': long_runs_driver,
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
        'compare_drivers_season': compare_drivers_season,
        'qualy_results_ergast': qualy_results_ergast,
        'get_position_changes': get_position_changes,
        'get_circuitos': get_circuitos,
        'get_retirements_per_driver': get_retirements_per_driver,
        'team_wdc_history': team_wdc_history,
        'bar_season': bar_season,
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
        'find_similar_func': find_similar_func,
        'wins_per_year': wins_per_year,
        'compare_my_ergast_teammates': compare_my_ergast_teammates,
        'get_driver_laps': get_driver_laps,
        'winning_positions_per_circuit': winning_positions_per_circuit,
        'q3_appearances': q3_appearances,
        'results_from_pole': results_from_pole,
        'highest_qualy': highest_qualy,
        'long_runs_scatter': long_runs_scatter,
        'last_result_grid_pos': last_result_grid_pos,
        'teams_diff_session': teams_diff_session,
        'air_track_temps': air_track_temps,
        'comebacks_in_circuit': comebacks_in_circuit,
        'session_results': session_results,
        'driver_grid_positions': driver_grid_positions,
        'laps_led': laps_led,
        'points_percentage_diff': points_percentage_diff,
        'fastest_pit_stop_by_team': fastest_pit_stop_by_team,
        'predict_race_pace': predict_race_pace,
        'help': help
    }

    return FUNCTION_MAP
