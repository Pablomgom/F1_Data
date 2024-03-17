from src.analysis.data_analysis import *
from src.analysis.drivers import *
from src.analysis.history import *
from src.analysis.overtakes import *
from src.analysis.pit_stops import *
from src.analysis.teams import *
from src.analysis.upgrades import *
from src.elo.elo import elo_execution, best_season
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
        'lucky_drivers': lucky_drivers,
        'session_diff_last_year': session_diff_last_year,
        'qualy_diff_teammates': qualy_diff_teammates,
        'driver_race_times_per_tyre': driver_race_times_per_tyre,
        'get_fastest_data': get_fastest_data,
        'get_fastest_sectors': get_fastest_sectors,
        'win_wdc': win_wdc,
        'race_pace_top_10': race_pace_top_10,
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
        'long_runs_scatter': long_runs_scatter,
        'races_by_driver_dorsal': races_by_driver_dorsal,
        'plot_overtakes': plot_overtakes,
        'get_historical_race_days': get_historical_race_days,
        'wins_and_poles_circuit': wins_and_poles_circuit,
        'race_qualy_h2h': race_qualy_h2h,
        'qualy_results_ergast': qualy_results_ergast,
        'get_position_changes': get_position_changes,
        'get_circuitos': get_circuitos,
        'get_retirements_per_driver': get_retirements_per_driver,
        'team_wdc_history': team_wdc_history,
        'bar_season': bar_season,
        'get_retirements': get_retirements,
        'wdc_comparation': wdc_comparation,
        'get_driver_results_circuit': get_driver_results_circuit,
        'race_qualy_avg_metrics': race_qualy_avg_metrics,
        'qualy_margin': qualy_margin,
        'avg_driver_position': avg_driver_position,
        'full_compare_drivers_season': full_compare_drivers_season,
        'simulate_season_different_psystem': simulate_season_different_psystem,
        'simulate_qualy_championship': simulate_qualy_championship,
        'wins_per_year': wins_per_year,
        'compare_my_ergast_teammates': compare_my_ergast_teammates,
        'laps_completed': laps_completed,
        'winning_positions_per_circuit': winning_positions_per_circuit,
        'q3_appearances': q3_appearances,
        'results_from_grid_position': results_from_grid_position,
        'air_track_temps': air_track_temps,
        'comebacks_in_circuit': comebacks_in_circuit,
        'driver_grid_winning_positions': driver_grid_winning_positions,
        'laps_led': laps_led,
        'points_percentage_diff': points_percentage_diff,
        'fastest_pit_stop_by_team': fastest_pit_stop_by_team,
        'driver_fuel_corrected_laps': driver_fuel_corrected_laps,
        'fuel_correct_factor': fuel_correct_factor,
        'cluster_circuits': cluster_circuits,
        'race_diff_v2': race_diff_v2,
        'delta_reference_team': delta_reference_team,
        'qualy_diff': qualy_diff,
        'predict_race_pace': predict_race_pace,
        'race_pace_between_drivers': race_pace_between_drivers,
        'overtakes_by_race': overtakes_by_race,
        'victories_per_driver_team': victories_per_driver_team,
        'pitstops_per_year': pitstops_per_year,
        'pitstops_pirelli_era': pitstops_pirelli_era,
        'drs_efficiency': drs_efficiency,
        'percentage_qualy_ahead': percentage_qualy_ahead,
        'percentage_race_ahead': percentage_race_ahead,
        'difference_q1': difference_q1,
        'difference_P2': difference_P2,
        'difference_second_team': difference_second_team,
        'team_gap_to_next_or_fastest': team_gap_to_next_or_fastest,
        'driver_results_per_year': driver_results_per_year,
        'avg_qualy_pos_dif': avg_qualy_pos_dif,
        'avg_race_pos_dif': avg_race_pos_dif,
        'all_drivers_race_h2h': all_drivers_race_h2h,
        'times_lapped_per_team': times_lapped_per_team,
        'avg_qualy_pos_dif_per_years': avg_qualy_pos_dif_per_years,
        'best_season': best_season,
        'avg_position_season': avg_position_season,
        'dfns_per_year': dfns_per_year,
        'pole_position_evolution': pole_position_evolution,
        'points_per_year': points_per_year,
        'all_laps_driver_session': all_laps_driver_session,
        'race_simulation_test_day': race_simulation_test_day,
        'difference_Q3': difference_Q3,
        'help': help
    }

    return FUNCTION_MAP
