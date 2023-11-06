from src.general_analysis.my_ergast_funcs import compare_my_ergast_teammates, get_driver_laps, \
    winning_positions_per_circuit, q3_appearances, results_from_pole, highest_qualy, last_result_grid_pos
from src.general_analysis.race_plots import long_runs_FP2_v2, teams_diff_session


def get_funcs():
    from src.utils.utils import find_similar_func
    from src.onetime_analysis.onetime_analysis import wins_per_year
    from src.elo.elo import elo_execution
    from src.general_analysis.ergast import qualy_results_ergast, get_position_changes
    from src.general_analysis.qualy import qualy_diff_last_year, team_performance_vs_qualy_last_year, qualy_diff, \
        telemetry_lap, overlying_laps, fastest_by_point, track_dominance, plot_circuit_with_data, qualy_results, \
        qualy_margin
    from src.general_analysis.race_plots import qualy_diff_teammates, race_pace_teammates, driver_race_times_per_tyre, \
        race_pace_top_10, race_diff, position_changes, race_distance, long_runs_FP2, tyre_strategies
    from src.general_analysis.race_videos import bar_race
    from src.general_analysis.wdc import win_wdc, team_wdc_history, wdc_comparation
    from src.onetime_analysis.onetime_analysis import plot_upgrades, dhl_pitstops, cluster_circuits, lucky_drivers, \
        get_fastest_data, get_fastest_punctuable_lap, races_by_driver_dorsal, plot_overtakes, get_historical_race_days, \
        wins_and_poles_circuit, get_pit_stops_ergast, compare_drivers_season, get_circuitos, get_retirements_per_driver, \
        get_retirements, get_topspeed, get_driver_results_circuit, race_qualy_avg_metrics, compare_amount_points, \
        compare_qualy_results, avg_driver_position, full_compare_drivers_season, simulate_season_different_psystem, \
        get_DNFs_team, simulate_qualy_championship
    from src.utils.utils import load_session

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
        'find_similar_func': find_similar_func,
        'wins_per_year': wins_per_year,
        'compare_my_ergast_teammates': compare_my_ergast_teammates,
        'get_driver_laps': get_driver_laps,
        'winning_positions_per_circuit': winning_positions_per_circuit,
        'q3_appearances': q3_appearances,
        'results_from_pole': results_from_pole,
        'highest_qualy': highest_qualy,
        'long_runs_FP2_v2': long_runs_FP2_v2,
        'last_result_grid_pos': last_result_grid_pos,
        'teams_diff_session': teams_diff_session,
        'help': help
    }

    return FUNCTION_MAP


team_colors_2023 = {
    'Mercedes': '#00d2be',
    'Ferrari': '#dc0000',
    'Red Bull': '#0600ef',
    'McLaren': '#ff8700',
    'Alpine': '#FF69B4',
    'Alpine F1 Team': '#FF69B4',
    'Aston Martin': '#006f62',
    'Alfa Romeo': '#900000',
    'AlphaTauri': '#2b4562',
    'Haas': '#ffffff',
    'Haas F1 Team': '#ffffff',
    'Williams': '#005aff'
}


team_colors_2009 = {
    'Brawn': '#E4FFD7',
    'Ferrari': '#dc0000',
    'Red Bull': '#0600ef',
    'McLaren': '#C0C0C0',
    'BMW Sauber': '#0000FF',
    'Renault': '#FFCC00',
    'Toyota': '#FF8686',
    'Toro Rosso': '#00004D',
    'Force India': '#FF6600',
    'Williams': '#000080'
}

team_colors_2008 = {
    'Honda': '#E4FFD7',
    'Ferrari': '#dc0000',
    'Red Bull': '#0600ef',
    'McLaren': '#C0C0C0',
    'BMW Sauber': '#0000FF',
    'Renault': '#FFCC00',
    'Toyota': '#FF8686',
    'Toro Rosso': '#00004D',
    'Force India': '#FF6600',
    'Williams': '#000080',
    'Super Aguri': '#FFFFFF'
}

team_colors = {
    2008: team_colors_2008,
    2009: team_colors_2009,
    2023: team_colors_2023
}



driver_colors_2023 = {
    'BOT': '#900000',
    'ZHO': '#500000',
    'DEV': '#1e3d61',
    'TSU': '#356cac',
    'RIC': '#2b4562',
    'LAW': '#2b4562',
    'GAS': '#0090ff',
    'OCO': '#70c2ff',
    'ALO': '#006f62',
    'STR': '#25a617',
    'DRU': '#2f9b90',
    'LEC': '#dc0000',
    'SAI': '#ff8181',
    'SHW': '#9c0000',
    'MAG': '#ffffff',
    'HUL': '#cacaca',
    'PIA': '#ff8700',
    'NOR': '#eeb370',
    'HAM': '#00d2be',
    'RUS': '#24ffff',
    'VER': '#0600ef',
    'PER': '#716de2',
    'ALB': '#005aff',
    'SAR': '#012564'}

driver_colors_historical = {
    'michael schumacher': '#dc0000',
    'sebastian vettel': '#239eda',
    'alain prost': '#FFA2A2',
    'ayrton senna': '#FFFFFF',
    'nigel mansell': '#E8E817',
    'jackie stewart': '#D1DAFF',
    'niki lauda': '#FF5D5D',


}

max_races = {
    1950 : 4,
    1951 : 4,
    1952 : 4,
    1953 : 4,
    1954 : 5,
    1955 : 5,
    1956 : 5,
    1957:  5,
    1958:  6,
    1959:  5,
    1960:  6,
    1961:  5,
    1962:  5,
    1963:  6
}

point_system_2009 = {
    1: 10,
    2: 8,
    3: 6,
    4: 5,
    5: 4,
    6: 3,
    7: 2,
    8: 1
}

point_system_2010 = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1
}

point_systems = {
    2009: point_system_2009,
    2010: point_system_2010
}

