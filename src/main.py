from fastf1.ergast import Ergast
from src.general_analysis.ergast import *
from src.general_analysis.qualy import *
from src.general_analysis.race_plots import *
from src.general_analysis.race_videos import *
from src.general_analysis.wdc import *
from src.onetime_analysis.onetime_analysis import *

if __name__ == '__main__':

    fastf1.ergast.interface.BASE_URL = 'http://ergast.com/api/f1'
    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    pitstops(2023)

    # plot_circuit()

    cluster_circuits(2023, 15, 2022, 'Suzuka', 3)

    fastf1.Cache.enable_cache('../cache')
    session = fastf1.get_session(2022, 'Suzuka', 'R')
    # session.load()

    # lucky_drivers(2017,2018)

    # qualy_diff_last_year(14)

    # performance_vs_last_year('AlphaTauri', ['imola', 'catalunya', 'villeneuve', 'silverstone'])

    # race_pace_teammates('Ferrari', 15)

    # driver_race_times_per_tyre(session, 'LEC')

    # get_topspeed_in_session(session, 'Sector1Time')

    # win_wdc(2023)

    # driver_laptimes(session)

    # qualy_diff('Alpine', 'Ferrari', 15)

    # race_diff('Aston Martin', 'Mercedes', 2023)

    # position_changes(session)

    # overlying_laps(session, 'SAI', 'RUS')

    # race_distance(session, 'HAM', 'SAI')

    # driver_lap_times(session, 'SAI')

    # fastest_by_point_v2(session, 'Ferrari', 'Mercedes')

    # gear_changes(session, 'Speed')

    # tyre_strategies(session)

    # qualy_results(session)

    ergast = Ergast()
    drivers = ergast.get_driver_info(season=1959, limit=1000)
    races = ergast.get_race_results(season=2022, round=18, limit=1000)
    qualy = ergast.get_qualifying_results(season=2018, round=16, limit=1000)
    sprints = ergast.get_sprint_results(season=1962, limit=1000)
    schedule = ergast.get_race_schedule(season=1962, limit=1000)
    circuitos = ergast.get_circuits(season=2023, round=16, limit=1000)
    circuito = circuitos.circuitId.min()

    # get_fastest_punctuable_lap('marina_bay', start=2008, all_drivers=False)

    # races_by_number(0)

    # overtakes()

    # day_all_races()

    # wins_in_circuit(circuito, end=2023)

    # get_pit_stops(2018)

    # compare_drivers_season('Vettel', 'Alonso', 2012)

    # qualy_results_ergast(qualy)

    # get_position_changes(races)

    # get_circuitos()

    # get_retirements_per_driver('Schumacher', 1991, 2012)

    # team_wdc_history('McLaren')

    # bar_race(races, sprints, schedule)

    # get_retirements()

    # wdc_comparation('Daniel Ricciardo', 2011, 2024)

    # get_topspeed()

    # get_driver_results_circuit('sainz', 'suzuka', 2015)