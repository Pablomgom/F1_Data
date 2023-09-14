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
    # pitstops(2023, 14)

    # plot_circuit()

    fastf1.Cache.enable_cache('../cache')
    session = fastf1.get_session(2022, 'Singapur', 'FP2')
    #session.load()

    # lucky_drivers()

    # qualy_diff_last_year(14)

    # performance_vs_last_year('AlphaTauri', ['imola', 'catalunya', 'villeneuve', 'silverstone'])

    # race_pace_teammates('Ferrari')

    # driver_race_times_per_tyre(session, 'VER')

    # get_topspeed_in_session(session, 'Speed', fastest_lap=True)

    # win_wdc(standings)

    # driver_laptimes(session)

    # qualy_diff('Alpine', 'Ferrari', 12)

    # race_diff('Red Bull Racing', 'Aston Martin', 2023)

    # position_changes(session)

    # overlying_laps(session, 'HAM', 'BOT')

    # race_distance(session, 'HAM', 'VER')

    # driver_lap_times(session, 'VER')

    # fastest_by_point_v2(session, 'Ferrari', 'Red Bull Racing')

    # gear_changes(session, 'Speed')

    # tyre_strategies(session)

    # qualy_results(session)

    ergast = Ergast()
    drivers = ergast.get_driver_info(season=1959, limit=1000)
    races = ergast.get_race_results(season=2010, round=15, limit=1000)
    qualy = ergast.get_qualifying_results(season=2008, round=6, limit=1000)
    sprints = ergast.get_sprint_results(season=1962, limit=1000)
    schedule = ergast.get_race_schedule(season=1962, limit=1000)
    circuitos = ergast.get_circuits(season=1951, round=6, limit=1000)
    circuito = circuitos.circuitId.min()

    # get_fastest_punctuable_lap('marina_bay', start=2008, all_drivers=False)

    # races_by_number(0)

    # overtakes()

    # day_all_races()

    # wins_in_circuit(circuito, end=1977)

    # get_pit_stops(2018)

    # compare_drivers_season('Hamilton', 'Rosberg', 2016)

    # qualy_results_ergast(qualy)

    # get_position_changes(races)

    # get_circuitos()

    # get_retirements_per_driver('Schumacher', 1991, 2012)

    # team_wdc_history('McLaren')

    # bar_race(races, sprints, schedule)

    # get_retirements()

    # wdc_comparation('Mario Andretti', 1950, 1983)

    # get_topspeed()

    get_driver_results_circuit('hamilton', 'marina_bay', 2008)
