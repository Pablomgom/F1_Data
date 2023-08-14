from fastf1.ergast import Ergast

from src.general_analysis.ergast import get_position_changes
from src.general_analysis.qualy import qualy_results, gear_changes, fastest_by_point, overlying_laps
from src.general_analysis.race_plots import *
from src.general_analysis.race_videos import bar_race
from src.general_analysis.wdc import wdc_comparation
from src.onetime_analysis.onetime_analysis import get_circuitos, get_topspeed

if __name__ == '__main__':
    fastf1.Cache.enable_cache('../cache')

    session = fastf1.get_session(2022, 15, 'R')
    #session.load(telemetry=True, weather=False)

    # win_wdc(standings)

    #driver_laptimes(session)

    #qualy_diff('Alpine', 'Ferrari', 12)
    #race_diff('Williams', 'Alfa Romeo', 12)
    #position_changes(session)

    #overlying_laps(session, 'VER', 'LEC')

    #race_distance(session, 'PER', 'VER')

    #driver_lap_times(session, 'VER')

    #fastest_by_point(session, 'Red Bull Racing', 'Mercedes')

    #gear_changes(session, 'HAM')

    #tyre_strategies(session)

    #qualy_results(session)


    ergast = Ergast()
    races = ergast.get_race_results(season=2022, round=15, limit=1000)
    qualy = ergast.get_qualifying_results(season=1990, round=10,  limit=1000)
    #sprints = ergast.get_sprint_results(season=1955,  limit=1000)
    schedule = ergast.get_race_schedule(season=1990, round=10,  limit=1000)
    circuitos = ergast.get_circuits(season=1954, limit=1000)

    #qualy_results_ergast(qualy)


    #get_position_changes(races)

    #get_circuitos()


    #bar_race(races, sprints, schedule)

    #get_retirements()

    #wdc_comparation('Alonso', 2001)

    get_topspeed()