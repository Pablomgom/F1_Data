from fastf1.ergast import Ergast
from src.general_analysis.general_analysis import *
from src.general_analysis.race_plots import *

if __name__ == '__main__':
    fastf1.Cache.enable_cache('../cache')

    session = fastf1.get_session(2022, 1, 'R')
    # session.load(telemetry=True, weather=False)

    # position_changes(session)

    # overlying_laps(session, 'VER', 'PER')

    # driver_lap_times(session, 'NOR')

    # gear_changes(session, 'VER')

    # tyre_strategies(session)

    # qualy_results(session)

    ergast = Ergast()
    races = ergast.get_race_results(season=1952, limit=1000)
    sprints = ergast.get_sprint_results(season=1952, limit=1000)
    schedule = ergast.get_race_schedule(season=1952, limit=1000)

    bar_race(races, sprints, schedule)
