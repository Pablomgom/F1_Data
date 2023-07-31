import fastf1
from src.general_analysis.general_analysis import position_changes, overlying_laps, driver_lap_times, gear_changes, \
    tyre_strategies

if __name__ == '__main__':

    fastf1.Cache.enable_cache('../cache')

    session = fastf1.get_session(2023, 12, 'R')
    session.load(telemetry=True, weather=False)

    # position_changes(session)

    # overlying_laps(session, 'VER', 'PER')

    # driver_lap_times(session, 'NOR')

    # gear_changes(session, 'VER')

    tyre_strategies(session)