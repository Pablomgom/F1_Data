import fastf1
from src.general_analysis.general_analysis import position_changes, overlying_laps

if __name__ == '__main__':

    fastf1.Cache.enable_cache('../cache')

    session = fastf1.get_session(2023, 12, 'Q')
    session.load(telemetry=True, weather=False)

    overlying_laps(session, 'PIA', 'NOR')

