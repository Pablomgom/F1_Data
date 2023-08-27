import fastf1
from fastf1.ergast import Ergast
from fastf1.livetiming.data import LiveTimingData

from src.general_analysis.ergast import get_position_changes, qualy_results_ergast
from src.general_analysis.qualy import overlying_laps, qualy_results
from src.general_analysis.race_plots import position_changes, tyre_strategies, race_distance, race_diff, \
    driver_laptimes, driver_lap_times, driver_race_times_per_tyre
from src.general_analysis.race_videos import bar_race
from src.general_analysis.wdc import wdc_comparation
from src.onetime_analysis.onetime_analysis import get_topspeed, get_retirements_per_driver, compare_drivers_season, \
    get_pit_stops, wins_in_circuit, day_all_races, overtakes, get_topspeed_in_session, races_by_number

if __name__ == '__main__':

    fastf1.Cache.enable_cache('../cache')
    session = fastf1.get_session(2023, 13, 'R')
    session.load()

    #driver_race_times_per_tyre(session, 'ALB')

    #get_topspeed_in_session(session, 'Sector3Time')

    #win_wdc(standings)

    #driver_laptimes(session)

    #qualy_diff('Alpine', 'Ferrari', 12)

    #race_diff('Aston Martin', 'McLaren', 12)

    position_changes(session)

    #overlying_laps(session, 'RUS', 'ALB')

    #race_distance(session, 'BOT', 'HAM')

    #driver_lap_times(session, 'LEC')

    #fastest_by_point(session, 'Red Bull Racing', 'Mercedes')

    #gear_changes(session, 'HAM')

    #tyre_strategies(session)

    #qualy_results(session)

    '''
    ergast = Ergast()
    drivers = ergast.get_driver_info(season=1959, limit=1000)
    races = ergast.get_race_results(season=2003, round=13, limit=1000)
    qualy = ergast.get_qualifying_results(season=2008, round=6, limit=1000)
    sprints = ergast.get_sprint_results(season=1959,  limit=1000)
    schedule = ergast.get_race_schedule(season=1959, limit=1000)
    circuitos = ergast.get_circuits(season=2023, round=14, limit=1000)
    circuito = circuitos.circuitId.min()
    '''

    #races_by_number(40)

    #overtakes()

    #day_all_races()

    #wins_in_circuit(circuito)


    #get_pit_stops(2016)


    #compare_drivers_season('Schumacher', 'Magnussen', 2022)

    #qualy_results_ergast(qualy)


    #get_position_changes(races)

    #get_circuitos()

    #get_retirements_per_driver('Ricciardo', 2011, 2024)

    #bar_race(races, sprints, schedule)

    #get_retirements()

    #wdc_comparation('Hülkenberg', 2007)

    #get_topspeed()