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
    #pitstops(2023, 14)

    #plot_circuit()

    fastf1.Cache.enable_cache('../cache')
    session = fastf1.get_session(2023, 14, 'R')
    session.load()

    #driver_race_times_per_tyre(session, 'ALB')

    #get_topspeed_in_session(session, 'Speed')

    #win_wdc(standings)

    #driver_laptimes(session)

    #qualy_diff('Alpine', 'Ferrari', 12)

    #race_diff('Ferrari', 'Aston Martin', 14)

    #position_changes(session)

    #overlying_laps(session, 'PER', 'VER')

    race_distance(session, 'BOT', 'LAW')

    #driver_lap_times(session, 'VER')

    #fastest_by_point_v2(session, 'Ferrari', 'Red Bull Racing')

    #gear_changes(session, 'HAM')

    #tyre_strategies(session)

    #qualy_results(session)


    ergast = Ergast()
    drivers = ergast.get_driver_info(season=1959, limit=1000)
    #races = ergast.get_race_results(season=1961, limit=1000)
    qualy = ergast.get_qualifying_results(season=2008, round=6, limit=1000)
    #sprints = ergast.get_sprint_results(season=1961,  limit=1000)
    #schedule = ergast.get_race_schedule(season=1961, limit=1000)
    circuitos = ergast.get_circuits(season=2023, round=14, limit=1000)
    circuito = circuitos.circuitId.min()


    #races_by_number(13)

    #overtakes()

    #day_all_races()

    #wins_in_circuit(circuito)


    #get_pit_stops(2018)


    #compare_drivers_season('Schumacher', 'Magnussen', 2022)

    #qualy_results_ergast(qualy)


    #get_position_changes(races)

    #get_circuitos()

    #get_retirements_per_driver('Schumacher', 1991, 2012)

    #team_wdc_history('McLaren')

    #bar_race(races, sprints, schedule)

    #get_retirements()

    #wdc_comparation('Schumacher', 1991, 2013)

    #get_topspeed()