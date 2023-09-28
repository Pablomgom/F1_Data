from fastf1.ergast import Ergast
from matplotlib.font_manager import FontProperties

from src.general_analysis.ergast import *
from src.general_analysis.qualy import *
from src.general_analysis.race_plots import *
from src.general_analysis.race_videos import *
from src.general_analysis.wdc import *
from src.variables.variables import *
from src.onetime_analysis.onetime_analysis import *

if __name__ == '__main__':

    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    fastf1.ergast.interface.BASE_URL = 'http://ergast.com/api/f1'
    plt.rcParams["font.family"] = "Fira Sans"

    # plot_upgrades('Circuit Specific')

    # dhl_pitstops(2023, points=True)

    # plot_circuit()

    # cluster_circuits(2023, 16, 2021, 'Qatar',  clusters=3)

    fastf1.Cache.enable_cache('../cache')
    session = fastf1.get_session(2023, 'Silverstone', 'R')
    # session.load()

    # lucky_drivers(1950,2024)

    # qualy_diff_last_year(16, 'suzuka')

    # team_performance_vs_qualy_last_year('McLaren', ['Imola', 'Spanish', 'Canadian', 'British', 'Singapore'])

    # qualy_diff_teammates('Aston Martin', 16)

    race_pace_teammates('Aston Martin', 16)

    # driver_race_times_per_tyre(session, 'LEC')

    # get_fastest_data(session, 'Sector3Time')

    # win_wdc(2023)

    # race_pace_top_10(session)

    # qualy_diff('Aston Martin', 'McLaren', 16)

    # race_diff('Aston Martin', 'Mercedes', 2023)

    # position_changes(session)

    # overlying_laps(session, 'VER', 'NOR')

    # race_distance(session, 'LEC', 'SAI')

    # long_runs_FP2(session, 'RUS')

    # fastest_by_point(session, 'PIA', 'NOR', scope='D')

    # track_dominance(session, 'Red Bull Racing', 'Ferrari')

    # plot_circuit_with_data(session, 'Speed')

    # tyre_strategies(session)

    # qualy_results(session)

    ergast = Ergast()
    drivers = ergast.get_driver_info(season=1959, limit=1000)
    races = ergast.get_race_results(season=1963, limit=1000)
    qualy = ergast.get_qualifying_results(season=1963, round=16, limit=1000)
    sprints = ergast.get_sprint_results(season=1963, limit=1000)
    schedule = ergast.get_race_schedule(season=1963, limit=1000)
    circuitos = ergast.get_circuits(season=2023, round=16, limit=1000)
    circuito = circuitos.circuitId.min()

    # get_fastest_punctuable_lap('marina_bay', start=2008, all_drivers=False)

    # races_by_driver_dorsal(0)

    # plot_overtakes()

    # get_historical_race_days()

    # wins_and_poles_circuit(circuito, end=2023)

    # get_pit_stops_ergast(2023)

    # compare_drivers_season('Hamilton', 'Russell', 2023, DNFs=True)

    # qualy_results_ergast(qualy)

    # get_position_changes(races)

    # get_circuitos()

    # get_retirements_per_driver('Schumacher', 1991, 2012)

    # team_wdc_history('McLaren')

    # bar_race(races, sprints, schedule)

    # get_retirements()

    # wdc_comparation('Daniel Ricciardo', 2011, 2024)

    # get_topspeed(16)

    # get_driver_results_circuit('max_verstappen', 'suzuka', 2015)

    # race_qualy_avg_metrics(2023, session='R', predict=True)

    # qualy_margin('suzuka', start=1950, end=2024)

    # compare_amount_points('mclaren', -1, end=2010)

    # compare_qualy_results('alphatauri', 19, end=2010)

    # avg_driver_position('perez', 'red_bull', 2023, session='Q')

    full_compare_drivers_season(2012, 'vettel', 'alonso', 'red_bull',
                                d1_team='red_bull', d2_team='ferrari')


