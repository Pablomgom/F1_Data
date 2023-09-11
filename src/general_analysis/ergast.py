import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from timple.timedelta import strftimedelta

from src.general_analysis.table import render_mpl_table
from src.variables.variables import team_colors


def qualy_results_ergast(qualy):
    n_drivers = len(qualy.content[0]['Q1'])

    n_drivers_session = int((n_drivers - 10) / 2)
    qualy_times = []
    pole_lap = []
    colors = []
    for i in range(n_drivers):
        if i < 10:
            qualy_times.append(qualy.content[0]['Q3'][i])
        elif i >= 10 and i < (10 + n_drivers_session):
            qualy_times.append(qualy.content[0]['Q2'][i])
        elif i >= (10 + n_drivers_session):
            qualy_times.append(qualy.content[0]['Q1'][i])

        pole_lap.append(qualy.content[0]['Q3'][0])
        colors.append(team_colors[qualy.content[0]['constructorName'][i]])

    delta_time = pd.Series(qualy_times) - pd.Series(pole_lap)
    delta_time = delta_time.fillna(pd.Timedelta(days=0))

    fig, ax = plt.subplots()
    ax.barh(qualy.content[0]['driverCode'], delta_time, color=colors, edgecolor='grey')

    ax.set_yticks(qualy.content[0]['driverCode'])
    ax.set_yticklabels(qualy.content[0]['driverCode'])

    # show fastest at the top
    ax.invert_yaxis()
    ax.axhline(9.5, color='black', linestyle='-', linewidth=1)
    ax.text(max(delta_time).total_seconds() * 1e9, 11, 'Q2', va='bottom',
            ha='right', fontsize=14)
    ax.text(max(delta_time).total_seconds() * 1e9, 1, 'Q3', va='bottom',
            ha='right', fontsize=14)
    ax.text(max(delta_time).total_seconds() * 1e9, 16, 'Q1', va='bottom',
            ha='right', fontsize=14)
    # Horizontal bar at index 16
    ax.axhline(n_drivers_session + 9.5, color='black', linestyle='-', linewidth=1)

    # draw vertical lines behind the bars
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)

    lap_time_string = strftimedelta(pole_lap[0], '%m:%s.%ms')

    plt.suptitle(f"{qualy.description['raceName'][0]} {qualy.description['season'].min()} Qualifying\n"
                 f"Fastest Lap: {lap_time_string} ({qualy.content[0]['driverCode'][0]})")

    def custom_formatter(x, pos):
        return round(x * 100000, 1)

    ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))

    plt.xlabel("Seconds")
    plt.ylabel("Driver")

    plt.savefig(f"../PNGs/{qualy.description['raceName'][0]} {qualy.description['season'].min()} Qualifying", dpi=400)
    plt.show()


def get_position_changes(race):
    finish = race.content[0][['familyName', 'givenName', 'grid', 'status', 'constructorName']]
    finish['Driver'] = finish['givenName'] + ' ' + finish['familyName']
    finish['Finish'] = range(1, finish.shape[0] + 1)
    finish.loc[(finish['grid'] == 5) & (finish['Driver'] == 'Guanyu Zhou'), 'grid'] = 15
    finish['grid'].replace(0, 20, inplace=True)
    finish.loc[finish['status'].isin(['Did not qualify', 'Did not prequalify']), 'grid'] = finish['Finish']
    finish['Grid change'] = finish['grid'] - finish['Finish']
    #finish['grid'].replace(20, 'Pit Lane', inplace=True)
    finish['Team'] = finish['constructorName']


    race_diff_times = []
    for race_content in race.content:
        for i in range(len(race_content['totalRaceTime'])):
            if i == 0:
                race_time = race_content['totalRaceTime'][i]
                hours = race_time.seconds // 3600
                minutes = ((race_time.seconds // 60) % 60)
                seconds = race_time.seconds % 60
                milliseconds = race_time.microseconds // 1000
                race_time = f"{hours}:{str(minutes).ljust(2, '0')}:{str(seconds).ljust(3, '0')}" \
                            f".{str(milliseconds).ljust(3, '0')}"
                race_diff_times.append(race_time)
            else:
                race_time = race_content['totalRaceTime'][i]
                if pd.isna(race_time):
                    race_diff_times.append(None)
                else:
                    minutes = (race_time.seconds // 60) % 60
                    seconds = race_time.seconds % 60
                    milliseconds = race_time.microseconds // 1000
                    race_time = f"+{str(minutes).zfill(1)}:{str(seconds).zfill(2)}.{(str(milliseconds).zfill(2)).rjust(3, '0')}"
                    race_diff_times.append(race_time)

    finish['status'] = pd.Series(race_diff_times).combine_first(finish['status'])


    def modify_grid_change(value):
        if value > 0:
            return '+' + str(value)
        elif value == 0:
            return 'Equal'
        else:
            return str(value)

    finish['Grid change'] = finish['Grid change'].apply(modify_grid_change)
    finish.rename(columns={'status': 'Status', 'grid': 'Starting position'}, inplace=True)

    finish = finish[['Finish', 'Driver', 'Team', 'Starting position', 'Status', 'Grid change']]

    fig, axs = plt.subplots(1, 1, figsize=(14, 21))
    render_mpl_table(finish)
    plt.title(f"Race Results - {race.description['raceName'].min()} - {race.description['season'].min()}", fontsize=20)
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"../PNGs/Race Results - {race.description['raceName'].min()} - {race.description['season'].min()}",
                bbox_inches='tight', dpi=400)
    plt.show()
