import fastf1
import numpy as np
import pandas as pd

from fastf1.ergast import Ergast
from matplotlib import pyplot as plt, cm


from src.general_analysis.table import render_mpl_table


def get_retirements():
    races = []
    ergast = Ergast()

    for i in range(1950, 2023):
        races.append(ergast.get_race_results(season=i, limit=1000))
        races.append(ergast.get_sprint_results(season=i, limit=1000))
        print(i)

    init = pd.Series(dtype=object)

    for season in races:
        for race in season.content:
            init = pd.concat([init, race['status']], ignore_index=True)

    status = init.value_counts()
    status = status[~status.index.str.contains('+', regex=False)]
    status = status.drop('Finished')

    N = 10
    top_N = status.nlargest(N)
    top_N['Other'] = status.iloc[N:].sum()

    figsize = (8, 8)
    fig, ax = plt.subplots(figsize=figsize)

    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    total_value = top_N.sum()

    colormap = cm.get_cmap('tab20', len(top_N))
    colors = [colormap(i) for i in range(len(top_N))]

    top_N.plot.pie(ax=ax, autopct='%1.1f%%', labels=['' for _ in top_N.index], legend=False,
                   wedgeprops={'linewidth': 1, 'edgecolor': 'white'}, colors=colors)  # Set line color to black

    ax.legend(title="Cause of Retirement", loc="center left", labels=top_N.index, bbox_to_anchor=(0.8, 0.1))

    plt.title(f'Retirements in F1 History (Total: {total_value})', fontsize=16, color='white')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'../PNGs/Retirements in F1 History', dpi=400)
    plt.show()

    items_per_table = 25
    start = 11
    end = 0
    other_values = status.iloc[N:]
    chunks = np.array_split(other_values, np.ceil(len(other_values) / items_per_table))

    for i in range(len(chunks)):
        other_values = pd.DataFrame(chunks[i]).reset_index().rename(columns={0: 'Total times',
                                                                             'index': 'Retirement cause'})

        end = start + len(other_values)
        other_values['Ranking'] = range(start, end)
        other_values = other_values[['Ranking', 'Retirement cause', 'Total times']]

        render_mpl_table(other_values, diff_column='No', col_Widths=[0.1, 0.3, 0.15],
                         title=f'TOP {start} - {end} retirement cause')

        start = end
        plt.tight_layout()
        plt.savefig(f'../PNGs/TOP {start} - {end} retirement cause', dpi=400)
        plt.show()


def get_circuitos():

    ergast = Ergast()
    circuitos = []

    for i in range(1950, 2024):
        season_circuits = ergast.get_circuits(season=i, limit=10000)
        circuitos.append(season_circuits)

    circuit_names = []
    for season in circuitos:
        circuit_names.extend(season['country'])
    series = pd.Series(circuit_names)

    series[series == 'United States'] = 'USA'
    series = series.value_counts()
    N = 12
    top_N = series.nlargest(N)
    top_N['Other'] = series.iloc[N:].sum()

    figsize = (8, 8)
    fig, ax = plt.subplots(figsize=figsize)

    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    total_value = top_N.sum()

    colormap = cm.get_cmap('tab20', len(top_N))
    colors = [colormap(i) for i in range(len(top_N))]

    top_N.plot.pie(ax=ax, autopct='%1.1f%%', labels=['' for _ in top_N.index], legend=False,
                   wedgeprops={'linewidth': 1, 'edgecolor': 'white'}, colors=colors)  # Set line color to black

    ax.legend(title="Country", loc="center left", labels=top_N.index, bbox_to_anchor=(0.8, 0.1))

    plt.title(f'Times race in a country (Total races: {total_value})', fontsize=16, color='white')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'../PNGs/Countrys in F1 History', dpi=800)
    plt.show()

    items_per_table = 20
    start = 1
    end = 1
    chunks = np.array_split(series, np.ceil(len(series) / items_per_table))

    for i in range(len(chunks)):
        other_values = pd.DataFrame(chunks[i]).reset_index().rename(columns={0: 'Total times',
                                                                             'index': 'Country'})
        end = start + len(other_values)
        other_values['Ranking'] = range(start, end)
        other_values = other_values[['Ranking', 'Country', 'Total times']]

        render_mpl_table(other_values, diff_column='No', col_Widths=[0.1, 0.3, 0.15],
                         title=f'TOP {start} - {end - 1} countries')

        start = end
        plt.tight_layout()
        plt.savefig(f'../PNGs/TOP {start} - {end - 1} countries', dpi=600)
        plt.show()


def get_topspeed():

    top_speed_array = []

    for i in range(12):

        session = fastf1.get_session(2023, i+1, 'Q')
        session.load(telemetry=True, weather=False)
        circuit_speed = {}

        for lap in session.laps.pick_quicklaps().iterrows():
            top_speed = max(lap[1].telemetry['Speed'])
            driver = lap[1]['Driver']
            driver_speed = circuit_speed.get(driver)
            if driver_speed is not None:
                if top_speed > driver_speed:
                    circuit_speed[driver] = top_speed
            else:
                circuit_speed[driver] = top_speed

            print(circuit_speed)

        max_key = max(circuit_speed, key=circuit_speed.get)
        driver_top_speed = f'{max_key} - {circuit_speed[max_key]} - {session.event["EventName"]}'

        top_speed_array.append(driver_top_speed)

        print(top_speed_array)

    print(top_speed_array)
