import pickle
import statistics

import fastf1
import numpy as np
import pandas as pd
from adjustText import adjust_text
from fastf1 import plotting
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt, ticker, cm
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as path_effects

from src.exceptions import qualy_by_year
from src.exceptions.custom_exceptions import QualyException
from src.utils import utils
from src.plots.plots import round_bars, annotate_bars
from src.utils.utils import darken_color, plot_turns, rotate, remove_close_rows
import seaborn as sns
from src.variables.team_colors import team_colors_2023


def long_runs_driver(session, driver, threshold=1.07):
    """
         Plots the long runs in FP2 for a given driver

         Parameters:
         race (Session): Session to analyze
         driver (str): Driver

    """

    plotting.setup_mpl(misc_mpl_mods=False)

    driver_laps = session.laps.pick_driver(driver)
    max_stint = driver_laps['Stint'].value_counts().index[0]
    driver_laps_filter = driver_laps[driver_laps['Stint'] == max_stint].pick_quicklaps(threshold)

    driver_laps_filter = driver_laps_filter[driver_laps_filter['LapTime'].notna()]
    driver_laps_filter['LapNumber'] = driver_laps_filter['LapNumber'] - driver_laps_filter['LapNumber'].min() + 1
    driver_laps_filter['LapTime_seconds'] = driver_laps_filter['LapTime'].dt.total_seconds()
    driver_laps_filter['MovingAverage'] = driver_laps_filter['LapTime_seconds'].rolling(window=3, min_periods=1).mean()
    driver_laps_filter = driver_laps_filter.reset_index()

    fig, ax = plt.subplots(figsize=(8, 8))

    plotting.COMPOUND_COLORS['TEST_UNKNOWN'] = 'grey'

    sns.scatterplot(data=driver_laps_filter,
                    x="LapNumber",
                    y="LapTime_seconds",
                    ax=ax,
                    hue="Compound",
                    palette=plotting.COMPOUND_COLORS,
                    s=125,
                    linewidth=0,
                    legend='auto')

    ax.set_xlabel("Lap Number", font='Fira Sans', fontsize=16)
    ax.set_ylabel("Lap Time", font='Fira Sans', fontsize=16)
    plt.grid(color='w', which='major', axis='both', linestyle='--')

    def format_func(value, tick_number):
        minutes = np.floor(value / 60)
        seconds = int(value % 60)
        milliseconds = int((value * 1000) % 1000)  # multiplying by 1000 to get milliseconds
        return f"{int(minutes)}:{seconds:02}.{milliseconds:03}"

    ax.plot(driver_laps_filter['LapNumber'], driver_laps_filter['MovingAverage'], label="Avg of the last 3 laps",
            color=(0, 191 / 255, 255 / 255))

    # 1. Plot the MA points
    sns.scatterplot(data=driver_laps_filter,
                    x="LapNumber",
                    y="MovingAverage",
                    ax=ax,
                    color=(0, 191 / 255, 255 / 255),
                    s=125,
                    linewidth=0,
                    legend=False,
                    zorder=-5)

    bbox_props = dict(boxstyle="square,pad=0.4", fc="white", ec="none", lw=0)
    annotations = []
    for lap, time in zip(driver_laps_filter['LapNumber'], driver_laps_filter['LapTime_seconds']):
        if not np.isnan(time):
            text = ax.annotate(f"{format_func(time, None)}",
                               (lap, time),
                               textcoords="offset points",
                               xytext=(0, 10),
                               ha='center',
                               fontsize=11,
                               color='red',
                               bbox=bbox_props)
            annotations.append(text)

    adjust_text(annotations, ax=ax, autoalign='xy')

    font_properties = FontProperties(family='Fira Sans', size='x-large')
    plt.title(f'{driver} LONG RUNS IN {str(session.event.year) + " " + session.event.Country + " " + session.name}',
              font='Fira Sans', fontsize=20)
    ax.legend(prop=font_properties, loc='upper left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    sns.despine(left=True, bottom=True)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.figtext(0.01, 0.02, '@Big_Data_Master', font='Fira Sans', fontsize=17, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"../PNGs/{driver} LAPS {session.event.OfficialEventName}.png", dpi=450)
    plt.show()


def long_runs_scatter(session, threshold=1.07):
    """
         Plots the long runs in FP2 for a given driver

         Parameters:
         race (Session): Session to analyze
         driver (str): Driver

    """

    plotting.setup_mpl(misc_mpl_mods=False)
    drivers = list(session.laps['Driver'].unique())
    driver_laps = pd.DataFrame(columns=['Driver', 'Laps', 'Compound', 'Average'])
    for d in drivers:
        d_laps = session.laps.pick_driver(d)
        max_stint = d_laps['Stint'].value_counts()
        max_stint = max_stint.reset_index()
        max_stint.columns = ['Stint', 'Counts']
        max_stint = max_stint.sort_values(by=['Counts', 'Stint'], ascending=[False, False])
        max_stint = max_stint['Stint'].iloc[0]
        print(f'{d}\n {d_laps[["Stint", "Compound"]].value_counts()}')
        driver_laps_filter = d_laps[d_laps['Stint'] == max_stint].pick_quicklaps(threshold).pick_wo_box()
        stint_index = 1
        try:
            while len(driver_laps_filter) < 5:
                max_stint = d_laps['Stint'].value_counts().index[stint_index]
                driver_laps_filter = d_laps[d_laps['Stint'] == max_stint].pick_quicklaps(threshold).pick_wo_box()
                stint_index += 1
            driver_laps_filter = driver_laps_filter[driver_laps_filter['LapTime'].notna()]
            driver_laps_filter['LapNumber'] = driver_laps_filter['LapNumber'] - driver_laps_filter[
                'LapNumber'].min() + 1
            driver_laps_filter = driver_laps_filter.reset_index()
            df_append = pd.DataFrame({
                'Driver': d,
                'Laps': [driver_laps_filter['LapTime'].to_list()],
                'Compound': [driver_laps_filter['Compound'].iloc[0]],
                'Average': driver_laps_filter['LapTime'].median()
            })
            driver_laps = pd.concat([driver_laps, df_append], ignore_index=True)
        except:
            print(f'NO DATA FOR {d}')

    driver_laps = driver_laps.sort_values(by=['Compound', 'Average'], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(8, 8))

    for idx, row in driver_laps.iterrows():
        driver = row['Driver']
        laps = row['Laps']
        tyre = row['Compound']
        hex_color = plotting.COMPOUND_COLORS[tyre]
        color_factor = np.linspace(0, 0.6, len(laps))
        color_index = 0
        for lap in laps:
            color = darken_color(hex_color, amount=round(color_factor[color_index], 1))
            plt.scatter(lap, driver, color=color, s=80)
            color_index += 1

    ax.set_xlabel("Lap Time", font='Fira Sans', fontsize=16)
    ax.set_ylabel("Driver", font='Fira Sans', fontsize=16)
    plt.grid(color='w', which='major', axis='x', linestyle='--')
    plt.xticks(font='Fira Sans', fontsize=14)
    plt.yticks(font='Fira Sans', fontsize=14)
    plt.title(f'LONG RUNS IN {str(session.event.year) + " " + session.event.Country + " " + session.name}',
              font='Fira Sans', fontsize=20)
    sns.despine(left=True, bottom=True)
    plt.figtext(0.01, 0.02, '@Big_Data_Master', font='Fira Sans', fontsize=17, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"../PNGs/ LAPS {session.event.OfficialEventName}.png", dpi=450)
    plt.show()


def session_diff_last_year(year, round_id, circuit=None, session='Q'):
    """
       Plot the performance of all teams against last year qualify in a specific circuit

       Parameters:
       year (int): Year
       round_id (int): Round of the GP
       circuit (str, optional): Only to get the teams in case of error. Default = None

    """
    ergast = Ergast()
    current = ergast.get_qualifying_results(season=2023, round=round_id, limit=1000)
    previous = ergast.get_qualifying_results(season=year, limit=1000)
    if circuit is not None:
        get_teams = ergast.get_qualifying_results(season=2023, round=1, limit=1000)
        current_circuit = circuit
        teams = get_teams.content[0]['constructorId'].values
    else:
        current_circuit = current.description['circuitId'].values[0]
        teams = current.content[0]['constructorId'].values
    index_pre_circuit = previous.description[previous.description['circuitId'] == current_circuit]['circuitId'].index[0]

    teams = pd.Series(teams).drop_duplicates(keep='first').values
    teams[teams == 'alfa'] = 'alfa romeo'

    previous = fastf1.get_session(year, index_pre_circuit + 1, session)
    previous.load()
    current = fastf1.get_session(2023, round_id, session)
    current.load()

    teams_to_plot = current.results['TeamName'].values
    teams_to_plot = pd.Series(teams_to_plot).drop_duplicates(keep='first').values

    delta_times = []
    colors = []
    ordered_teams = {}
    for team in teams_to_plot:
        fastest_lap = current.laps.pick_team(team).pick_fastest()['LapTime']
        ordered_teams[team] = fastest_lap

    teams_to_plot = list(dict(sorted(ordered_teams.items(), key=lambda item: item[1], reverse=False)).keys())

    for team in teams_to_plot:
        fast_current = current.laps.pick_team(team).pick_fastest()['LapTime']
        if team == 'Alfa Romeo' and year == 2021:
            team_prev = 'Alfa Romeo Racing'
        else:
            team_prev = team
        fast_prev = previous.laps.pick_team(team_prev).pick_fastest()['LapTime']

        delta_time = fast_current.total_seconds() - fast_prev.total_seconds()
        delta_times.append(round(delta_time, 3))

        color = '#' + current.results[current.results['TeamName'] == team]['TeamColor'].values[0]
        if team == 'Alpine':
            colors.append('#FF69B4')
        else:
            colors.append(color)

    fig, ax = plt.subplots(figsize=(8, 8))
    bars = ax.bar(teams_to_plot, delta_times, color=colors)

    round_bars(bars, ax, colors, y_offset_rounded=0)
    annotate_bars(bars, ax, 0.02, 12.25, '+{height}s',
                  ceil_values=False, round=3, y_negative_offset=-0.06)

    plt.axhline(0, color='white', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', linewidth=0.7, color='gray')
    plt.title(f'{current.event.Location.upper()} {current.name.upper()} COMPARISON: {year} vs. 2023', font='Fira Sans',
              fontsize=18)
    plt.xlabel('Team', font='Fira Sans', fontsize=16)
    plt.ylabel('Time diff (seconds)', font='Fira Sans', fontsize=16)
    plt.xticks(rotation=90, font='Fira Sans', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'../PNGs/{current.event.EventName} QUALY COMPARATION {year} vs 2023.png', dpi=450)
    plt.show()


def telemetry_lap(session, d1, lap):
    """
       Plot the telemetry of a lap

       Parameters:
       session (Session): Session of the lap
       d1 (str): Driver
       lap (int): Number of the lap

    """

    d1_lap = None
    for i in session.laps.pick_driver(d1).pick_lap(lap).iterlaps():
        d1_lap = i[1]
    d1_tel = d1_lap.get_telemetry()

    # d1_tel = session.car_data['16'].add_distance()
    # d1_tel = d1_tel[13300:13800]
    # initial_value = d1_tel['Distance'].iloc[0]
    # d1_tel['Distance'] = d1_tel['Distance'] - initial_value

    fig, ax = plt.subplots(nrows=4, figsize=(9, 7.5), gridspec_kw={'height_ratios': [4, 1, 1, 2]}, dpi=150)

    ax[0].plot(d1_tel['Distance'], d1_tel['Speed'],
               color='#FFA500')

    ax[0].set_xlabel('Distance')
    ax[0].set_ylabel('Speed')

    ax[1].plot(d1_tel['Distance'], d1_tel['Throttle'],
               color='#FFA500')
    ax[1].set_xlabel('Distance')
    ax[1].set_ylabel('Throttle')

    ax[2].plot(d1_tel['Distance'], d1_tel['Brake'],
               color='#FFA500')
    ax[2].set_xlabel('Distance')
    ax[2].set_ylabel('Brakes')

    ax[3].plot(d1_tel['Distance'], d1_tel['RPM'],
               color='#FFA500')
    ax[3].set_xlabel('Distance')
    ax[3].set_ylabel('RPM')

    ax[2].set_yticks([0, 1])  # Assuming the 'Brakes' data is normalized between 0 and 1
    ax[2].set_yticklabels(['OFF', 'ON'])

    ax[0].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[0].set_title(f'{d1} LAP {lap} IN {session.event.EventName}', font='Fira Sans', fontsize=18)
    ax[1].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[2].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[3].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'../PNGs/{d1} lap {lap}.png', dpi=500)
    plt.show()


def overlying_laps(session, driver_1, driver_2, lap=None):
    """
       Compare the telemetry of 2 different laps

       Parameters:
       session (Session): Session of the lap
       driver_1 (str): Driver 1
       driver_2 (str): Driver 2
       lap (int, optional): Number of the lap. Default: None

    """

    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['figure.facecolor'] = 'black'

    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'

    if lap is not None:
        for i in session.laps.pick_driver(driver_1).pick_lap(lap).iterlaps():
            d1_lap = i[1]

        for i in session.laps.pick_driver(driver_2).pick_lap(lap).iterlaps():
            d2_lap = i[1]

    else:
        d1_lap = session.laps.pick_driver('HAM')
        count = 1
        for i in d1_lap.iterlaps():
            if count == 20:
                d1_lap = i[1]
            elif count == 19:
                d2_lap = i[1]
            count += 1
        # # d2_lap = session.laps.split_qualifying_sessions()[2].pick_driver('NOR').pick_quicklaps()
        # d1_lap = session.laps.pick_driver(driver_1).pick_fastest()
        # d2_lap = session.laps.pick_driver(driver_2).pick_fastest()

    delta_time, ref_tel, compare_tel = utils.delta_time(d1_lap, d2_lap)

    final_value = ((d2_lap['LapTime'] - d1_lap['LapTime']).total_seconds())

    def adjust_to_final(series, final_value):
        diff = final_value - series.iloc[-1]
        adjustments = [diff * (i + 1) / len(series) for i in range(len(series))]
        adjusted_series = series + adjustments
        return adjusted_series

    delta_time = adjust_to_final(delta_time, final_value)
    fig, ax = plt.subplots(nrows=3, figsize=(9, 8), gridspec_kw={'height_ratios': [4, 1, 1]}, dpi=150)

    ax[0].plot(ref_tel['Distance'], ref_tel['Speed'],
               color='#0000FF',
               label=driver_1, linewidth=2.75)
    ax[0].plot(compare_tel['Distance'], compare_tel['Speed'],
               color='#FFA500',
               label=driver_2, linewidth=2.75)

    colors = ['green' if x > 0 else 'red' for x in delta_time]
    twin = ax[0].twinx()
    for i in range(1, len(delta_time)):
        twin.plot(ref_tel['Distance'][i - 1:i + 1], delta_time[i - 1:i + 1], color=colors[i],
                  alpha=0.6, label='delta', linewidth=2)

    # Set the labels for the axes
    ax[0].set_xlabel('Distance')
    ax[0].set_ylabel('Speed')
    ax[0].set_title(f'{str(session.date.year) + " " + session.event.EventName + " " + session.name}'
                    f'{" Lap " + str(lap) if lap is not None else ""} comparison: {driver_1} VS {driver_2}',
                    font='Fira Sans', fontsize=16, y=1.1)

    # ax[0].set_title(f'ALONSO POTENTIAL BRAKE CHECK',
    #                 font='Fira Sans', fontsize=18, y=1.1)

    twin.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    twin.set_ylabel('Time diff (s)')

    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label=f'{driver_1} ahead'),
        Line2D([0], [0], color='red', lw=2, label=f'{driver_2} ahead')
    ]

    handles1, labels1 = ax[0].get_legend_handles_labels()
    handles1 = handles1 + legend_elements
    labels1 = labels1 + [entry.get_label() for entry in legend_elements]
    ax[0].legend(handles1, labels1, loc='lower left')
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)

    ax[1].plot(ref_tel['Distance'], ref_tel['Brake'],
               color='#0000FF',
               label=driver_1)
    ax[1].plot(compare_tel['Distance'], compare_tel['Brake'],
               color='#FFA500',
               label=driver_2)

    ax[1].set_xlabel('Distance')
    ax[1].set_ylabel('Brakes')

    ax[1].set_yticks([0, 1])  # Assuming the 'Brakes' data is normalized between 0 and 1
    ax[1].set_yticklabels(['OFF', 'ON'])

    ax[2].plot(ref_tel['Distance'], ref_tel['Throttle'],
               color='#0000FF',
               label=driver_1)
    ax[2].plot(compare_tel['Distance'], compare_tel['Throttle'],
               color='#FFA500',
               label=driver_2)

    ax[2].set_xlabel('Distance')
    ax[2].set_ylabel('Throttle')

    ax[2].set_yticks([0, 50, 100])  # Assuming the 'Brakes' data is normalized between 0 and 1
    ax[2].set_yticklabels(['0%', '50%', '100%'])

    # ax[3].plot(ref_tel['Distance'], ref_tel['nGear'],
    #            color='#0000FF',
    #            label=driver_1)
    # ax[3].plot(compare_tel['Distance'], compare_tel['nGear'],
    #            color='#FFA500',
    #            label=driver_2)
    #
    # ax[3].set_xlabel('Distance')
    # ax[3].set_ylabel('Gear')
    #
    # ax[3].set_yticks([2, 3, 4, 5, 6, 7, 8])
    # ax[3].set_yticklabels(['2', '3', '4', '5', '6', '7', '8'])

    ax[1].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[2].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    # ax[3].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'../PNGs/{driver_1} - {driver_2} + {session.event.EventName + " " + session.name}.png', dpi=450)
    plt.show()


def plot_circuit_with_data(session, col='nGear'):
    """
       Plot the circuit with the data desired

       Parameters:
       session (Session): Session of the lap
       col (str): Name of the data to plot

    """

    lap = session.laps.pick_fastest()
    tel = lap.get_telemetry()

    x = np.array(tel['X'].values)
    y = np.array(tel['Y'].values)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    data = tel[col].to_numpy().astype(float)
    text = col
    if col == 'nGear':
        cmap = cm.get_cmap('Paired')
        lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N + 1), cmap=cmap)
        text = 'Gear Shift'
    else:
        cmap = cm.get_cmap('coolwarm')
        min_speed, max_speed = data.min(), data.max()  # Min and max speeds for normalization
        lc_comp = LineCollection(segments, norm=plt.Normalize(min_speed, max_speed), cmap=cmap)
        lc_comp.set_array(data)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    lc_comp.set_array(data)
    lc_comp.set_linewidth(4)

    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    plt.suptitle(
        f"Fastest Lap {text} Visualization\n"
        f"{lap['Driver']} - {session.event['EventName']} {session.event.year}", font='Fira Sans', fontsize=20
    )
    if col == 'nGear':
        cbar = plt.colorbar(mappable=lc_comp, label="Gear", boundaries=np.arange(1, 10))
        cbar.set_ticks(np.arange(1.5, 9.5), fontsize=14, fontname='Fira Sans')
        cbar.set_ticklabels(np.arange(1, 9), fontsize=14, fontname='Fira Sans')
    else:
        cbar = plt.colorbar(mappable=lc_comp)
        cbar.set_label("Speed", fontsize=16, fontname='Fira Sans', x=0.8)
    for label in cbar.ax.get_yticklabels():
        label.set_size(14)
        label.set_family('Fira Sans')
    plt.xticks(fontsize=14, fontname='Fira Sans')
    plt.yticks(fontsize=14, fontname='Fira Sans')
    plt.savefig(f"../PNGs/GEAR CHANGES {session.event.OfficialEventName}", dpi=400)
    plt.show()


def fastest_by_point(session, team_1, team_2, scope='Team'):
    """
       Plot the circuit with the time diff between 2 laps at each points

       Parameters:
       session (Session): Session of the lap
       team_1 (str): Team 1
       team_2 (str): Team 2
       scope (str, optional): Scope of the plot (Team|Driver). Default: Team

    """

    if scope == 'Team':
        lap_team_1 = session.laps.pick_team(team_1).pick_fastest()
        tel_team_1 = lap_team_1.get_telemetry()
        lap_team_2 = session.laps.pick_team(team_2).pick_fastest()
    else:
        lap_team_1 = session.laps.pick_driver(team_1).pick_fastest()
        tel_team_1 = lap_team_1.get_telemetry()
        lap_team_2 = session.laps.pick_driver(team_2).pick_fastest()

    delta_time, ref_tel, compare_tel = utils.delta_time(lap_team_1, lap_team_2)
    final_value = ((lap_team_2['LapTime'] - lap_team_1['LapTime']).total_seconds())

    def adjust_to_final(series, final_value):
        diff = final_value - series.iloc[-1]
        adjustments = [diff * (i + 1) / len(series) for i in range(len(series))]
        adjusted_series = series + adjustments
        return adjusted_series

    delta_time = adjust_to_final(delta_time, final_value)

    x = np.array(tel_team_1['X'].values)
    y = np.array(tel_team_1['Y'].values)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if len(delta_time) != len(segments):
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(delta_time))
        x_new = np.linspace(0, 1, len(segments))
        f = interp1d(x_old, delta_time, kind='linear')
        delta_time = f(x_new)
    cmap = cm.get_cmap('coolwarm')

    vmax = np.max(np.abs(delta_time))
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    segments = rotate(segments, session.get_circuit_info().rotation / 180 * np.pi)
    lc_comp = LineCollection(segments, norm=norm, cmap=cmap)
    lc_comp.set_array(delta_time)
    lc_comp.set_linewidth(7)


    plt.subplots(figsize=(10, 10))
    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    plot_turns(session.get_circuit_info(), session.get_circuit_info().rotation / 180 * np.pi, plt)
    cbar = plt.colorbar(mappable=lc_comp, label="DeltaTime")
    legend_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=4)]

    plt.legend(legend_lines, [f'{team_1} ahead', f'{team_2} ahead'], fontsize='x-large')
    cbar.set_label('Time Difference (s)', rotation=270, labelpad=20, fontsize=20)

    cbar.ax.tick_params(labelsize=12)
    for label in cbar.ax.get_yticklabels():
        label.set_size(18)
        label.set_family('Fira Sans')

    plt.suptitle(f"{team_1} vs {team_2}:"
                 f" {str(session.session_info['StartDate'].year) + ' ' + session.event.EventName + ' ' + session.name} \n",
                 font='Fira Sans', fontsize=24)
    plt.tight_layout()
    path = (f"../PNGs/Dif by point {team_1} vs {team_2} - {str(session.session_info['StartDate'].year)}"
            f" {session.event.EventName + ' ' + session.name}.png")
    plt.savefig(path, dpi=150)
    plt.show()


def track_dominance(session, team_1, team_2):
    """
       Plot the track dominance of 2 teams in their fastest laps

       Parameters:
       session(Session): Session to analyze
       team_1 (str): Team 1
       team_2 (str): Team 2
    """

    lap_team_1 = session.laps.pick_team(team_1).pick_fastest()
    tel_team_1 = lap_team_1.get_telemetry()
    lap_team_2 = session.laps.pick_team(team_2).pick_fastest()
    delta_time, ref_tel, compare_tel = utils.delta_time(lap_team_1, lap_team_2)
    final_value = ((lap_team_2['LapTime'] - lap_team_1['LapTime']).total_seconds())

    def adjust_to_final(series, final_value):
        diff = final_value - series.iloc[-1]
        adjustments = [diff * (i + 1) / len(series) for i in range(len(series))]
        adjusted_series = series + adjustments
        return adjusted_series

    delta_time = adjust_to_final(delta_time, final_value)
    delta_time = delta_time.reset_index(drop=True)
    num_stretches = 30
    stretch_len = len(delta_time) // num_stretches
    stretched_delta_time = []

    for i in range(num_stretches):
        start_idx = i * stretch_len + delta_time.index[0]
        if start_idx != 0:
            start_idx -= 1
        end_idx = (i + 1) * stretch_len - 1 + delta_time.index[0]
        start_value = delta_time[start_idx]
        end_value = delta_time[end_idx]
        stretch_value = 0 if start_value < end_value else 1
        stretched_delta_time.extend([stretch_value] * stretch_len)

    if len(delta_time) % num_stretches != 0:
        start_value = delta_time[num_stretches * stretch_len]
        end_value = delta_time.iloc[-1]
        stretch_value = 0 if start_value < end_value else 1
        stretched_delta_time.extend([stretch_value] * (len(delta_time) - num_stretches * stretch_len))

    delta_time_team = np.array(stretched_delta_time)

    x = np.array(tel_team_1['X'].values)
    y = np.array(tel_team_1['Y'].values)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    segments = rotate(segments, session.get_circuit_info().rotation / 180 * np.pi)
    colors = [plotting.team_color(team_1), plotting.team_color(team_2)]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=2)
    lc_comp = LineCollection(segments, cmap=cmap, joinstyle='bevel')
    lc_comp.set_array(delta_time_team)
    lc_comp.set_linewidth(10)

    plt.subplots(figsize=(10, 8))
    border_width = 12
    border_color = 'white'
    lc_border = LineCollection(segments, colors=border_color, linewidths=border_width)
    plt.gca().add_collection(lc_border)
    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    plot_turns(session.get_circuit_info(), session.get_circuit_info().rotation / 180 * np.pi, plt)
    legend_lines = [Line2D([0], [0], color=plotting.team_color(team_1), lw=4),
                    Line2D([0], [0], color=plotting.team_color(team_2), lw=4)]

    plt.legend(legend_lines, [f'{team_1} faster', f'{team_2} faster'], loc='lower left', fontsize='x-large')

    plt.suptitle(f"TRACK DOMINANCE {team_1} vs {team_2}:"
                 f" {str(session.session_info['StartDate'].year) + ' ' + session.event.EventName} \n", font='Fira Sans',
                 fontsize=20)
    plt.tight_layout()
    path = (f"../PNGs/TRACK DOMINANCE{team_1} vs {team_2} - {str(session.session_info['StartDate'].year)}"
            f" {session.event.EventName}.png")
    plt.savefig(path, dpi=400)
    plt.show()





def get_fastest_data(session, column='Speed', fastest_lap=False, DRS=True):
    """
        Get the fastest data in a session

        Parameters:
        session (Session): Session to be analyzed
        column (str): Data to plot
        fastest_lap (bool): Get only data from the fastest lap

   """

    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    drivers = session.laps['Driver'].groupby(session.laps['Driver']).size()
    drivers = drivers.reset_index(name='Count')['Driver'].to_list()
    circuit_speed = {}
    colors_dict = {}
    for driver in drivers:
        try:
            d_laps = session.laps.pick_driver(driver)
            if fastest_lap:
                d_laps = session.laps.pick_driver(driver).pick_fastest()
            if len(d_laps) > 0:
                if not DRS:
                    d_laps = d_laps.telemetry[d_laps.telemetry['DRS'] >= 10]
                top_speed = max(d_laps.telemetry['Speed'])
                if fastest_lap:
                    team = d_laps['Team'].lower()
                else:
                    team = d_laps['Team'].values[0].lower()
                if team == 'red bull racing':
                    team = 'red bull'
                elif team == 'haas f1 team':
                    team = 'haas'
                circuit_speed[driver] = top_speed
                colors_dict[driver] = team
        except KeyError:
            print(f'No data for {driver}')
        print(circuit_speed)

    order = True
    if fastest_lap:
        column = 'Top Speeds (only the fastest lap from each driver)'
    else:
        column = 'Top Speeds'
    x_fix = 5
    y_fix = 0.25
    annotate_fontsize = 13
    y_offset_rounded = 0.035
    round_decimals = 0

    circuit_speed = {k: v for k, v in sorted(circuit_speed.items(), key=lambda item: item[1], reverse=order)}

    fig, ax1 = plt.subplots(figsize=(8, 6.5), dpi=175)

    colors = []
    for i in range(len(circuit_speed)):
        colors.append(fastf1.plotting.TEAM_COLORS[colors_dict[list(circuit_speed.keys())[i]]])

    bars = ax1.bar(list(circuit_speed.keys()), list(circuit_speed.values()), color=colors,
                   edgecolor='white')

    round_bars(bars, ax1, colors, y_offset_rounded=y_offset_rounded)
    annotate_bars(bars, ax1, y_fix, annotate_fontsize, text_annotate='default', ceil_values=False, round=round_decimals)

    ax1.set_title(f'{column} in {str(session.event.year) + " " + session.event.Location + " " + session.name}',
                  font='Fira Sans', fontsize=14)
    ax1.set_xlabel('Driver', fontweight='bold', fontsize=12)

    y_label = 'Max speed'
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.set_ylabel(y_label, fontweight='bold', fontsize=12)
    max_value = max(circuit_speed.values())
    ax1.set_ylim(min(circuit_speed.values()) - x_fix, max_value + x_fix)
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)

    def format_ticks(val, pos):
        return '{:.0f}'.format(val)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks))

    plt.xticks(fontsize=12, rotation=45)
    color_index = 0
    for label in ax1.get_xticklabels():
        label.set_color('white')
        label.set_fontsize(14)
        label.set_rotation(35)
        for_color = colors[color_index]
        if for_color == '#ffffff':
            for_color = '#FF7C7C'
        label.set_path_effects([path_effects.withStroke(linewidth=2, foreground=for_color)])
        color_index += 1
    ax1.yaxis.grid(True, linestyle='--')
    ax1.xaxis.grid(False)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'../PNGs/{column} IN {str(session.event.year) + " " + session.event.Country + " " + session.name}',
                dpi=450)
    plt.show()


def air_track_temps():
    plt.rcParams['font.family'] = 'Fira Sans'
    plt.rcParams['font.sans-serif'] = 'Fira Sans'
    with open("awards/sessions.pkl", "rb") as file:
        sessions = pickle.load(file)
    race_names = []
    track_temp = []
    air_temp = []
    count = 0
    for s in sessions:
        if s.event.Location not in race_names:
            race_names.append(s.event.Location)
            track_temp.append(min(s.weather_data['TrackTemp']))
            air_temp.append(min(s.weather_data['AirTemp']))
            count += 1
        else:
            prev_track_temp = track_temp[count - 1]
            curr_track_temp = min(s.weather_data['TrackTemp'])
            curr_air_temp = min(s.weather_data['AirTemp'])

            if curr_track_temp < prev_track_temp and curr_track_temp != 0:
                track_temp[count - 1] = curr_track_temp
                air_temp[count - 1] = curr_air_temp

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.scatter(track_temp, air_temp, color='orange')

    texts = []
    delta = []
    delta_dict = {}
    for i, txt in enumerate(race_names):
        ax.scatter(track_temp[i], air_temp[i], color='orange', s=70)
        texts.append(ax.text(track_temp[i], air_temp[i], txt, fontname='Fira Sans', color='white', fontsize=11))
        diff = track_temp[i] - air_temp[i]
        delta.append(diff)
        delta_dict[race_names[i]] = diff

    print(f'MEAN: {statistics.mean(delta)}')
    print(f'MEDIAN: {statistics.median(delta)}')

    delta_dict = dict(sorted(delta_dict.items(), key=lambda item: item[1], reverse=False))
    for r, d in delta_dict.items():
        print(f'{r}: {round(d, 2)}')

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # Find the lower bound of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # Find the upper bound of both axes
    ]
    ax.plot(lims, lims, color='white', alpha=0.6, zorder=-10)  # Plot the x=y line

    adjust_text(texts)
    print(race_names)
    print(track_temp)
    print(air_temp)
    plt.title("Minimum Track and Air Temperatures in 2023 season", font='Fira Sans', fontsize=18)
    plt.xlabel("Track Temperature (°C)", fontname='Fira Sans', fontsize=17)
    plt.ylabel("Air Temperature (°C)", fontname='Fira Sans', fontsize=17)
    plt.xticks(fontname='Fira Sans', fontsize=15)
    plt.yticks(fontname='Fira Sans', fontsize=15)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('../PNGs/Track temps.png', dpi=450)
    plt.show()


def session_results(start, end, session=2):
    count = 0
    for i in range(end, start - 1, -1):
        year_data = fastf1.get_event_schedule(i)
        year_data = year_data[year_data['EventFormat'] != 'testing']
        for j in range(len(year_data), 0, -1):
            try:
                practice_times = {}
                s = fastf1.get_session(i, j, session)
                s.load()
                drivers = s.laps['Driver'].unique()
                for d in drivers:
                    d_lap = s.laps.pick_driver(d).pick_fastest()['LapTime']
                    team = s.laps.pick_driver(d).pick_fastest()['Team']
                    if not pd.isna(d_lap):
                        practice_times[f'{d} + {team}'] = d_lap
                practice_times = dict(sorted(practice_times.items(), key=lambda item: item[1], reverse=False))
                top2 = list(practice_times.keys())[2]
                print(top2)
                if 'PIA + McLaren' in top2:
                    print(s)
                    if count == 1:
                        exit(0)
                    count += 1
            except:
                print('No data')


def drs_efficiency(year):

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    drs_dict = {}
    for i in range(len(schedule)):
        from src.utils.utils import call_function_from_module
        try:
            call_function_from_module(qualy_by_year, f"year_{year}", i + 1)
            if i == 3:
                raise QualyException
            current_session_delta = {}
            session = fastf1.get_session(year, i + 1, 'Q')
            session.load()
            teams = session.laps['Team'].unique()
            teams = [t for t in teams if not pd.isna(t)]
            for t in teams:
                team_lap = session.laps.pick_team(t).pick_fastest()
                drs_data = team_lap.telemetry['DRS'].replace(14, 12).replace(10, 12).diff().fillna(0)
                open_close = drs_data[drs_data != 0]
                open_close = remove_close_rows(open_close)
                if len(open_close) == 0:
                    raise QualyException
                indexes = list(open_close.index.values)
                if open_close[0] == -4:
                    first_index = open_close.index[0]
                    indexes.remove(first_index)
                    indexes.append(first_index)
                for d in range(int(len(indexes)/2)):
                    speed_open = team_lap.telemetry['Speed'][indexes[0+d*2]]
                    try:
                        speed_close = max(team_lap.telemetry['Speed'][indexes[0+d*2]:indexes[1+d*2]])
                    except ValueError:
                        max_index = max(team_lap.telemetry['Speed'].index)
                        before_finish_line = team_lap.telemetry['Speed'][indexes[0+d*2]:max_index]
                        after_finish_line = team_lap.telemetry['Speed'][0:indexes[1+d*2]+1]
                        speed_close = max(pd.concat([before_finish_line, after_finish_line]))
                    delta_diff = ((speed_close - speed_open) / speed_open) * 100
                    if t not in current_session_delta:
                        current_session_delta[t] = [delta_diff]
                    else:
                        current_session_delta[t].append(delta_diff)

            if len(current_session_delta) == 10:
                for t in current_session_delta.keys():
                    print(f'{t} - {len(current_session_delta[t])}')
                    if t not in drs_dict:
                        drs_dict[t] = current_session_delta[t]
                    else:
                        drs_dict[t].extend(current_session_delta[t])
            else:
                print(current_session_delta)
        except QualyException:
            print(f'No data for {year}-{i + 1}')
    df = pd.DataFrame(drs_dict).melt(var_name='Team', value_name='Delta').groupby('Team')['Delta'].mean().reset_index()
    df = df.sort_values(by='Delta', ascending=False)
    df['Team'].replace({'Haas F1 Team': 'Haas', 'Red Bull Racing': 'Red Bull'}, inplace=True)
    fix, ax = plt.subplots(figsize=(9, 8))
    bars = plt.bar(df['Team'], df['Delta'])
    colors = [team_colors_2023.get(i) for i in df['Team']]
    round_bars(bars, ax, colors, color_1=None, color_2=None, y_offset_rounded=0.03, corner_radius=0.1, linewidth=4)
    annotate_bars(bars, ax, 0.1, 14, text_annotate='default', ceil_values=False, round=2,
                  y_negative_offset=0.04, annotate_zero=False, negative_offset=0, add_character='%')

    plt.title(f'DRS EFFICIENCY IN {year} SEASON', font='Fira Sans', fontsize=24)
    plt.xlabel('Team', font='Fira Sans', fontsize=20)
    plt.ylabel('Drs efficiency (%)', font='Fira Sans', fontsize=20)
    plt.xticks(rotation=35, font='Fira Sans', fontsize=18)
    plt.yticks(font='Fira Sans', fontsize=16)
    plt.ylim(bottom=min(df['Delta'])-1, top=max(df['Delta'])+1)
    color_index = 0
    for label in ax.get_xticklabels():
        label.set_color('white')
        label.set_fontsize(16)
        label.set_rotation(35)
        for_color = colors[color_index]
        if for_color == '#ffffff':
            for_color = '#FF7C7C'
        label.set_path_effects([path_effects.withStroke(linewidth=2, foreground=for_color)])
        color_index += 1
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'../PNGs/DRS EFFICIENCY {year}.png', dpi=450)
    plt.show()


def get_fastest_sectors(session):
    """
        Get the fastest data in a session

        Parameters:
        session (Session): Session to be analyzed
   """

    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    drivers = session.laps['Driver'].groupby(session.laps['Driver']).size()
    drivers = drivers.reset_index(name='Count')['Driver'].to_list()
    sector_times = pd.DataFrame(columns=['Sector1Time', 'Sector2Time', 'Sector3Time', 'Color'])
    for driver in drivers:
        try:
            d_laps = session.laps.pick_driver(driver)
            for sector in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
                if len(d_laps) > 0:
                    time = round(min(d_laps[sector].dropna()).total_seconds(), 3)
                    team = d_laps['Team'].values[0]
                    sector_times.loc[driver, sector] = time
                    sector_times.loc[driver, f'Color'] = team_colors_2023.get(team)
        except KeyError:
            print(f'No data for {driver}')

    y_fix = 0.015
    annotate_fontsize = 13.25
    y_offset_rounded = 0.085
    round_decimals = 3

    fig, ax = plt.subplots(nrows=3, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 2, 2]}, dpi=150)
    fig.suptitle(f'SECTOR TIMES IN {str(session.event.year) + " " + session.event.Country + " " + session.name}',
                 font='Fira Sans', fontsize=20, fontweight='bold')
    for i in range(3):
        times = sector_times.sort_values(by=f'Sector{i+1}Time', ascending=True)
        colors = times['Color'].values
        bars = ax[i].bar(times.index.values, times[f'Sector{i+1}Time'], color=colors)

        round_bars(bars, ax[i], colors, y_offset_rounded=y_offset_rounded, linewidth=2.75)
        annotate_bars(bars, ax[i], y_fix, annotate_fontsize, text_annotate='default',
                      ceil_values=False, round=round_decimals, linewidth=1.5, egdecolor='#004764')

        ax[i].set_title(f'Sector {i+1} Times', font='Fira Sans', fontsize=18)
        ax[i].set_xlim(-0.5, len(times)-0.5)

        max_value = max(times[f'Sector{i+1}Time'])
        ax[i].set_ylim(min(times[f'Sector{i+1}Time']) - 0.25, max_value + 0.25)
        labels = [item.get_text() for item in ax[i].get_xticklabels()]
        ax[i].set_xticklabels(labels, fontsize=12, rotation=45)
        color_index = 0
        for label in ax[i].get_xticklabels():
            label.set_color('white')
            label.set_fontsize(14)
            label.set_rotation(35)
            for_color = colors[color_index]
            if for_color == '#ffffff':
                for_color = '#FF7C7C'
            label.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground=for_color)])
            color_index += 1
        ax[i].yaxis.grid(True, linestyle='--', alpha=0.25)
        ax[i].xaxis.grid(False)
        labels = [item.get_text() for item in ax[i].get_yticklabels()]
        ax[i].set_yticklabels(labels, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'../PNGs/SECTOR TIMES IN {str(session.event.year) + " " + session.event.Country + " " + session.name}',
                dpi=450)
    plt.show()