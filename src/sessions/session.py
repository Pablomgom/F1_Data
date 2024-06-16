import pickle
import statistics

import fastf1
import numpy as np
import pandas as pd
from adjustText import adjust_text
from fastf1 import plotting, set_log_level
from matplotlib import pyplot as plt, ticker, cm
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as path_effects
from scipy import stats
from src.exceptions.custom_exceptions import QualyException
from src.utils import utils
from src.plots.plots import round_bars, annotate_bars
from src.utils.utils import darken_color, plot_turns, rotate, remove_close_rows, create_rounded_barh, format_timedelta
import seaborn as sns
from src.variables.team_colors import team_colors_2023, team_colors


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
    ax.legend(prop=font_properties, loc='lower right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    sns.despine(left=True, bottom=True)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.figtext(0.01, 0.02, '@F1BigData', font='Fira Sans', fontsize=17, color='gray', alpha=0.5)
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
    driver_laps = pd.DataFrame(columns=['Driver', 'Laps', 'Compound', 'Average', 'Top speeds', 'RPMs'])
    for d in drivers:
        d_laps = session.laps.pick_driver(d)
        max_stint = d_laps[d_laps['Compound'].isin(['HARD', 'MEDIUM', 'SOFT'])]['Stint'].value_counts()
        max_stint = max_stint.reset_index()
        max_stint.columns = ['Stint', 'Counts']
        max_stint = max_stint.sort_values(by=['Counts', 'Stint'], ascending=[False, False])
        stint_index = 0
        driver_laps_dict = {}
        try:
            for i in range(len(max_stint)):
                stint_number = max_stint.iloc[stint_index]['Stint']
                driver_laps_filter = d_laps[d_laps['Stint'] == stint_number].pick_quicklaps(threshold).pick_wo_box()
                driver_laps_dict[stint_number] = len(driver_laps_filter)
                stint_index += 1

            driver_laps_dict = dict(sorted(driver_laps_dict.items(), key=lambda item: (-int(item[1]), -int(item[0]))))
            final_stint_number, number_laps = next(iter(driver_laps_dict.items()))
            if number_laps < 5:
                raise Exception
            print(f'{d} - STINT: {final_stint_number} - LAPS: {number_laps}')
            driver_laps_filter = d_laps[d_laps['Stint'] == final_stint_number].pick_quicklaps(threshold).pick_wo_box()
            driver_laps_filter = driver_laps_filter.reset_index()
            top_speeds = []
            total_rpms = []
            for l in driver_laps_filter.iterlaps():
                current_lap = l[1].telemetry
                speed = max(current_lap['Speed'])
                top_speeds.append(speed)
                rpms = current_lap[current_lap['nGear'] == 7]['RPM'].values
                total_rpms.extend(rpms)

            df_append = pd.DataFrame({
                'Driver': f'{d} ({len(driver_laps_filter)})',
                'Laps': [driver_laps_filter['LapTime'].to_list()],
                'Compound': [driver_laps_filter['Compound'].iloc[0]],
                'Average': driver_laps_filter['LapTime'].median(),
                'Top speeds': np.median(top_speeds),
                'RPMs': stats.trim_mean(total_rpms, 0.10)
            })
            driver_laps = pd.concat([driver_laps, df_append], ignore_index=True)
        except Exception as e:
            print(e)
            print(f'NO DATA FOR {d}')

    driver_laps = driver_laps.sort_values(by=['Compound', 'Average'], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(8, 8))
    for idx, row in driver_laps.iterrows():
        driver = row['Driver']
        laps = row['Laps']
        tyre = row['Compound']
        try:
            hex_color = plotting.COMPOUND_COLORS[tyre]
        except KeyError:
            hex_color = '#434649'
        color_factor = np.linspace(0, 0.6, len(laps))
        color_index = 0
        for lap in laps:
            color = darken_color(hex_color, amount=round(color_factor[color_index], 1))
            plt.scatter(lap.total_seconds(), driver, color=color, s=100)
            color_index += 1

    prev_tyre = None
    prev_time = None
    driver_laps = driver_laps.sort_values(by=['Compound', 'Average'], ascending=[False, True])
    for idx, row in driver_laps.iterrows():
        driver = row['Driver']
        laps = row['Average']
        tyre = row['Compound']
        if prev_time is not None and tyre == prev_tyre:
            time_diff = f'(+{format_timedelta(laps - prev_time)}s)'.replace('0:0', '')
        else:
            time_diff = ''
            prev_time = laps
        print(f'{driver.split(" (")[0]}: {format_timedelta(laps)} {time_diff}')
        prev_tyre = tyre
    print("""
        The higher the RPMs -> the higher the engine mode.
        These are the average RPMs while being in 7th gear.
    """)
    for idx, row in driver_laps.iterrows():
        driver = row['Driver']
        speeds = row['Top speeds']

        print(f'{driver.split(" (")[0]}: {speeds} km/h - {row["RPMs"]:.2f}')

    def format_timedelta_to_mins(seconds, pos):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f'{mins:01d}:{secs:02d}.{millis:03d}'

    formatter = FuncFormatter(format_timedelta_to_mins)
    plt.gca().xaxis.set_major_formatter(formatter)

    ax.set_xlabel("Lap Time", font='Fira Sans', fontsize=16)
    ax.set_ylabel("Driver", font='Fira Sans', fontsize=16)
    plt.grid(color='w', which='major', axis='x', linestyle='--', alpha=0.4)
    plt.grid(color='w', which='major', axis='y', linestyle='--', alpha=0.4)
    plt.xticks(font='Fira Sans', fontsize=14)
    plt.yticks(font='Fira Sans', fontsize=14)
    plt.title(f'LONG RUNS IN {str(session.event.year) + " " + session.event.Country + " " + session.name}',
              font='Fira Sans', fontsize=20)
    sns.despine(left=True, bottom=True)
    plt.figtext(0.01, 0.02, '@F1BigData', font='Fira Sans', fontsize=17, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"../PNGs/LAPS {session.event.OfficialEventName}.png", dpi=450)
    plt.show()


def session_diff_last_year(year, round_id, prev_year, session='Q'):
    """
       Plot the performance of all teams against last year qualify in a specific circuit

       Parameters:
       year (int): Year
       round_id (int): Round of the GP
       circuit (str, optional): Only to get the teams in case of error. Default = None

    """
    previous = fastf1.get_session(prev_year, round_id, session)
    previous.load()
    current = fastf1.get_session(year, round_id, session)
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
        # current_top_speed = max(current.laps.pick_team(team).pick_fastest().telemetry['Speed'])
        if team == 'Kick Sauber':
            team_prev = 'Alfa Romeo'
        elif team == 'RB':
            team_prev = 'AlphaTauri'
        else:
            team_prev = team

        fast_prev = previous.laps.pick_team(team_prev).pick_fastest()['LapTime']
        # prev_top_speed = max(previous.laps.pick_team(team_prev).pick_fastest().telemetry['Speed'])
        delta_time = fast_current.total_seconds() - fast_prev.total_seconds()
        delta_times.append(round(delta_time, 3))
        # print(f'{team} {current_top_speed - prev_top_speed}')
        color = team_colors.get(year)[team]
        colors.append(color)

    tweet = list(zip(teams_to_plot, delta_times))
    tweet = sorted(tweet, key=lambda x: x[1])
    slower = False
    print(f'🟢 Faster than in {prev_year}')
    for t in tweet:
        if t[1] > 0 and not slower:
            print(f'🔴 Slower than in {prev_year}')
            slower = True
        print(f'{t[0]}: {"+" if slower else ""}{t[1]:.3f}s')

    combined = zip(teams_to_plot, colors, delta_times)
    sorted_combined = sorted(combined, key=lambda x: x[2])
    unzipped = zip(*sorted_combined)
    teams_to_plot, colors, delta_times = [list(t) for t in unzipped]

    fig, ax = plt.subplots(figsize=(8, 8))
    bars = ax.bar(teams_to_plot, delta_times, color=colors)

    round_bars(bars, ax, colors, y_offset_rounded=0, linewidth=3.75)
    annotate_bars(bars, ax, 0.01, 14, '+{height}s',
                  ceil_values=False, round=3, y_negative_offset=-0.075)

    plt.axhline(0, color='white', linewidth=4)
    plt.grid(axis='y', linestyle='--', linewidth=0.7, color='gray')
    plt.title(f'{current.event.Location.upper()} {current.name.upper()} COMPARISON: {year} vs. {prev_year}',
              font='Fira Sans',
              fontsize=20)
    plt.xticks(rotation=90, font='Fira Sans', fontsize=16)
    plt.yticks(font='Fira Sans', fontsize=14)
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
    print(f'{d1_lap["LapTime"]}')

    # d1_tel = session.car_data['16'].add_distance()
    # d1_tel = d1_tel[13300:13800]
    # initial_value = d1_tel['Distance'].iloc[0]
    # d1_tel['Distance'] = d1_tel['Distance'] - initial_value

    fig, ax = plt.subplots(nrows=4, figsize=(9, 7.5), gridspec_kw={'height_ratios': [4, 1, 1, 1]}, dpi=150)

    ax[0].plot(d1_tel['Distance'], d1_tel['Speed'],
               color='#FFA500', linewidth=2)

    ax[0].set_xlabel('Distance')
    ax[0].set_ylabel('Speed')

    ax[1].plot(d1_tel['Distance'], d1_tel['Throttle'],
               color='#FFA500', linewidth=2)
    ax[1].set_xlabel('Distance')
    ax[1].set_ylabel('Throttle')

    ax[2].plot(d1_tel['Distance'], d1_tel['Brake'],
               color='#FFA500', linewidth=2)
    ax[2].set_xlabel('Distance')
    ax[2].set_ylabel('Brakes')

    ax[2].set_yticks([0, 1])  # Assuming the 'Brakes' data is normalized between 0 and 1
    ax[2].set_yticklabels(['OFF', 'ON'])

    ax[3].plot(d1_tel['Distance'], d1_tel['DRS'],
               color='#FFA500', linewidth=2)
    ax[3].set_xlabel('Distance')
    ax[3].set_ylabel('DRS')

    ax[0].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[0].set_title(f'{d1} FASTEST LAP IN {session.event.EventName}', font='Fira Sans', fontsize=18)
    ax[1].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[2].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

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
        for i in session.laps.pick_driver(driver_1).pick_lap(15).iterlaps():
            d1_lap = i[1]

        d2_lap = session.laps.pick_driver(driver_2).pick_fastest()

    else:
        # d1_lap = session.laps.pick_driver('HAM')
        # count = 1
        # for i in d1_lap.iterlaps():
        #     if count == 20:
        #         d1_lap = i[1]
        #     elif count == 19:
        #         d2_lap = i[1]
        #     count += 1
        # # # d2_lap = session.laps.split_qualifying_sessions()[2].pick_driver('NOR').pick_quicklaps()
        d1_lap = session.laps.pick_driver(driver_1).pick_fastest()
        d2_lap = session.laps.pick_driver(driver_2).pick_fastest()

    from fastf1.utils import delta_time as delta_improved
    delta_time, ref_tel, compare_tel = delta_improved(d1_lap, d2_lap)

    final_value = ((d2_lap['LapTime'] - d1_lap['LapTime']).total_seconds())

    def adjust_to_final(series, final_value):
        diff = final_value - series.iloc[-1]
        adjustments = [diff * (i + 1) / len(series) for i in range(len(series))]
        adjusted_series = series + adjustments
        return adjusted_series

    delta_time = adjust_to_final(delta_time, final_value)
    fig, ax = plt.subplots(nrows=3, figsize=(9, 8), gridspec_kw={'height_ratios': [4, 1, 1]}, dpi=150)

    ax[0].plot(ref_tel['Distance'], ref_tel['Speed'],
               color='#3399FF',
               label=driver_1, linewidth=3)
    ax[0].plot(compare_tel['Distance'], compare_tel['Speed'],
               color='#FFA500',
               label=driver_2, linewidth=3)

    colors = ['green' if x > 0 else 'red' for x in delta_time]
    twin = ax[0].twinx()
    for i in range(1, len(delta_time)):
        twin.plot(ref_tel['Distance'][i - 1:i + 1], delta_time[i - 1:i + 1], color=colors[i],
                  alpha=0.6, label='delta', linewidth=2.75)

    # Set the labels for the axes
    ax[0].set_xlabel('Distance')
    ax[0].set_ylabel('Speed')
    ax[0].set_title(f'{str(session.date.year) + " " + session.event.EventName + " " + session.name}'
                    f'{" Lap " + str(lap) if lap is not None else ""} comparison: {driver_1} VS {driver_2}',
                    font='Fira Sans', fontsize=20, y=1.1)

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
    plt.figtext(0.01, 0.02, '@F1BigData', fontsize=15, color='gray', alpha=0.5)

    ax[1].plot(ref_tel['Distance'], ref_tel['Brake'],
               color='#3399FF',
               label=driver_1, linewidth=2.75)
    ax[1].plot(compare_tel['Distance'], compare_tel['Brake'],
               color='#FFA500',
               label=driver_2, linewidth=2.75)

    ax[1].set_xlabel('Distance')
    ax[1].set_ylabel('Brakes')

    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels(['OFF', 'ON'])

    ax[2].plot(ref_tel['Distance'], ref_tel['Throttle'],
               color='#3399FF',
               label=driver_1, linewidth=2.75)
    ax[2].plot(compare_tel['Distance'], compare_tel['Throttle'],
               color='#FFA500',
               label=driver_2, linewidth=2.75)

    ax[2].set_xlabel('Distance')
    ax[2].set_ylabel('Throttle')

    ax[2].set_yticks([0, 50, 100])
    ax[2].set_yticklabels(['0%', '50%', '100%'])

    ax[1].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[2].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

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
    lc_comp.set_linewidth(8)

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


def fastest_by_point(session, team_1, team_2, scope='Team', lap=None):
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
        if lap is not None:
            for i in session.laps.pick_driver(team_1).pick_lap(lap).iterlaps():
                lap_team_1 = i[1]

            for i in session.laps.pick_driver(team_2).pick_lap(lap).iterlaps():
                lap_team_2 = i[1]
            tel_team_1 = lap_team_1.telemetry
        else:
            lap_team_1 = session.laps.pick_driver(team_1).pick_fastest()
            tel_team_1 = lap_team_1.telemetry
            lap_team_2 = session.laps.pick_driver(team_2).pick_fastest()
    from fastf1.utils import delta_time as delta_improved
    delta_time, ref_tel, compare_tel = delta_improved(lap_team_1, lap_team_2)
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

    def smooth_data(segments, window_size=7):
        smoothed_data = np.copy(segments)
        half_window = window_size // 2

        for i in range(segments.shape[1]):
            for j in range(segments.shape[0]):
                start = max(j - half_window, 0)
                end = min(j + half_window + 1, segments.shape[0])
                smoothed_data[j, i] = np.mean(segments[start:end, i], axis=0)

        return smoothed_data

    segments = smooth_data(segments, window_size=7)
    segments = rotate(segments, session.get_circuit_info().rotation / 180 * np.pi)
    plt.subplots(figsize=(8, 7))
    for i, segment in enumerate(segments):
        if i > 4:
            seg_x = [segment[0][0], segment[1][0]]
            seg_y = [segment[0][1], segment[1][1]]
            plt.plot(seg_x, seg_y, color=cmap(norm(delta_time[i])), linewidth=5)

    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    plot_turns(session.get_circuit_info(), session.get_circuit_info().rotation / 180 * np.pi, plt)

    legend_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=4)]

    plt.legend(legend_lines, [f'{team_1} ahead', f'{team_2} ahead'], fontsize='x-large')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='vertical')
    cbar.set_label('Delta Time', fontsize='x-large')
    cbar.ax.tick_params(labelsize=12)
    for label in cbar.ax.get_yticklabels():
        label.set_size(18)
        label.set_family('Fira Sans')

    plt.suptitle(f"{team_1} vs {team_2}:"
                 f" {str(session.session_info['StartDate'].year)} "
                 f"{session.event.EventName.replace('Grand Prix', 'GP') + ' ' + session.name} \n",
                 font='Fira Sans', fontsize=24)
    plt.tight_layout()
    path = (f"../PNGs/Dif by point {team_1} vs {team_2} - {str(session.session_info['StartDate'].year)}"
            f" {session.event.EventName + ' ' + session.name}.png")
    plt.savefig(path, dpi=450)
    plt.show()


def track_dominance(session, team_1, team_2, mode='T'):
    """
       Plot the track dominance of 2 teams in their fastest laps

       Parameters:
       session(Session): Session to analyze
       team_1 (str): Team 1
       team_2 (str): Team 2
    """
    if mode == 'T':
        lap_team_1 = session.laps.pick_team(team_1).pick_fastest()
        lap_team_2 = session.laps.pick_team(team_2).pick_fastest()
    else:
        lap_team_1 = session.laps.pick_driver(team_1).pick_fastest()
        lap_team_2 = session.laps.pick_driver(team_2).pick_fastest()
    tel_team_1 = lap_team_1.telemetry
    delta_time, ref_tel, compare_tel = utils.delta_time(lap_team_1, lap_team_2)
    final_value = ((lap_team_2['LapTime'] - lap_team_1['LapTime']).total_seconds())
    year = session.date.year

    def adjust_to_final(series, final_value):
        diff = final_value - series.iloc[-1]
        adjustments = [diff * (i + 1) / len(series) for i in range(len(series))]
        adjusted_series = series + adjustments
        return adjusted_series

    delta_time = adjust_to_final(delta_time, final_value)
    delta_time = delta_time.reset_index(drop=True)
    num_stretches = 25
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
    track = np.concatenate([points[:-1], points[1:]], axis=1)

    def smooth_data(segments, window_size=7):
        smoothed_data = np.copy(segments)
        half_window = window_size // 2

        for i in range(segments.shape[1]):
            for j in range(segments.shape[0]):
                start = max(j - half_window, 0)
                end = min(j + half_window + 1, segments.shape[0])
                smoothed_data[j, i] = np.mean(segments[start:end, i], axis=0)

        return smoothed_data

    if mode == 'T':
        color_t1 = team_colors.get(year).get(team_1)
        color_t2 = team_colors.get(year).get(team_2)
    else:
        color_t1 = 'red'
        color_t2 = 'blue'
    segments = smooth_data(track, window_size=7)
    segments = rotate(segments, session.get_circuit_info().rotation / 180 * np.pi)
    colors = [color_t1 if value == 0 else color_t2 for value in delta_time_team]
    plt.subplots(figsize=(8, 7))
    for i, segment in enumerate(segments):
        seg_x = [segment[0][0], segment[1][0]]
        seg_y = [segment[0][1], segment[1][1]]
        plt.plot(seg_x, seg_y, color=colors[i], linewidth=5)

    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    plot_turns(session.get_circuit_info(), session.get_circuit_info().rotation / 180 * np.pi, plt)
    legend_lines = [Line2D([0], [0], color=color_t1, lw=4),
                    Line2D([0], [0], color=color_t2, lw=4)]

    plt.legend(legend_lines, [f'{team_1} faster', f'{team_2} faster'], loc='lower left', fontsize='x-large')

    plt.suptitle(f"{team_1} vs {team_2}:"
                 f" {session.session_info['StartDate'].year} {session.event.EventName}", font='Fira Sans',
                 fontsize=18)
    plt.tight_layout()
    path = (f"../PNGs/TRACK DOMINANCE{team_1} vs {team_2} - {str(session.session_info['StartDate'].year)}"
            f" {session.event.EventName}.png")
    plt.savefig(path, dpi=400)
    plt.show()


def get_fastest_data(session, fastest_lap=True):
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
    drivers.remove('SAI')
    drivers.remove('ALB')
    drivers.remove('LEC')
    drivers.remove('SAR')
    drivers.remove('PER')
    year = session.date.year
    circuit_speed = {}
    colors_dict = {}
    for driver in drivers:
        try:
            d_laps = session.laps.pick_driver(driver)
            if fastest_lap:
                d_laps = session.laps.pick_driver(driver).pick_fastest()
            if len(d_laps) > 0:
                top_speed = max(d_laps.telemetry['Speed'])
                if fastest_lap:
                    team = d_laps['Team']
                    column = 'Top Speeds (only fastest lap)'
                else:
                    team = d_laps['Team'].values[0]
                    column = 'Top Speeds'
                if top_speed != 0:
                    circuit_speed[driver] = top_speed
                    colors_dict[driver] = team
        except (KeyError, ValueError):
            print(f'No data for {driver}')
        print(circuit_speed)

    x_fix = 5
    y_fix = 0.25
    annotate_fontsize = 14
    y_offset_rounded = 0.035
    round_decimals = 0

    circuit_speed = {k: v for k, v in sorted(circuit_speed.items(), key=lambda item: item[1], reverse=True)}
    fig, ax1 = plt.subplots(figsize=(8, 6.5), dpi=175)

    colors = []
    for i in range(len(circuit_speed)):
        colors.append(team_colors[year][colors_dict[list(circuit_speed.keys())[i]]])

    bars = ax1.bar(list(circuit_speed.keys()), list(circuit_speed.values()), color=colors,
                   edgecolor='white')

    round_bars(bars, ax1, colors, y_offset_rounded=y_offset_rounded)
    annotate_bars(bars, ax1, y_fix, annotate_fontsize, text_annotate='default', ceil_values=False, round=round_decimals)

    ax1.set_title(f'{column} in {str(session.event.year) + " " + session.event.Location + " " + session.name}',
                  font='Fira Sans', fontsize=16)

    max_value = max(circuit_speed.values())
    ax1.set_ylim(min(circuit_speed.values()) - x_fix, max_value + x_fix)
    plt.figtext(0.725, 0.88, '@F1BigData', fontsize=15, color='gray', alpha=0.5)

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


def drs_efficiency(session):
    teams = set(session.laps['Team'])
    drs_data = pd.DataFrame(columns=['Team', 'DRS_Zone', 'Efficiency'])
    year = session.event.year
    for t in teams:
        fastest_lap = session.laps.pick_team(t).pick_fastest()
        telemetry = fastest_lap.telemetry.reset_index(drop=True)
        telemetry['DRS'] = telemetry['DRS'].replace(14, 12)
        telemetry['DRS'] = telemetry['DRS'].replace(10, 8)
        drs_zones = telemetry['DRS'].diff().fillna(0)
        check_zones = drs_zones[drs_zones != 0].values
        if check_zones[0] == -4 and check_zones[1] == 4 and len(check_zones) == 2:
            driver = fastest_lap['Driver']
            lap_number = fastest_lap['LapNumber'] - 1
            prev_lap = session.laps.pick_driver(driver).pick_lap(lap_number)
            prev_telemetry = prev_lap.telemetry.reset_index(drop=True)
            prev_telemetry['DRS'] = prev_telemetry['DRS'].replace(14, 12)
            prev_telemetry['DRS'] = prev_telemetry['DRS'].replace(10, 8)
            prev_drs_zones = prev_telemetry['DRS'].diff().fillna(0)
            drs_zones = pd.concat([prev_drs_zones, drs_zones]).reset_index(drop=True)
            telemetry = pd.concat([prev_telemetry, telemetry]).reset_index(drop=True)
        sequences_indices = []
        current_sequence_indices = []
        in_sequence = False

        for index, value in drs_zones.items():
            if value == 4:
                if in_sequence:
                    sequences_indices.append(current_sequence_indices)
                in_sequence = True
                current_sequence_indices = [index]

            elif value == -4 and in_sequence:
                current_sequence_indices.append(index)
                sequences_indices.append(current_sequence_indices)
                in_sequence = False

            elif in_sequence:
                current_sequence_indices.append(index)
        count = 1
        for d_zone in sequences_indices:
            first_point = d_zone[0] - 1
            start_speed = telemetry['Speed'][first_point]
            top_speed = max(telemetry['Speed'][d_zone])
            print(f'{t} - From {start_speed} to {top_speed} ({top_speed - start_speed})')
            percentage_difference = ((top_speed - start_speed) / start_speed) * 100
            current_drs_zone = pd.DataFrame([[t, count, percentage_difference]], columns=drs_data.columns)
            drs_data = pd.concat([drs_data, current_drs_zone], ignore_index=True)

    print_data = drs_data.groupby('Team')['Efficiency'].mean().reset_index().sort_values('Efficiency',
                                                                                         ascending=False).reset_index(
        drop=True)

    colors = []
    for index, row in print_data.iterrows():
        print(f'{index + 1} - {row[0]}: {row[1]:.2f}%')
        colors.append(team_colors.get(year)[row[0]])

    fix, ax = plt.subplots(figsize=(9, 8))
    bars = plt.bar(print_data['Team'], print_data['Efficiency'])
    round_bars(bars, ax, colors, color_1=None, color_2=None, y_offset_rounded=0.03, corner_radius=0.1, linewidth=4)
    annotate_bars(bars, ax, 0.1, 14, text_annotate='default', ceil_values=False, round=2,
                  y_negative_offset=0.04, annotate_zero=False, negative_offset=0, add_character='%')

    plt.title(f'DRS EFFICIENCY IN {session.event.year} {str(session.event.Country).upper()} '
              f'{str(session.name).upper()} SEASON', font='Fira Sans', fontsize=20)
    plt.xlabel('Team', font='Fira Sans', fontsize=20)
    plt.ylabel('Drs efficiency (%)', font='Fira Sans', fontsize=20)
    plt.xticks(rotation=90, font='Fira Sans', fontsize=18)
    plt.yticks(font='Fira Sans', fontsize=16)
    plt.ylim(bottom=min(print_data['Efficiency']) - 1, top=max(print_data['Efficiency']) + 1)
    color_index = 0
    for label in ax.get_xticklabels():
        label.set_color('white')
        label.set_fontsize(16)
        for_color = colors[color_index]
        if for_color == '#ffffff':
            for_color = '#FF7C7C'
        label.set_path_effects([path_effects.withStroke(linewidth=2, foreground=for_color)])
        color_index += 1
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'../PNGs/DRS EFFICIENCY {year}.png', dpi=450)
    plt.show()

    print("""
    This is done by taking the speed just before opening the DRS and the top speed reached during each DRS zone. 
    Lap corresponds to the fastest lap of each team.
    """)


def get_fastest_sectors(session, average=False):
    """
        Get the fastest data in a session

        Parameters:
        session (Session): Session to be analyzed
   """

    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    drivers = session.laps['Driver'].groupby(session.laps['Driver']).size()
    drivers = drivers.reset_index(name='Count')['Driver'].to_list()
    # drivers.remove('ZHO')
    year = session.date.year
    sector_times = pd.DataFrame(columns=['Sector1Time', 'Sector2Time', 'Sector3Time', 'Color'])
    for driver in drivers:
        try:
            d_laps = session.laps.pick_driver(driver)
            for sector in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
                if len(d_laps) > 0:
                    if average:
                        d_laps = d_laps.pick_quicklaps().pick_wo_box()
                        time = round(np.mean(d_laps[sector].dropna()).total_seconds(), 3)
                        method = 'AVERAGE'
                    else:
                        time = round(min(d_laps[sector].dropna()).total_seconds(), 3)
                        method = 'FASTEST'
                    team = d_laps['Team'].values[0]
                    sector_times.loc[driver, sector] = time
                    sector_times.loc[driver, f'Color'] = team_colors[year][team]
        except (KeyError, IndexError):
            print(f'No data for {driver}')

    fig, ax = plt.subplots(ncols=3, figsize=(8, 8), dpi=150)
    fig.suptitle(f'{method} SECTOR TIMES IN {session.event.year} {str(session.event.Country).upper()} '
                 f'{str(session.name).upper()}',
                 font='Fira Sans', fontsize=20, fontweight='bold')
    for i in range(3):
        sector_time = f'Sector{i + 1}Time'
        colors = sector_times.sort_values(by=f'Sector{i + 1}Time', ascending=True)['Color'].values
        create_rounded_barh(ax[i], sector_times, sector_time, 'Color')
        ax[i].set_title(f'Sector {i + 1}', font='Fira Sans', fontsize=18, y=0.99)
        ax[i].invert_yaxis()
        color_index = 0
        for label in ax[i].get_yticklabels():
            label.set_color('white')
            label.set_fontsize(14)
            for_color = colors[color_index]
            if for_color == '#ffffff':
                for_color = '#FF7C7C'
            label.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground=for_color)])
            color_index += 1
        ax[i].yaxis.grid(False)
        ax[i].xaxis.grid(False)
        ax[i].set_xticklabels([])
        ax[i].tick_params(axis='x', length=0)
    plt.tight_layout()
    plt.savefig(f'../PNGs/SECTOR TIMES IN {session.event.year} {str(session.event.Country).upper()} '
                f'{str(session.name).upper()}',
                dpi=450)
    plt.show()


def all_laps_driver_session(session, driver):
    laps = session.laps.pick_driver(driver)
    for l in laps.iterlaps():
        current_lap = l[1]
        tyre_life = current_lap['TyreLife']
        in_lap = ' --- IN' if not pd.isna(current_lap['PitInTime']) else ''
        tyre = current_lap['Compound']
        if not pd.isna(current_lap['LapTime']):
            lap_time = current_lap['LapTime']
            lap_time_f = f"{lap_time.seconds // 60:02d}:{lap_time.seconds % 60:02d}.{int(lap_time.microseconds / 1000):03d}"
            try:
                print(f'{tyre[0]} - ({int(tyre_life)}): {lap_time_f} - {max(current_lap.get_telemetry()["Speed"])}')
            except:
                print('NO VALID LAP')
            if in_lap != '':
                print('----------- BOX -----------')


def race_simulation_test_day(session):
    plotting.setup_mpl(misc_mpl_mods=False)
    drivers = list(session.laps['Driver'].unique())
    driver_stints = {}
    for d in drivers:
        d_laps = session.laps.pick_driver(d)
        max_stint = d_laps[d_laps['Compound'].isin(['HARD', 'MEDIUM', 'SOFT'])]['Stint'].value_counts()
        max_stint = max_stint.reset_index()
        max_stint.columns = ['Stint', 'Counts']
        max_stint = max_stint.sort_values(by=['Counts', 'Stint'], ascending=[False, False])
        stint_index = 0
        driver_laps_dict = {}

        for i in range(len(max_stint)):
            stint_number = max_stint.iloc[stint_index]['Stint']
            driver_laps_filter = d_laps[d_laps['Stint'] == stint_number].pick_wo_box().pick_quicklaps()
            driver_laps_dict[stint_number] = len(driver_laps_filter)
            stint_index += 1

        driver_laps_dict = dict(sorted(driver_laps_dict.items(), key=lambda item: (-int(item[1]), -int(item[0]))))
        keys = list(driver_laps_dict)[:3]
        first_three_values = [item[1] for item in list(driver_laps_dict.items())[:3]]
        if len(first_three_values) == 3 and all(value > 7 for value in first_three_values):
            driver_stints[d] = keys
        else:
            print(f'{d} NOT VALID')

    average_race_simulation = {}
    for d in list(driver_stints):
        d_laps = session.laps.pick_driver(d)
        d_laps = pd.DataFrame(d_laps[d_laps['Stint'].isin(driver_stints[d])].pick_wo_box().pick_quicklaps())
        d_laps['LapStartDate'] = pd.to_datetime(d_laps['LapStartDate'])
        d_laps['MinLapStartTimePerStint'] = d_laps.groupby('Stint')['LapStartDate'].transform('min')
        d_laps['MinLapStartTimePerStint'] = d_laps['MinLapStartTimePerStint'].dt.strftime('%H:%M')
        stints = pd.DataFrame(d_laps.groupby(['Stint', 'Compound', 'MinLapStartTimePerStint']).size())
        print(f'-- DATA FOR {d}')
        for index, row in stints.iterrows():
            print(f'{index[1]}: {row[0]} - {index[2]}')
        average_race_simulation[d] = d_laps['LapTime'].mean()

    print('------------------------')

    average_race_simulation = dict(sorted(average_race_simulation.items(), key=lambda item: (item[1].total_seconds())))
    prev_time = None
    for d, t in average_race_simulation.items():
        if prev_time is None:
            time_diff = ''
            prev_time = t
        else:
            time_diff = t - prev_time
            time_diff = f'(+{format_timedelta(time_diff)}s)'.replace('00:0', '')
        print(f'{d}: {format_timedelta(t)} {time_diff}')


def track_limits(session):
    messages = session.race_control_messages[session.race_control_messages['Message'].str.contains('TRACK LIMITS AT')]
    messages['Driver'] = messages['Message'].str.extract(r'\((\w+)\)')
    messages['Turn'] = messages['Message'].str.extract(r'TURN (\d+)').astype(int)
    messages['Lap'] = messages['Message'].str.extract(r'LAP (\d+)').astype(int)
    messages = messages.sort_values(by=['Driver', 'Lap'])
    final_output = messages.apply(lambda x: f"{x['Driver']}: LAP {x['Lap']} TURN {x['Turn']}", axis=1).tolist()
    for line in final_output:
        print(line)


def track_temps(session):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    time = session.weather_data['Time']
    track_temp = session.weather_data['TrackTemp']
    plt.plot(time, track_temp, color='orange', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.title('Track Temperature Over Time', font='Fira Sans', fontsize=24)
    plt.xlabel('Time', font='Fira Sans', fontsize=18)
    plt.ylabel('Temperature (ºC)', font='Fira Sans', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'../PNGs/Track Temperature Over Time {session.event.year} {str(session.event.Country).upper()} '
                f'{str(session.name).upper()}',
                dpi=450)
    plt.show()
