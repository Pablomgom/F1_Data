import fastf1
import numpy as np
import pandas as pd
from fastf1.core import Laps
from matplotlib import pyplot as plt, cm
from fastf1 import utils, plotting
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.signal import savgol_filter
from timple.timedelta import strftimedelta
from scipy.interpolate import interp1d, CubicSpline
from scipy.interpolate import UnivariateSpline


def overlying_laps(session, driver_1, driver_2, lap=None):
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['figure.facecolor'] = 'black'

    # Set the color of text, labels, and ticks to white
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
        #session.laps.split_qualifying_sessions()[0].pick_driver('ALO')
        d1_lap = session.laps.pick_driver(driver_1).pick_fastest()
        d2_lap = session.laps.pick_driver(driver_2).pick_fastest()

    delta_time, ref_tel, compare_tel = utils.delta_time(d1_lap, d2_lap, lap)

    final_value = ((d2_lap['LapTime'] - d1_lap['LapTime']).total_seconds())

    def adjust_to_final(series, final_value):
        # Calculate the adjustment required for each element
        diff = final_value - series.iloc[-1]
        adjustments = [diff * (i + 1) / len(series) for i in range(len(series))]

        # Adjust the original series
        adjusted_series = series + adjustments

        return adjusted_series

    delta_time = adjust_to_final(delta_time, final_value)

    fig, ax = plt.subplots(nrows=5,figsize=(16, 11), gridspec_kw={'height_ratios': [4,1,1,2,2]})

    if lap is None:
        ax[0].plot(ref_tel['Distance'], ref_tel['Speed'],
                color='#0000FF',
                label=driver_1)
        ax[0].plot(compare_tel['Distance'], compare_tel['Speed'],
                color='#FFA500',
                label=driver_2)

        colors = ['green' if x > 0 else 'red' for x in delta_time]
        twin = ax[0].twinx()
        for i in range(1, len(delta_time)):
            twin.plot(ref_tel['Distance'][i-1:i+1], delta_time[i-1:i+1], color=colors[i],
                      alpha=0.5, label='delta')

    else:
        ax[0].plot(ref_tel['Distance'], ref_tel['Speed'],
                color='#0000FF',
                label=driver_1)
        ax[0].plot(compare_tel['Distance'], compare_tel['Speed'],
                color='#FFA500',
                label=driver_2)

        colors = ['green' if x > 0 else 'red' for x in delta_time]
        twin = ax[0].twinx()
        for i in range(1, len(delta_time)):
            twin.plot(compare_tel['Distance'][i - 1:i + 1], delta_time[i - 1:i + 1], color=colors[i],
                      alpha=0.5, label='delta')

    # Set the labels for the axes
    ax[0].set_xlabel('Distance')
    ax[0].set_ylabel('Speed')
    ax[0].set_title(f'{str(session.date.year) + " " + session.event.EventName + " " + session.name}'
                    f'{" Lap " + str(lap) if lap is not None else ""} comparation: {driver_1} VS {driver_2}',
                    fontsize=24, y=1.1)

    twin.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    twin.set_ylabel('Time diff (s)')

    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label=f'{driver_1} ahead'),
        Line2D([0], [0], color='red', lw=2, label=f'{driver_2} ahead')
    ]

    # Get the legend handles and labels from the first axes
    handles1, labels1 = ax[0].get_legend_handles_labels()

    # Combine the handles and labels
    handles = handles1 + legend_elements
    labels = labels1 + [entry.get_label() for entry in legend_elements]

    # Create a single legend with the handles and labels
    ax[0].legend(handles, labels, loc='lower right')
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
    ax[1].set_yticklabels(['ON', 'OFF'])

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

    ax[3].plot(ref_tel['Distance'], ref_tel['nGear'],
               color='#0000FF',
               label=driver_1)
    ax[3].plot(compare_tel['Distance'], compare_tel['nGear'],
               color='#FFA500',
               label=driver_2)

    ax[3].set_xlabel('Distance')
    ax[3].set_ylabel('Gear')

    ax[4].plot(ref_tel['Distance'], ref_tel['RPM'],
               color='#0000FF',
               label=driver_1)
    ax[4].plot(compare_tel['Distance'], compare_tel['RPM'],
               color='#FFA500',
               label=driver_2)

    ax[4].set_xlabel('Distance')
    ax[4].set_ylabel('RPM')

    ax[3].set_yticks([2, 3, 4, 5, 6, 7, 8])  # Assuming the 'Brakes' data is normalized between 0 and 1
    ax[3].set_yticklabels(['2', '3', '4', '5', '6', '7', '8'])

    ax[1].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[2].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[3].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[4].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f'../PNGs/{driver_1} - {driver_2} + {session.event.EventName + " " + session.name}.png', dpi=400)
    plt.show()


def gear_changes(session, driver):
    lap = session.laps.pick_driver(driver).pick_fastest()
    tel = lap.get_telemetry()

    x = np.array(tel['X'].values)
    y = np.array(tel['Y'].values)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    gear = tel['nGear'].to_numpy().astype(float)

    cmap = cm.get_cmap('Paired')
    lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N + 1), cmap=cmap)
    lc_comp.set_array(gear)
    lc_comp.set_linewidth(4)

    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    title = plt.suptitle(
        f"Fastest Lap Gear Shift Visualization\n"
        f"{lap['Driver']} - {session.event['EventName']} {session.event.year}"
    )

    cbar = plt.colorbar(mappable=lc_comp, label="Gear", boundaries=np.arange(1, 10))
    cbar.set_ticks(np.arange(1.5, 9.5))
    cbar.set_ticklabels(np.arange(1, 9))
    plt.savefig(f"../PNGs/GEAR CHANGES {driver} {session.event.OfficialEventName}", dpi=400)
    plt.show()





def fastest_by_point(session, team_1, team_2):
    lap_team_1 = session.laps.pick_team(team_1).pick_fastest()
    tel_team_1 = lap_team_1.get_telemetry()

    lap_team_2 = session.laps.pick_team(team_2).pick_fastest()
    tel_team_2 = lap_team_2.get_telemetry()

    delta_time, ref_tel, compare_tel = utils.delta_time(lap_team_1, lap_team_2, special_mode=True)

    final_value = ((lap_team_2['LapTime'] - lap_team_1['LapTime']).total_seconds())

    def adjust_to_final(series, final_value):
        # Calculate the adjustment required for each element
        diff = final_value - series.iloc[-1]
        adjustments = [diff * (i + 1) / len(series) for i in range(len(series))]

        # Adjust the original series
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
    # Change the colormap to a diverging colormap
    cmap = cm.get_cmap('coolwarm')

    # Get the maximum absolute value of delta_time for symmetric coloring
    vmax = np.max(np.abs(delta_time))
    vmin = -vmax

    # Initialize the TwoSlopeNorm with 0 as the center
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Update LineCollection with the new colormap and normalization
    lc_comp = LineCollection(segments, norm=norm, cmap=cmap)
    lc_comp.set_array(delta_time)
    lc_comp.set_linewidth(7)

    plt.subplots(figsize=(12,8))

    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    cbar = plt.colorbar(mappable=lc_comp, label="DeltaTime")
    # Create custom legend
    legend_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=4)]

    plt.legend(legend_lines, [f'{team_1} faster', f'{team_2} faster'])
    cbar.set_label('Time Difference (s)', rotation=270, labelpad=20, fontsize=14)

    cbar.ax.tick_params(labelsize=12)

    plt.suptitle(f"{team_1} vs {team_2}:"
                 f" {str(session.session_info['StartDate'].year) + ' ' + session.event.EventName + ' ' + session.name} \n",
                 fontsize=18)
    plt.tight_layout()
    path = (f"../PNGs/Dif by point {team_1} vs {team_2} - {str(session.session_info['StartDate'].year)}"
            f" {session.event.EventName + ' ' + session.name}.png")
    plt.savefig(path, dpi=400)
    plt.show()


def qualy_results(session):
    drivers = pd.unique(session.laps['Driver'])
    list_fastest_laps = list()
    for drv in drivers:
        drvs_fastest_lap = session.laps.pick_driver(drv).pick_fastest()
        list_fastest_laps.append(drvs_fastest_lap)
    fastest_laps = Laps(list_fastest_laps).sort_values(by='LapTime').reset_index(drop=True)

    pole_lap = fastest_laps.pick_fastest()
    fastest_laps['LapTimeDelta'] = fastest_laps['LapTime'] - pole_lap['LapTime']
    fastest_laps.dropna(how='all', inplace=True)

    team_colors = list()
    for index, lap in fastest_laps.iterlaps():
        if lap['Team'] == 'Sauber':
            color = '#FD8484'
        else:
            color = plotting.team_color(lap['Team'])
        team_colors.append(color)

    fig, ax = plt.subplots()
    ax.barh(fastest_laps.index, fastest_laps['LapTimeDelta'],
            color=team_colors, edgecolor='grey')

    ax.set_yticks(fastest_laps.index)
    ax.set_yticklabels(fastest_laps['Driver'])

    # show fastest at the top
    ax.invert_yaxis()

    # draw vertical lines behind the bars
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)

    lap_time_string = strftimedelta(pole_lap['LapTime'], '%m:%s.%ms')

    plt.suptitle(f"{session.event['EventName']} {session.event.year} Qualifying\n"
                 f"Fastest Lap: {lap_time_string} ({pole_lap['Driver']})")

    plt.xlabel("Seconds")
    plt.ylabel("Driver")

    def custom_formatter(x, pos):
        return x / 1e9

    ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))

    # Adding a watermark at the bottom left of the figure
    plt.figtext(0.01, 0.02, '@Big_Data_Master', fontsize=15, color='gray', alpha=0.5)

    plt.savefig(f"../PNGs/QUALY OVERVIEW {session.event.OfficialEventName}.png", dpi=400)
    plt.show()


def qualy_diff(team_1, team_2, session):
    qualys = []
    session_names = []

    for i in range(session):
        session = fastf1.get_session(2023, i + 1, 'Q')
        session.load(telemetry=True)
        qualys.append(session)
        session_names.append(session.event['Location'].split('-')[0])

    delta_laps = []

    for qualy in qualys:

        qualy_result = qualy.results['Q3'][:10].append(qualy.results['Q2'][10:15]).append(qualy.results['Q1'][15:20])
        team_1_lap = qualy.laps.pick_team(team_1).pick_fastest()['LapTime']
        index = qualy_result[qualy_result == team_1_lap].index
        if len(index) == 0:
            index_q2 = qualy.results['Q2'][qualy.results['Q2']
                                           == qualy.laps.pick_team(team_1).pick_fastest()['LapTime']].index
            q_session_team_1 = 'Q2'

            if len(index_q2) == 0:
                q_session_team_1 = 'Q1'
        else:
            position = qualy_result.index.get_loc(qualy_result[qualy_result == team_1_lap].index[0])
            if position < 10:
                q_session_team_1 = 'Q3'
            elif 10 <= position < 15:
                q_session_team_1 = 'Q2'
            else:
                q_session_team_1 = 'Q1'

        team_2_lap = qualy.laps.pick_team(team_2).pick_fastest()['LapTime']
        index = qualy_result[qualy_result == team_2_lap].index
        if len(index) == 0:
            index_q2 = qualy.results['Q2'][qualy.results['Q2']
                                           == qualy.laps.pick_team(team_1).pick_fastest()['LapTime']].index
            q_session_team_2 = 'Q2'

            if len(index_q2) == 0:
                q_session_team_2 = 'Q1'

        else:
            position = qualy_result.index.get_loc(qualy_result[qualy_result == team_2_lap].index[0])
            if position < 10:
                q_session_team_2 = 'Q3'
            elif 10 <= position < 15:
                q_session_team_2 = 'Q2'
            else:
                q_session_team_2 = 'Q1'

        session = min(q_session_team_1, q_session_team_2)

        team_numbers_team_1 = qualy.results[qualy.results['TeamName'] == team_1].index
        team_numbers_team_2 = qualy.results[qualy.results['TeamName'] == team_2].index
        team_1_lap = qualy.results[session].loc[team_numbers_team_1].min().total_seconds() * 1000
        team_2_lap = qualy.results[session].loc[team_numbers_team_2].min().total_seconds() * 1000

        percentage_diff = (team_2_lap - team_1_lap) / team_1_lap * 100

        delta_laps.append(percentage_diff)

    plt.figure(figsize=(13, 7))

    for i in range(len(session_names)):
        color = plotting.team_color(team_1) if delta_laps[i] > 0 else plotting.team_color(team_2)
        label = f'{team_1} faster' if delta_laps[i] > 0 else f'{team_2} faster'
        plt.bar(session_names[i], delta_laps[i], color=color, label=label)

    mean_y = np.mean(delta_laps)
    # Draw horizontal line at y=mean_y
    plt.axhline(mean_y, color='red', linewidth=2, label='Mean distance')
    if min(delta_laps) < 0:
        plt.axhline(0, color='black', linewidth=2)

    # Add exact numbers above or below every bar based on whether it's a maximum or minimum
    for i in range(len(session_names)):
        if delta_laps[i] > 0:  # If the bar is above y=0
            plt.text(session_names[i], delta_laps[i] + 0.1, "{:.2f} %".format(delta_laps[i]),
                     ha='center', va='top')
        else:  # If the bar is below y=0
            plt.text(session_names[i], delta_laps[i] - 0.08, "{:.2f} %".format(delta_laps[i]),
                     ha='center', va='bottom')

    # Set the labels and title
    plt.ylabel(f'Percentage time difference', fontsize=14)
    plt.xlabel('Circuito', fontsize=14)
    plt.title(f'{team_1} VS {team_2} time difference', fontsize=14)

    step = 0.2

    start = np.ceil(abs(min(delta_laps) / step))
    if min(delta_laps) < 0:
        start = np.floor(min(delta_laps) / step) * step
    else:
        start = np.ceil(min(delta_laps) / step) * step
    end = np.ceil(max(delta_laps) / step) * step

    # Generate a list of ticks from minimum to maximum y values considering 0.0 value and step=0.2
    yticks = list(np.arange(start, end + step, step))
    yticks.append(mean_y)
    delete = None
    for i in range(len(yticks)):
        if abs(yticks[i] - mean_y) < 0.075:
            delete = yticks[i]
            break
    if delete is not None:
        yticks.remove(delete)
    yticks = sorted(yticks)

    plt.yticks(yticks, [f'{tick:.2f} %' if tick != mean_y else f'{tick:.2f} %' for tick in yticks])

    # To avoid repeating labels in the legend, we handle them separately
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.savefig(f"../PNGs/{team_2} VS {team_1} time difference.png",
                dpi=400)

    # Show the plot
    plt.show()




def fastest_by_point_v2(session, team_1, team_2):
    lap_team_1 = session.laps.pick_team(team_1).pick_fastest()
    tel_team_1 = lap_team_1.get_telemetry()

    lap_team_2 = session.laps.pick_team(team_2).pick_fastest()
    tel_team_2 = lap_team_2.get_telemetry()

    delta_time, ref_tel, compare_tel = utils.delta_time(lap_team_1, lap_team_2, special_mode=True)

    final_value = ((lap_team_2['LapTime'] - lap_team_1['LapTime']).total_seconds())

    def adjust_to_final(series, final_value):
        # Calculate the adjustment required for each element
        diff = final_value - series.iloc[-1]
        adjustments = [diff * (i + 1) / len(series) for i in range(len(series))]

        # Adjust the original series
        adjusted_series = series + adjustments

        return adjusted_series

    delta_time = adjust_to_final(delta_time, final_value)

    num_stretches = 30
    stretch_len = len(delta_time) // num_stretches
    stretched_delta_time = []

    for i in range(num_stretches):
        start_idx = i * stretch_len + delta_time.index[0]
        end_idx = (i + 1) * stretch_len - 1 + delta_time.index[0]  # -1 to get the last element of the stretch

        start_value = delta_time[start_idx]
        end_value = delta_time[end_idx]

        stretch_value = 1 if start_value < end_value else 0  # 1 for team_1, 0 for team_2

        stretched_delta_time.extend([stretch_value] * stretch_len)

    # Handle remaining elements if delta_time is not exactly divisible by num_stretches
    if len(delta_time) % num_stretches != 0:
        start_value = delta_time[num_stretches * stretch_len]
        end_value = delta_time.iloc[-1]  # last value in delta_time

        stretch_value = 1 if start_value < end_value else 0  # 1 for team_1, 0 for team_2

        stretched_delta_time.extend([stretch_value] * (len(delta_time) - num_stretches * stretch_len))

    # Replace your original delta_time with the new stretched_delta_time
    delta_time = np.array(stretched_delta_time)

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
    # Change the colormap to a diverging colormap
    cmap = cm.get_cmap('coolwarm', 2)  # Discrete colormap with 2 colors
    # Initialize the TwoSlopeNorm with 0.5 as the center

    # Update LineCollection with the new colormap and normalization
    lc_comp = LineCollection(segments, cmap=cmap)
    lc_comp.set_array(delta_time)
    lc_comp.set_linewidth(7)

    plt.subplots(figsize=(12,8))

    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    # Create custom legend
    legend_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=4)]

    plt.legend(legend_lines, [f'{team_1} faster', f'{team_2} faster'])

    plt.suptitle(f"TRACK DOMINANCE {team_1} vs {team_2}:"
                 f" {str(session.session_info['StartDate'].year) + ' ' + session.event.EventName} \n",
                 fontsize=18)
    plt.tight_layout()
    path = (f"../PNGs/TRACK DOMINANCE{team_1} vs {team_2} - {str(session.session_info['StartDate'].year)}"
            f" {session.event.EventName}.png")
    plt.savefig(path, dpi=400)
    plt.show()
