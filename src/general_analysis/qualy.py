import fastf1
import numpy as np
import pandas as pd
from fastf1.core import Laps
from matplotlib import pyplot as plt, cm
from fastf1 import utils, plotting
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from scipy.signal import savgol_filter
from timple.timedelta import strftimedelta
from scipy.interpolate import interp1d, CubicSpline
from scipy.interpolate import UnivariateSpline


def overlying_laps(session, driver_1, driver_2, driver_3=None):
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['figure.facecolor'] = 'black'

    # Set the color of text, labels, and ticks to white
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'

    d1_lap = session.laps.pick_driver(driver_1).pick_fastest()
    d2_lap = session.laps.pick_driver(driver_2).pick_fastest()

    delta_time, ref_tel, compare_tel = utils.delta_time(d1_lap, d2_lap)

    final_value = ((d2_lap['LapTime'] - d1_lap['LapTime']).total_seconds())

    def adjust_to_final(series, final_value):
        # Calculate the adjustment required for each element
        diff = final_value - series.iloc[-1]
        adjustments = [diff * (i + 1) / len(series) for i in range(len(series))]

        # Adjust the original series
        adjusted_series = series + adjustments

        return adjusted_series

    delta_time = adjust_to_final(delta_time, final_value)

    if driver_3 is not None:
        d3_lap = session.laps.pick_driver(driver_3).pick_fastest()
        delta_time2, _, compare_tel2 = utils.delta_time(d1_lap, d3_lap)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the speed for driver 1 (reference) and driver 2 (compare)
    ax.plot(ref_tel['Distance'], ref_tel['Speed'],
            color='#0000FF',
            label=driver_1)
    ax.plot(compare_tel['Distance'], compare_tel['Speed'],
            color='#FFA500',
            label=driver_2)

    if driver_3 is not None:
        ax.plot(compare_tel2['Distance'], compare_tel2['Speed'],
                color='#00FF00',  # or any other color
                label=driver_3)

    colors = ['green' if x > 0 else 'red' for x in delta_time]
    twin = ax.twinx()
    for i in range(1, len(delta_time)):
        twin.plot(ref_tel['Distance'][i-1:i+1], delta_time[i-1:i+1], color=colors[i],
                  alpha=0.5, label='delta')
    if driver_3 is not None:
        twin.plot(ref_tel['Distance'], delta_time2, ':', color='yellow', alpha=0.5, label=f'delta_{driver_3}')
    twin.axhline(y=0, color='white', linestyle='--')

    # Set the labels for the axes
    ax.set_xlabel('Distance')
    ax.set_ylabel('Speed')
    if driver_3 is not None:
        ax.set_title(f'Qualy lap comparison among {driver_1}, {driver_2}, and {driver_3}')
    else:
        ax.set_title(f'Qualy lap comparison: {driver_1} VS {driver_2}')

    # Get the legend handles and labels from the first axes
    handles1, labels1 = ax.get_legend_handles_labels()

    # Get the legend handles and labels from the second (twin) axes
    handles2, labels2 = twin.get_legend_handles_labels()

    # Combine the handles and labels
    handles = handles1 + handles2
    labels = labels1 + labels2

    ax.legend(handles, set(labels), loc='lower right')

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"../PNGs/{driver_1} - {driver_2} QUALY LAPS {session.event.OfficialEventName}.png", dpi=1000)
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


def interpolate_data(reference_t, target_t, target_data):
    cs = CubicSpline(target_t, target_data, extrapolate=True)
    return cs(reference_t)


def resample_telemetry(distance, data, resample_interval=10):
    """
    Resample telemetry data at regular distance intervals using cubic spline interpolation.
    """
    cs = CubicSpline(distance, data)
    max_dist = distance[-1]
    resampled_distance = np.arange(0, max_dist, resample_interval)
    resampled_data = cs(resampled_distance)
    return resampled_distance, resampled_data


def apply_smoothing(data, window_length=51, polyorder=2):
    """
    Apply Savitzky-Golay filter to the data for smoothness.
    If the length of the data is less than the window_length, we don't apply the filter.
    """
    if len(data) < window_length:
        return data

    return savgol_filter(data, window_length, polyorder)

def fastest_by_point(session, team_1, team_2):
    lap_team_1 = session.laps.pick_team(team_1).pick_fastest()
    tel_team_1 = lap_team_1.get_telemetry()

    lap_team_2 = session.laps.pick_team(team_2).pick_fastest()
    tel_team_2 = lap_team_2.get_telemetry()

    x_1 = np.array(tel_team_1['X'].values)
    y_1 = np.array(tel_team_1['Y'].values)
    speed_1 = np.array(tel_team_1['Speed'].values)

    x_2 = np.array(tel_team_2['X'].values)
    y_2 = np.array(tel_team_2['Y'].values)
    speed_2 = np.array(tel_team_2['Speed'].values)

    distance_1 = np.array(tel_team_1['Distance'].values)
    distance_2 = np.array(tel_team_2['Distance'].values)

    resample_interval = 8 # This represents the distance between data points (e.g., every 10 meters)

    # Resample telemetry data at regular distance intervals.
    distance_1_resampled, x_1_resampled = resample_telemetry(distance_1, x_1, resample_interval)
    _, y_1_resampled = resample_telemetry(distance_1, y_1, resample_interval)
    _, speed_1_resampled = resample_telemetry(distance_1, speed_1, resample_interval)

    distance_2_resampled, x_2_resampled = resample_telemetry(distance_2, x_2, resample_interval)
    _, y_2_resampled = resample_telemetry(distance_2, y_2, resample_interval)
    _, speed_2_resampled = resample_telemetry(distance_2, speed_2, resample_interval)

    x_1_resampled = apply_smoothing(x_1_resampled)
    y_1_resampled = apply_smoothing(y_1_resampled)
    speed_1_resampled = apply_smoothing(speed_1_resampled)

    x_2_resampled = apply_smoothing(x_2_resampled)
    y_2_resampled = apply_smoothing(y_2_resampled)
    speed_2_resampled = apply_smoothing(speed_2_resampled)

    # Take the common distance range for comparison
    max_common_distance = min(distance_1_resampled[-1], distance_2_resampled[-1])
    common_indices_1 = distance_1_resampled <= max_common_distance
    common_indices_2 = distance_2_resampled <= max_common_distance

    speed_1_common = speed_1_resampled[common_indices_1]
    speed_2_common = speed_2_resampled[common_indices_2]

    faster = [team_1 if speed_1_common[i] > speed_2_common[i] else team_2 for i in range(len(speed_1_common))]

    team_colors = {team_1: 0, team_2: 1}
    color_values = [team_colors[team] for team in faster]

    colors = [plotting.team_color(team_1),
              plotting.team_color(team_2)]  # R -> G -> B, Red for team_1 and Blue for team_2
    n_bins = [2]  # Discretizes the interpolation into bins
    cmap_name = 'custom_div_cmap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)

    points = np.array([x_2, y_2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc_comp = LineCollection(segments, cmap=cm)
    lc_comp.set_array(np.array(color_values))
    lc_comp.set_linewidth(4)

    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    title = plt.suptitle(
        f"Fastest Lap Gear Shift Visualization\n"
    )

    cbar = plt.colorbar(mappable=lc_comp, label="Team")
    cbar.set_ticks([0.25, 0.75])  # Set ticks in the middle of the bins
    cbar.set_ticklabels([team_1, team_2])  # Set tick labels to team names
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
