import inspect

import fastf1
import numpy as np
import matplotlib.colors as mcolors
import nltk
from matplotlib import patches
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

from src.plots.plots import lighten_color

name_count = {}


def restart_name_count():
    name_count = {}


def is_session_first_arg(func):
    signature = inspect.signature(func)
    parameters = list(signature.parameters)
    return parameters[0] == 'session' if parameters else False


def find_similar_func(name, functions):
    func_list = list(functions.keys())

    for func_name in func_list:
        if name in func_name:
            print(help(functions[func_name]))


def parse_args(args_input, function_map, session):
    args = []
    kwargs = {}

    for arg in args_input.split(','):
        arg = arg.strip()
        if '=' in arg:
            key, value = [x.strip() for x in arg.split('=')]
            if value == 'False':
                value = False
            elif value == 'True':
                value = True
            elif value == 'None':
                value = None
            elif value[0] == '[':
                value = value.replace('[', '').replace(']', '')
                list_values = value.split(',')
                value = [val for val in list_values]
            else:
                try:
                    float_value = float(value)
                    if float_value.is_integer():
                        value = int(float_value)
                    else:
                        value = float_value
                except ValueError:
                    pass

            kwargs[key] = value

        else:
            if arg == 'session':
                args.append(session)
            elif arg == 'False':
                args.append(False)
            elif arg == 'True':
                args.append(True)
            elif arg == 'dict':
                args.append(function_map)
            elif arg in function_map:
                args.append(function_map[arg])
            elif arg == 'None':
                args.append(None)
            elif arg != '':
                try:
                    float_arg = float(arg)
                    if float_arg.is_integer():
                        args.append(int(float_arg))
                    else:
                        args.append(float_arg)
                except ValueError:
                    args.append(arg)

    return args, kwargs


def load_session(year, gp, race_type):
    session = fastf1.get_session(year, gp, race_type)
    print(session.api_path)
    session.load()
    return session


def rotate(xy, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)


def plot_turns(circuit_info, track_angle, plt):
    offset_vector = [500, 0]
    for _, corner in circuit_info.corners.iterrows():
        txt = f"{corner['Number']}{corner['Letter']}"
        offset_angle = corner['Angle'] / 180 * np.pi
        offset_x, offset_y = rotate(offset_vector, angle=offset_angle)
        text_x = corner['X'] + offset_x
        text_y = corner['Y'] + offset_y
        text_x, text_y = rotate([text_x, text_y], angle=track_angle)
        track_x, track_y = rotate([corner['X'], corner['Y']], angle=track_angle)
        # plt.scatter(text_x, text_y, color='grey', s=350, zorder=10000)
        # plt.plot([track_x, text_x], [track_y, text_y], color='grey', zorder=10000)
        plt.text(text_x, text_y, txt,
                 va='center_baseline', ha='center', size='large', color='white', zorder=10000)


def call_function_from_module(module_name, func_name, *args, **kwargs):
    return getattr(module_name, func_name)(*args, **kwargs)


def append_duplicate_number(arr):
    counts = {}
    result = []

    for item in arr:
        if arr.count(item) > 1:
            counts[item] = counts.get(item, 0) + 1
            result.append(f"{item} {counts[item]}")
        else:
            result.append(item)

    result_arr = np.array(result)

    return result_arr


def update_name(name):
    if name in name_count:
        name_count[name] += 1
    else:
        name_count[name] = 1
    return f"{name} {name_count[name]}"


def get_percentile(data, percentile):
    size = len(data)
    return data[int((size - 1) * percentile)]


def get_quartiles(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0:
        median = (data[size // 2 - 1] + data[size // 2]) / 2
    else:
        median = data[size // 2]

    lower_half = data[:size // 2] if size % 2 == 0 else data[:size // 2]
    upper_half = data[size // 2:] if size % 2 == 0 else data[size // 2 + 1:]

    q1 = get_percentile(lower_half, 0.5)
    q3 = get_percentile(upper_half, 0.5)

    avg = sum(data) / size

    return q1, median, q3, avg


def split_at_discontinuities(arr, threshold=1):
    result = []
    current_segment = [arr[0]]

    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] > threshold:
            result.append(current_segment)
            current_segment = [arr[i]]
        else:
            current_segment.append(arr[i])

    result.append(current_segment)
    return result


def delta_time(reference_lap, compare_lap):
    # ref = reference_lap.get_car_data(interpolate_edges=True).add_distance()
    # comp = compare_lap.get_car_data(interpolate_edges=True).add_distance()
    ref = reference_lap.telemetry
    comp = compare_lap.telemetry

    def mini_pro(stream):
        # Ensure that all samples are interpolated
        dstream_start = stream[1] - stream[0]
        dstream_end = stream[-1] - stream[-2]
        return np.concatenate([[stream[0] - dstream_start], stream, [stream[-1] + dstream_end]])

    ltime = mini_pro(comp['Time'].dt.total_seconds().to_numpy())
    ldistance = mini_pro(comp['Distance'].to_numpy())
    lap_time = np.interp(ref['Distance'], ldistance, ltime)

    delta = lap_time - ref['Time'].dt.total_seconds()

    return delta, ref, comp


def darken_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = list(mcolors.to_rgb(c))
    c = [max(0, i - i * amount) for i in c]
    return c


def find_nearest_non_repeating(array1, array2):
    used_indices = set()  # To keep track of used indices from array2
    nearest_values = []  # To store the nearest values for each element in array1

    for value in array1:
        nearest = None
        nearest_dist = float('inf')

        for i, value2 in enumerate(array2):
            if i not in used_indices:
                dist = abs(value - value2)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = value2
                    nearest_index = i

        nearest_values.append(nearest)
        used_indices.add(nearest_index)  # Mark this index as used

    return nearest_values


def remove_close_rows(df):
    drop_indices = []
    previous_index = None
    for index in df.index:
        if previous_index is not None and (index - previous_index) < 15:
            drop_indices.extend([previous_index, index])
        previous_index = index
    # Remove duplicates from the list
    drop_indices = list(set(drop_indices))
    # Drop the rows
    df = df.drop(index=drop_indices)
    return df


def get_country_names(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    country = ''
    for i in range(len(chunked[0])):
        country += f'{chunked[0][i][0]} '
    # return [leaf[0] for leaf in chunked if isinstance(leaf, Tree) and leaf.label() in ['GPE', 'GSP']]
    if country == 'United ':
        country = 'United Kingdom '
    return country


def get_medal(position):
    medal = ''
    if position == 'P1':
        medal = '🥇'
    elif position == 'P2':
        medal = '🥈'
    elif position == 'P3':
        medal = '🥉'
    return medal


def get_dot(diff):
    if diff > 0:
        return '🟢'
    return '🔴'


def value_to_color(value, min_value, max_value):
    if min_value == max_value:
        # Avoid division by zero if all values are the same
        return '#ffffff'  # White

    # Center the normalization around zero
    normalized_value = (value - min_value) / (max_value - min_value)
    # Rescale to [-1, 1]
    normalized_value = 2 * normalized_value - 1

    def smootherstep(edge0, edge1, x):
        x = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
        return x ** 3 * (x * (x * 6 - 15) + 10)

    if normalized_value < 0:
        scale = smootherstep(-1, 0, normalized_value)
        r = g = b = int(255 * scale)
        b = 255
    else:
        scale = smootherstep(0, 1, normalized_value)
        r = 255
        g = b = int(255 * (1 - scale))

    return f"#{r:02x}{g:02x}{b:02x}"


def create_rounded_barh(ax, data, sector_time, color_col, height=0.8, mode=0):
    times = data.sort_values(by=sector_time, ascending=True)
    colors = times[color_col].values
    y_positions = range(len(times))
    for y, time, color in zip(y_positions, times[sector_time], colors):
        if time != 0:
            lighter_color = lighten_color(color, factor=0.6)
            fixed_time = time if mode == 0 else time + 0.5
            rect = patches.FancyBboxPatch((0 if mode == 0 else -0.5, (y - height / 2)), fixed_time, height,
                                          boxstyle="round,pad=0.02,rounding_size=0.1",
                                          linewidth=2.75,
                                          edgecolor=lighter_color,
                                          facecolor=color)
            ax.add_patch(rect)
        ax.text(time + 0.05, y, f'{time:.3f}s', va='center', ha='left', color='white', fontsize=14.5)

    ax.set_ylim(-1, len(times))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(times.index.values)
    if mode == 0:
        ax.set_xlim(min(times[sector_time]) - 0.5, max(times[sector_time]) + 0.5)
    else:
        ax.set_xlim(0, max(times[sector_time]) + 0.5)


def create_rounded_barh_custom(ax, x_data, y_data, colors, text_to_annotate, height=0.8):
    ax.set_ylim(min(y_data) - 1, max(y_data) + 1)
    ax.set_xlim(min(x_data) - 10, max(x_data) + 50)

    for y, x, color, t in zip(y_data, x_data, colors, text_to_annotate):
        if x != 0:
            lighter_color = lighten_color(color, factor=0.6)
            rect = patches.FancyBboxPatch((0, y - height / 2), x, height,
                                          boxstyle="round,pad=0.05",
                                          linewidth=1,
                                          edgecolor=lighter_color,
                                          facecolor=color)
            ax.add_patch(rect)
        offset = 5
        x_pos = x if x > 0 else 0
        text_position = x_pos + offset
        ax.text(text_position, y, t, va='center', color='white', ha='left', font='Fira Sans', fontsize=13)


def format_timedelta(lap_time):
    return f"{lap_time.seconds // 60:01d}:{lap_time.seconds % 60:02d}.{int(lap_time.microseconds / 1000):03d}"
