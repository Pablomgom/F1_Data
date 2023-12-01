import fastf1
import numpy as np
import matplotlib.colors as mcolors

name_count = {}


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
        plt.scatter(text_x, text_y, color='grey', s=300)
        plt.plot([track_x, text_x], [track_y, text_y], color='grey')
        plt.text(text_x, text_y, txt,
                 va='center_baseline', ha='center', size='large', color='white')


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


def delta_time(reference_lap, compare_lap):
    ref = reference_lap.get_car_data(interpolate_edges=True).add_distance()
    comp = compare_lap.get_car_data(interpolate_edges=True).add_distance()

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