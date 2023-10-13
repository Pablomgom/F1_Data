import fastf1
import numpy as np


name_count = {}

def parse_args(args_input, function_map, session):
    args = []
    kwargs = {}

    for arg in args_input.split(','):
        arg = arg.strip()
        if '=' in arg:
            key, value = [x.strip() for x in arg.split('=')]
            kwargs[key] = int(value) if value.isdigit() else value
        else:
            if arg == 'session':
                args.append(session)
            elif arg in function_map:
                args.append(function_map[arg])
            else:
                args.append(int(arg) if arg.isdigit() else arg)

    return args, kwargs


def load_session(year, gp, race_type):
    session = fastf1.get_session(year, gp, race_type)
    session.load()
    return session


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
