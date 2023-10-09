import numpy as np

name_count = {}


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
