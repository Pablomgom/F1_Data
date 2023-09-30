def call_function_from_module(module_name, func_name, *args, **kwargs):
    return getattr(module_name, func_name)(*args, **kwargs)
