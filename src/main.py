import sys
import fastf1
import traceback
from fastf1.plotting import setup_mpl
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt

from src.awards.awards import awards_2023
from src.ergast_api.my_ergast import My_Ergast
from src.general_analysis.my_ergast_funcs import compare_my_ergast_teammates

from src.utils.utils import parse_args
from src.variables.variables import get_funcs

setup_mpl(misc_mpl_mods=False)
fastf1.ergast.interface.BASE_URL = 'http://ergast.com/api/f1'
plt.rcParams["font.family"] = "Fira Sans"
fastf1.Cache.enable_cache('../cache')

FUNCTION_MAP = get_funcs()
session = None
previous_input = ""

if __name__ == '__main__':

    while True:
        func = input(f"Enter the function name (or 'exit' to quit) [{previous_input}]: ")
        func_name = func.split('(')[0]
        if func_name.lower() == 'exit':
            print("Exiting...")
            sys.exit()
        try:
            args_input = func.split('(')[1].replace(')', '')
            args, kwargs = parse_args(args_input, FUNCTION_MAP, session)
            result = FUNCTION_MAP[func_name](*args, **kwargs)
            if func_name.lower() == 'load_session':
                session = result
            if result is not None:
                print(f"Result: {result}")
        except Exception as e:
            traceback.print_exc()
            # print(e)

    # awards_2023()
