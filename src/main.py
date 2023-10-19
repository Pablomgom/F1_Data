import sys
import fastf1
import pandas as pd
from fastf1.plotting import setup_mpl
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt

from src.ergast_api.my_ergast import My_Ergast
from src.general_analysis.my_ergast_funcs import compare_my_ergast_teammates, get_driver_laps
from src.utils.utils import parse_args
from src.variables.variables import get_funcs

setup_mpl(misc_mpl_mods=False)
fastf1.ergast.interface.BASE_URL = 'http://ergast.com/api/f1'
plt.rcParams["font.family"] = "Fira Sans"
fastf1.Cache.enable_cache('../cache')

FUNCTION_MAP = get_funcs()
session = None


def insert_qualy_data():
    pass


if __name__ == '__main__':
    '''
    while True:
        func = input("Enter the function name (or 'exit' to quit): ")
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
        except IndexError:
            print('Wrong format')
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    '''

    my_ergast = My_Ergast()
    q = my_ergast.get_qualy_results([2007], 11)
    r = my_ergast.get_race_results([2023], 8)
    # compare_my_ergast_teammates('Max', 'Verstappen', start=2013, end=2024)
    get_driver_laps(2014)
    # my_ergast.get_qualy_row(r.content[0], 'Mick Schumacher', 'Haas F1 Team', 47, 20)

