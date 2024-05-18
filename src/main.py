import sys
from time import sleep

import fastf1
import traceback

import pandas as pd
from fastf1.plotting import setup_mpl
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt

from src.ergast_api.my_ergast import My_Ergast
from src.menu.menu import get_funcs

from src.utils.utils import parse_args, is_session_first_arg

setup_mpl(misc_mpl_mods=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
fastf1.ergast.interface.BASE_URL = 'http://ergast.com/api/f1'
plt.rcParams["font.family"] = "Fira Sans"
fastf1.Cache.enable_cache('../cache')

FUNCTION_MAP = get_funcs()
session = None
previous_input = ""

if __name__ == '__main__':


#     data = """
#
# 1	1	Netherlands Max Verstappen	Red Bull Racing-Honda RBPT	1:15.762	1:15.176	1:14.746	1
# 2	81	Australia Oscar Piastri	McLaren-Mercedes	1:15.940	1:15.407	1:14.820	2
# 3	4	United Kingdom Lando Norris	McLaren-Mercedes	1:15.915	1:15.371	1:14.837	3
# 4	16	Monaco Charles Leclerc	Ferrari	1:15.823	1:15.328	1:14.970	4
# 5	55	Spain Carlos Sainz	Ferrari	1:16.015	1:15.512	1:15.233	5
# 6	63	United Kingdom George Russell	Mercedes	1:16.107	1:15.671	1:15.234	6
# 7	11	Japan Yuki Tsunoda	RB-Honda RBPT	1:15.894	1:15.358	1:15.465	7
# 8	44	United Kingdom Lewis Hamilton	Mercedes	1:16.604	1:15.677	1:15.504	8
# 9	3	Australia Daniel Ricciardo	RB-Honda RBPT	1:16.060	1:15.691	1:15.674	9
# 10	27	Germany Nico Hülkenberg	Haas-Ferrari	1:15.841	1:15.569	1:15.980	10
# 11	11	Mexico Sergio Pérez	Red Bull Racing-Honda RBPT	1:16.404	1:15.706	N/A	11
# 12	31	France Esteban Ocon	Alpine-Renault	1:16.361	1:15.906	N/A	12
# 13	18	Canada Lance Stroll	Aston Martin Aramco-Mercedes	1:16.458	1:15.992	N/A	13
# 14	23	Thailand Alexander Albon	Williams-Mercedes	1:16.524	1:16.200	N/A	14
# 15	10	France Pierre Gasly	Alpine-Renault	1:16.015	1:16.381	N/A	15
# 16	77	Finland Valtteri Bottas	Kick Sauber-Ferrari	1:16.626	N/A	N/A	16
# 17	24	China Guanyu Zhou	Kick Sauber-Ferrari	1:16.834	N/A	N/A	17
# 18	20	Denmark Kevin Magnussen	Haas-Ferrari	1:16.854	N/A	N/A	18
# 19	14	Spain Fernando Alonso	Aston Martin Aramco-Mercedes	1:16.917	N/A	N/A	19
# —	2	United States Logan Sargeant	Williams-Mercedes	No time	N/A	N/A	201
# """
#
#     My_Ergast().insert_qualy_data(2024, 7, data,
#                                   offset=0, character_sep='-', have_country=True, number=1)




    while True:
        func = input(f"Enter the function name (or 'exit' to quit) [{previous_input}]: ")
        func_name = func.split('(')[0]
        if func_name.lower() == 'exit':
            print("Exiting...")
            sys.exit()
        try:
            args_input = func.split('(')[1].replace(')', '')
            args, kwargs = parse_args(args_input, FUNCTION_MAP, session)
            func = FUNCTION_MAP[func_name]
            if is_session_first_arg(func):
                args.insert(0, session)
            result = FUNCTION_MAP[func_name](*args, **kwargs)
            if func_name.lower() == 'load_session':
                session = result
            if result is not None:
                print(f"Result: {result}")
        except Exception as e:
            traceback.print_exc()
            sleep(0.2)
