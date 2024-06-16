import sys
from time import sleep

import fastf1
import traceback

import numpy as np
import pandas as pd
from fastf1.plotting import setup_mpl
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt

from src.ergast_api.my_ergast import My_Ergast
from src.menu.menu import get_funcs

from src.utils.utils import parse_args, is_session_first_arg, load_session

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
# 1	63	United Kingdom George Russell	Mercedes	1:13.013	1:11.742	1:12.000	11
# 2	1	Netherlands Max Verstappen	Red Bull Racing-Honda RBPT	1:12.360	1:12.549	1:12.000	21
# 3	4	United Kingdom Lando Norris	McLaren-Mercedes	1:12.959	1:12.201	1:12.021	3
# 4	81	Australia Oscar Piastri	McLaren-Mercedes	1:12.907	1:12.462	1:12.103	4
# 5	3	Australia Daniel Ricciardo	RB-Honda RBPT	1:13.240	1:12.572	1:12.178	5
# 6	14	Spain Fernando Alonso	Aston Martin Aramco-Mercedes	1:13.117	1:12.635	1:12.228	6
# 7	44	United Kingdom Lewis Hamilton	Mercedes	1:12.851	1:11.979	1:12.280	7
# 8	22	Japan Yuki Tsunoda	RB-Honda RBPT	1:12.748	1:12.303	1:12.414	8
# 9	18	Canada Lance Stroll	Aston Martin Aramco-Mercedes	1:13.088	1:12.659	1:12.701	9
# 10	23	Thailand Alexander Albon	Williams-Mercedes	1:12.896	1:12.485	1:12.976	10
# 11	16	Monaco Charles Leclerc	Ferrari	1:13.107	1:12.691	N/A	11
# 12	55	Spain Carlos Sainz	Ferrari	1:13.038	1:12.728	N/A	12
# 13	2	United States Logan Sargeant	Williams-Mercedes	1:13.063	1:12.736	N/A	13
# 14	20	Denmark Kevin Magnussen	Haas-Ferrari	1:13.217	1:12.916	N/A	14
# 15	10	France Pierre Gasly	Alpine-Renault	1:13.289	1:12.940	N/A	15
# 16	11	Mexico Sergio Pérez	Red Bull Racing-Honda RBPT	1:13.326	N/A	N/A	16
# 17	77	Finland Valtteri Bottas	Kick Sauber-Ferrari	1:13.366	N/A	N/A	17
# 18	31	France Esteban Ocon	Alpine-Renault	1:13.435	N/A	N/A	202
# 19	27	Germany Nico Hülkenberg	Haas-Ferrari	1:13.978	N/A	N/A	18
# 20	24	China Guanyu Zhou	Kick Sauber-Ferrari	1:14.292	N/A	N/A	19
# """
#
#     My_Ergast().insert_qualy_data(2024, 9, data,
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
