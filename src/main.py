import sys
from time import sleep

import fastf1
import traceback

import pandas as pd
from fastf1.plotting import setup_mpl
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt

import src.utils.utils
from src.awards.awards import awards_2023
from src.db.db import Database
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
# 1	1	Max Verstappen	Red Bull-Honda RBPT	1:16.819	1:16.387	1:15.915	1
# 2	55	Carlos Sainz	Ferrari	1:16.731	1:16.189	1:16.185	2
# 3	11	Sergio Pérez	Red Bull-Honda RBPT	1:16.805	1:16.631	1:16.274	3
# 4	4	Lando Norris	McLaren-Mercedes	1:17.430	1:16.750	1:16.315	4
# 5	16	Charles Leclerc	Ferrari	1:16.984	1:16.304	1:16.435	5
# 6	81	Oscar Piastri	McLaren-Mercedes	1:17.369	1:16.601	1:16.572	6
# 7	63	George Russell	Mercedes	1:17.062	1:16.901	1:16.724	7
# 8	22	Yuki Tsunoda	RB-Honda RBPT	1:17.356	1:16.791	1:16.788	8
# 9	18	Lance Stroll	Aston Martin Aramco-Mercedes	1:17.376	1:16.780	1:17.072	9
# 10	14	Fernando Alonso	Aston Martin Aramco-Mercedes	1:16.991	1:16.710	1:17.552	10
# 11	44	Lewis Hamilton	Mercedes	1:17.499	1:16.960		11
# 12	23	Alexander Albon	Williams-Mercedes	1:17.130	1:17.167		12
# 13	77	Valtteri Bottas	Kick Sauber-Ferrari	1:17.543	1:17.340		13
# 14	20	Kevin Magnussen	Haas-Ferrari	1:17.709	1:17.427		14
# 15	31	Esteban Ocon	Alpine-Renault	1:17.617	1:17.697		15
# 16	27	Nico Hülkenberg	Haas-Ferrari	1:17.976			16
# 17	10	Pierre Gasly	Alpine-Renault	1:17.982			17
# 18	3	Daniel Ricciardo	RB-Honda RBPT	1:18.085			18
# 19	24	Guanyu Zhou	Kick Sauber-Ferrari	1:18.188			19
# """
#
#     My_Ergast().insert_qualy_data(2024, 3, data,
#                                   offset=0, character_sep='-', have_country=False, number=1)




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
