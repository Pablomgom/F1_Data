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
# 1	1	Netherlands Max Verstappen	Red Bull Racing-Honda RBPT	1:05.336	1:04.469	1:04.314	1
# 2	4	United Kingdom Lando Norris	McLaren-Mercedes	1:05.450	1:05.103	1:04.718	2
# 3	63	United Kingdom George Russell	Mercedes	1:05.585	1:05.016	1:04.840	3
# 4	55	Spain Carlos Sainz	Ferrari	1:05.263	1:05.016	1:04.851	4
# 5	44	United Kingdom Lewis Hamilton	Mercedes	1:05.541	1:05.053	1:04.903	5
# 6	16	Monaco Charles Leclerc	Ferrari	1:05.509	1:05.104	1:05.044	6
# 7	81	Australia Oscar Piastri	McLaren-Mercedes	1:05.311	1:05.070	1:05.048	7
# 8	11	Mexico Sergio Pérez	Red Bull Racing-Honda RBPT	1:05.587	1:05.144	1:05.202	8
# 9	27	Germany Nico Hülkenberg	Haas-Ferrari	1:05.596	1:05.262	1:05.385	9
# 10	31	France Esteban Ocon	Alpine-Renault	1:05.574	1:05.274	1:05.883	10
# 11	3	Australia Daniel Ricciardo	RB-Honda RBPT	1:05.569	1:05.289	N/A	11
# 12	20	Denmark Kevin Magnussen	Haas-Ferrari	1:05.508	1:05.347	N/A	12
# 13	10	France Pierre Gasly	Alpine-Renault	1:05.598	1:05.359	N/A	13
# 14	22	Japan Yuki Tsunoda	RB-Honda RBPT	1:05.563	1:05.412	N/A	14
# 15	14	Spain Fernando Alonso	Aston Martin Aramco-Mercedes	1:05.656	1:05.639	N/A	15
# 16	23	Thailand Alexander Albon	Williams-Mercedes	1:05.736	N/A	N/A	16
# 17	18	Canada Lance Stroll	Aston Martin Aramco-Mercedes	1:05.819	N/A	N/A	17
# 18	77	Finland Valtteri Bottas	Kick Sauber-Ferrari	1:05.847	N/A	N/A	8
# 19	2	United States Logan Sargeant	Williams-Mercedes	1:05.856	N/A	N/A	19
# 20	24	China Guanyu Zhou	Kick Sauber-Ferrari	1:06.061	N/A	N/A	19
# """
#
#     My_Ergast().insert_qualy_data(2024, 11, data,
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
