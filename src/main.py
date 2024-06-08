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
# 1	1	Monaco Charles Leclerc	Ferrari	1:11.584	1:10.825	1:10.270	1
# 2	16	Australia Oscar Piastri	McLaren-Mercedes	1:11.500	1:10.756	1:10.424	2
# 3	55	Spain Carlos Sainz	Ferrari	1:11.543	1:11.075	1:10.518	3
# 4	11	United Kingdom Lando Norris	McLaren-Mercedes	1:11.760	1:10.732	1:10.542	4
# 5	4	United Kingdom George Russell	Mercedes	1:11.492	1:10.929	1:10.543	5
# 6	81	Netherlands Max Verstappen	Red Bull Racing-Honda RBPT	1:11.711	1:10.745	1:10.567	6
# 7	63	United Kingdom Lewis Hamilton	Mercedes	1:11.528	1:11.056	1:10.621	7
# 8	44	Japan Yuki Tsunoda	RB-Honda RBPT	1:11.852	1:11.106	1:10.858	8
# 9	27	Thailand Alexander Albon	Williams-Mercedes	1:11.623	1:11.216	1:10.858	9
# 10	22	France Pierre Gasly	Alpine-Renault	1:11.714	1:10.896	1:11.311	10
# 11	18	France Esteban Ocon	Alpine-Renault	1:11.887	1:11.285	N/A	11
# 12	10	Germany Nico Hülkenberg	Haas-Ferrari	1:11.876	1:11.440	N/A	12
# 13	31	Australia Daniel Ricciardo	RB-Honda RBPT	1:11.785	1:11.482	N/A	13
# 14	23	Canada Lance Stroll	Aston Martin Aramco-Mercedes	1:11.728	1:11.563	N/A	14
# 15	14	Denmark Kevin Magnussen	Haas-Ferrari	1:11.832	1:11.440	N/A	15
# 16	77	Spain Fernando Alonso	Aston Martin Aramco-Mercedes	1:12.019	N/A	N/A	16
# 17	2	United States Logan Sargeant	Williams-Mercedes	1:12.020	N/A	N/A	17
# 18	3	Mexico Sergio Pérez	Red Bull Racing-Honda RBPT	1:12.060	N/A	N/A	18
# 19	20	Finland Valtteri Bottas	Kick Sauber-Ferrari	1:12.512	N/A	N/A	19
# 20	24	China Guanyu Zhou	Kick Sauber-Ferrari	1:13.028	N/A	N/A	20
# """
#
#     My_Ergast().insert_qualy_data(2024, 8, data,
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
