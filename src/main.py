import sys
import fastf1
import traceback

import pandas as pd
from fastf1.plotting import setup_mpl
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt

from src.awards.awards import awards_2023
from src.db.db import Database
from src.ergast_api.my_ergast import My_Ergast
from src.menu.menu import get_funcs

from src.utils.utils import parse_args, is_session_first_arg

# setup_mpl(misc_mpl_mods=False)
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
# 1	18	Argentina Juan Manuel Fangio	Alfa Romeo	1:58.6	 –
# 2	16	Italy Alberto Ascari	Ferrari	1:58.8	+ 0.2
# 3	10	Italy Nino Farina	Alfa Romeo	2:00.2	+ 1.6
# 4	46	Italy Consalvo Sanesi	Alfa Romeo	2:00.4	+ 1.8
# 5	36	Italy Luigi Fagioli	Alfa Romeo	2:04.0	+ 5.4
# 6	48	Italy Dorino Serafini	Ferrari	2:05.6	+ 7.0
# 7	60	Italy Piero Taruffi	Alfa Romeo	2:05.8	+ 7.2
# 8	12	France Raymond Sommer	Talbot-Lago-Talbot	2:08.6	+ 10.0
# 9	4	Italy Franco Rol	Maserati	2:10.0	+ 11.4
# 10	44	France Robert Manzon	Simca-Gordini	2:12.4	+ 13.8
# 11	40	France Guy Mairesse	Talbot-Lago-Talbot	2:13.2	+ 14.6
# 12	42	France Maurice Trintignant	Simca-Gordini	2:13.4	+ 14.8
# 13	58	France Louis Rosier	Talbot-Lago-Talbot	2:13.4	+ 14.8
# 14	64	France Henri Louveau	Talbot-Lago-Talbot	2:13.8	+ 15.2
# 15	30	Thailand Prince Bira	Maserati	2:14.0	+ 15.4
# 16	24	France Philippe Étancelin	Talbot-Lago-Talbot	2:14.4	+ 15.8
# 17	38	Switzerland Toulo de Graffenried	Maserati	2:14.4	+ 15.8
# 18	8	United Kingdom Peter Whitehead	Ferrari	2:16.2	+ 17.6
# 19	6	Monaco Louis Chiron	Maserati	2:17.2	+ 18.6
# 20	56	France Pierre Levegh	Talbot-Lago-Talbot	2:17.2	+ 18.6
# 21	32	United Kingdom Cuth Harrison	ERA	2:18.4	+ 19.8
# 22	2	Belgium Johnny Claes	Talbot-Lago-Talbot	2:18.6	+ 20.0
# 23	52	Italy Felice Bonetto	Milano-Speluzzi	2:19.8	+ 21.2
# 24	50	United Kingdom David Murray	Maserati	2:22.0	+ 23.4
# 25	22	Italy Clemente Biondetti	Ferrari-Jaguar	2:30.6	+ 32.0
# 26	62	Italy Franco Comotti	Maserati-Milano	2:33.6	+ 35.0
# 27	28	Germany Paul Pietsch	Maserati	3:00.2	+ 61.9
# """
#
#     My_Ergast().insert_qualy_data(1950, 7, data,
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
