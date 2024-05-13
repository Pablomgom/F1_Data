import csv
import sys
from time import sleep

import fastf1
import traceback

import pandas as pd
import pdfminer
import pdfplumber
from fastf1.plotting import setup_mpl
from fastf1.ergast import Ergast
from matplotlib import pyplot as plt

import src.utils.utils
from src.analysis.drivers import race_qualy_h2h
from src.awards.awards import awards_2023
from src.db.db import Database
from src.ergast_api.my_ergast import My_Ergast
from src.menu.menu import get_funcs
import requests
from bs4 import BeautifulSoup
import os

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
# 1	1	Netherlands Max Verstappen	Red Bull Racing-Honda RBPT	1:34.742	1:33.794	1:33.660	1
# 2	11	Mexico Sergio Pérez	Red Bull Racing-Honda RBPT	1:35.457	1:34.026	1:33.982	2
# 3	14	Spain Fernando Alonso	Aston Martin Aramco-Mercedes	1:35.116	1:34.652	1:34.148	3
# 4	4	United Kingdom Lando Norris	McLaren-Mercedes	1:34.842	1:34.460	1:34.165	4
# 5	81	Australia Oscar Piastri	McLaren-Mercedes	1:35.014	1:34.659	1:34.273	5
# 6	16	Monaco Charles Leclerc	Ferrari	1:34.797	1:34.399	1:34.289	6
# 7	55	Spain Carlos Sainz	Ferrari	1:34.970	1:34.368	1:34.297	7
# 8	63	United Kingdom George Russell	Mercedes	1:35.084	1:34.609	1:34.433	8
# 9	27	Germany Nico Hülkenberg	Haas-Ferrari	1:35.068	1:34.667	1:34.604	9
# 10	77	Finland Valtteri Bottas	Kick Sauber-Ferrari	1:35.169	1:34.769	1:34.665	10
# 11	18	Canada Lance Stroll	Aston Martin Aramco-Mercedes	1:35.334	1:34.838	N/A	11
# 12	3	Australia Daniel Ricciardo	RB-Honda RBPT	1:35.443	1:34.934	N/A	12
# 13	31	France Esteban Ocon	Alpine-Renault	1:35.356	1:35.223	N/A	13
# 14	23	Thailand Alexander Albon	Williams-Mercedes	1:35.384	1:35.241	N/A	14
# 15	10	France Pierre Gasly	Alpine-Renault	1:35.287	1:35.463	N/A	15
# 16	24	China Guanyu Zhou	Kick Sauber-Ferrari	1:35.505	N/A	N/A	16
# 17	20	Denmark Kevin Magnussen	Haas-Ferrari	1:35.516	N/A	N/A	17
# 18	44	United Kingdom Lewis Hamilton	Mercedes	1:35.573	N/A	N/A	18
# 19	22	Japan Yuki Tsunoda	RB-Honda RBPT	1:35.746	N/A	N/A	19
# 20	2	United States Logan Sargeant	Williams-Mercedes	1:36.358	N/A	N/A	20
# """
#
#     My_Ergast().insert_qualy_data(2024, 5, data,
#                                   offset=0, character_sep='-', have_country=True, number=1)
#
#

    # def extract_information(full_text):
    #     if full_text:
    #         data = {
    #             "Date": None,
    #             "No / Driver": None,
    #             "Fact": None,
    #             "Offence": None,
    #             "Decision": None,
    #             "Reason": None
    #         }
    #         text_lines = full_text.split('\n')
    #         decision_start = -1
    #         reason_end = -1
    #         reason_text = ""
    #
    #         for i, line in enumerate(text_lines):
    #             if 'Date' in line:
    #                 data['Date'] = line.split('Date')[-1].strip()
    #             if 'Fact' in line:
    #                 data['Fact'] = line.split('Fact')[-1].strip()
    #             if 'Offence' in line:
    #                 data['Offence'] = line.split('Offence')[-1].strip()
    #             if 'Infringment' in line:
    #                 data['Offence'] = line.split('Infringment')[-1].strip()
    #             if 'Infringement' in line:
    #                 data['Offence'] = line.split('Infringement')[-1].strip()
    #             if 'No / Driver' in line:
    #                 data['No / Driver'] = line.split('No / Driver')[-1].strip()
    #             if 'Decision' in line:
    #                 decision_start = i
    #             if 'Reason' in line:
    #                 reason_end = i  # Stop before this line
    #                 reason_text = line.split('Reason')[-1].strip()
    #                 break  # Assuming Reason is the last field to capture
    #
    #         if decision_start != -1 and reason_end != -1:
    #             # Capture everything between Decision and Reason
    #             data['Decision'] = "\n".join(text_lines[decision_start:reason_end]).strip().replace('Decision ', '')
    #
    #         # Find last period in the remaining text after 'Reason'
    #         last_period = full_text.rfind('.')
    #         if last_period != -1:
    #             data['Reason'] = full_text[full_text.find(reason_text):last_period + 1].strip()
    #
    #         return data
    #     return None
    #
    # base_url = 'https://www.fia.com'
    # page_url = base_url + '/documents/championships/fia-formula-one-world-championship-14/season/season-2024-2043'
    # csv_file_path = 'output_data.csv'
    # fieldnames = ['Date', 'No / Driver', 'Fact', 'Offence', 'Decision', 'Reason']
    # base_df = pd.DataFrame(columns=fieldnames)
    # response = requests.get(page_url)
    # soup = BeautifulSoup(response.text, 'html.parser')
    # links = soup.find_all('a', href=True)
    # file_links = [base_url + link['href'] if link['href'].startswith('/') else link['href'] for link in links if
    #               '.pdf' in link['href'].lower()]
    #
    # download_dir = 'downloaded_files'
    # os.makedirs(download_dir, exist_ok=True)
    # for file_link in file_links:
    #     filename = file_link.split('/')[-1]
    #     print(filename)
    #     if 'Decision' in filename or 'Offence' in filename or 'Infringement' in filename or 'Infringment' in filename:
    #         response = requests.get(file_link)
    #         file_path = os.path.join(download_dir, filename)
    #         with open(file_path, 'wb') as file:
    #             file.write(response.content)
    #         try:
    #             with pdfplumber.open(file_path) as pdf:
    #                 full_text = ''.join(page.extract_text() for page in pdf.pages if page.extract_text())
    #                 info = extract_information(full_text)
    #                 if info:
    #                     print("Writing row:", info)
    #                     if not pd.isna(info['No / Driver']):
    #                         df_to_append = pd.DataFrame([info])
    #                         base_df = base_df._append(df_to_append, ignore_index=True)
    #         except pdfminer.pdfparser.PDFSyntaxError:
    #             print(f'{filename} not avilable')

    # Print completion message
    # print("Download completed. Files containing 'Decision' are saved.")

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
