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
from flask import Flask, request, jsonify
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

app = Flask(__name__)

# Assuming your original script's functions are collected in FUNCTION_MAP
FUNCTION_MAP = get_funcs()
session = None


@app.route('/call_function/<func_name>', methods=['POST'])
def call_function(func_name):
    global session
    args_input = request.form.get('args', '')
    try:
        args, kwargs = parse_args(args_input, FUNCTION_MAP, session)
        func = FUNCTION_MAP[func_name]
        if is_session_first_arg(func):
            args.insert(0, session)
        result = func(*args, **kwargs)
        if func_name.lower() == 'load_session':
            session = result
        return jsonify({'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


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

    app.run(debug=True, port=5000)
