import pandas as pd
from pandas.core.series import Series
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
import re
from rapidfuzz import fuzz
from sklearn.cluster import AgglomerativeClustering
import requests
from bs4 import BeautifulSoup
import urllib.parse
import os, time, shutil, glob
import lxml
import requests_cache
import cProfile
from requests.adapters import HTTPAdapter
import random
from urllib3.util.retry import Retry
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import webdriver_manager.chrome
from webdriver_manager.chrome import ChromeDriverManager
from tempfile import TemporaryDirectory
import json
from datetime import datetime
from gensim.models.keyedvectors import KeyedVectors
from num2words import num2words
from transformers import BertTokenizer, BertModel
import torch

class MiscDataHelpers:
    @staticmethod
    def drop_columns(df: pd.DataFrame)->pd.DataFrame:
        """
        This function drops unused columns before processing school, job, and applicant ID data.

        :param df: Raw application data Fall 2020-Spring 2024
        :type df: pd.DataFrame - pandas dataframe

        :return: Application data Fall 2020-Spring 2024 with unused columns dropped
        :rtype: pd.DataFrame
        """
        # Drop all columns full of NaN
        df = df.dropna(axis=1, how='all')

        # Drop School Code columns bc not useful
        df = df[df.columns.drop(list(df.filter(regex=r'School [1-6] Code')))]

        # Rename columns
        df = df.rename(columns={'Application Slate ID': 'App ID', 'App - Official PUID': 'PUID', 'State or County of Residence':'State', 'Active Country':'Country', 
                                'App - Citizenship Status': 'Citizenship', 'App - Applicant Term/Year':'App Term', 'App - Program Choice':'Program Choice', 
                                'App - ECE Degree Objective':'Degree Objective', 'App - ECE Area of Interest 1':'ECE Area of Interest 1', 'App - ECE Area of Interest 2':'ECE Area of Interest 2',
                                'Decision History (all decisions)':'Decision History', 'School 1 Language of Instruction':'School 1 Language', 'School 2 Language of Instruction':'School 2 Language',
                                'School 3 Language of Instruction':'School 3 Language', 'School 4 Language of Instruction':'School 4 Language', 'School 5 Language of Instruction':'School 5 Language',
                                'School 6 Language of Instruction':'School 6 Language'})
        
        # Drop applications with no degree objective listed
        df = df.dropna(subset = ['Degree Objective'])

        # Drop unfinished applications
        df = df[df['Application Status'] != 'Awaiting Submission']
        df = df[df['Application Status'] != 'Awaiting Payment']

        # Drop PhD only applications
        df = df[~df['Degree Objective'].str.startswith("Ph.D.", na=False)]
        df = df[~df['Degree Objective'].str.startswith("Direct Ph.D.", na=False)]

        # Drop Application Status - don't need it since already have decison history
        df = df.drop(columns=['Application Status'], axis=1)

        # Drop incomplete applications
        df = df.dropna(subset = ['Decision History'])
        df = df[~df['Decision History'].str.startswith("Awaiting", na=False)]

        # Drop withdrawn / cancelled applications
        df = df[~df['Decision History'].str.startswith("Withdraw", na=False)]

        # Drop people who said they were 0, 5, 6 years old (3 entries)
        df = df[df['Age'] >= 18]

        # Drop columns referring to school and job cities - don't need this level of granularity
        s_cities = ['School ' + str(i) + ' City' for i in range(1,7)]
        j_cities = ['Job ' + str(i) + ' City' for i in range(1, 7)]
        df = df.drop(columns= s_cities + j_cities)
        return df

    @staticmethod
    def process_home_location(df: pd.DataFrame)->pd.DataFrame:
        """
        This function processes all valid applicants' home location.
        'State' column replaced with binary variable 'is_local': 0 = not living in/from Indiana, 1 = living in / from Indiana
        'in_midwest' (ternary variable) column added based on value of 'State' column: -1 = 'State' not listed / np.NaN, 0 = not in Midwest, 1 = in Midwest
        'is_domestic' (binary variable) column added based on value of 'Country' column: 0 = not living in US, 1 = living in US
        'app_income_tier' column added: corresponds to World Bank Level 1 - 4 of Country applicant living in
            1: Low-income
            2: Lower-middle income
            3: Upper-middle income
            4: Upper income
        'app_hdi' column added: corresponds to HDI level of country applicant living in (0-1 float) - 0 if applicant didn't list country
        'app_region' column added: corresponds to region of country applicant lives in
            0: Unknown / np.NaN
            1: North America
            2: South America
            3: Western Europe
            4: Eastern Europe / CIS
            5: Middle East / North Africa
            6: Sub-Saharan Africa
            7: South Asia
            8: East Asia
            9: Southeast Asia
            10: Australia
        Country is also one hot encoded.
        Citizenship is mapped to 0-4 values.

        :param df: raw application data Fall 2020-Spring 2024 only with unused columns dropped
        :type: pd.DataFrame

        :return: pandas dataframe with applicant's home location cleaned and preprocessed for ML models
        :rtype: pd.DataFrame
        """

        # Map applicant's state of residence to nominal label
        midwest_list = {"Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin", "Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska", "North Dakota", "South Dakota"}
        df['State'] = df['State'].fillna(-1)
        df['State'] = df['State'].replace(to_replace=r'^[A-Za-z ]*IN$', value="Indiana", regex=True)
        df['in_midwest'] = 0

        for idx, val in df['State'].items():
            if val in midwest_list:
                df.at[idx, 'in_midwest'] = 1
            if val == 'Indiana':
                df.at[idx, 'State'] = 1
            elif val != -1:
                df.at[idx, 'State'] = 0

        df['is_domestic'] = 0
        df['app_income_tier'] = 0
        df['app_hdi'] = 0
        df['app_region'] = 0

        with open('data/reference/country.json', 'r') as f:
            country_features_json = json.load(f)
        f.close()

        hdi_mapping = {k: v['HDI'] for k, v in country_features_json.items()}
        income_mapping = {k: v['income_tier'] for k, v in country_features_json.items()}
        region_mapping = {k: v['region'] for k, v in country_features_json.items()}

        df['Continent'] = df['Country'].fillna('MISSING') # Will become one hot encoding of country
        df['is_domestic'] = (df['Country'] == 'United States').astype(int)
        df['app_region'] = df['Country'].map(region_mapping)
        df['app_income_tier'] = df['Country'].map(income_mapping)
        df['app_hdi'] = df['Country'].map(hdi_mapping)

        # One Hot Encoding of Country
        df['Country'] = df['Country'].fillna('MISSING')
        df = pd.get_dummies(df, columns=['Country'])

        # Categorize Citizenship
        citizenship_mapping = {'None of the Above':'0', np.nan:'0', 'Asylee or Refugee':'1', 'Permanent Resident Non-US Ctzn':'2', 'U.S. Citizen':'3', 'International':'4'}
        df['Citizenship'] = df['Citizenship'].replace(citizenship_mapping).astype(int)
        return df
    
    @staticmethod
    def process_program_data(df:pd.DataFrame)->pd.DataFrame:
        """
        This function encodes the application term, program choice, degree objective, and ECE Areas of Interest 1 & 2.
        """
        application_terms = {
            'Fall 2020'  : 0,
            'Fall 2021'  : 0,
            'Fall 2022'  : 0,
            'Fall 2023'  : 0,
            'Fall 2024'  : 0,
            'Summer 2024': 1
        }

        program_choice_mapping = {
            'Third Choice': 3, 
            'Second Choice': 2, 
            np.nan: 0, 
            'First Choice': 1
        }

        degree_obj_mapping = {
            'MS/Ph.D.': 0, 
            'Professional Masters - Innovative Technologies': 1, 
            'Masters': 2
        }

        ece_areas = {
            np.nan: 0,
            'Professional Masters – ECE Innovative Technologies': 1,
            'Professional Masters - Innovative Technologies': 1,
            'Micro Electronics and Nanotechnology': 2, 
            'Automatic Control': 3, 
            'Computer Engineering': 4, 
            'Communications Networking Signal and Image Processing': 5, 
            'VLSI and Circuit Design': 6, 
            'Fields and Optics': 7, 
            'Power and Energy Systems': 8, 
            'Biomedical Engineering': 9, 
        }

        # Encode App Term
        df['is_fall'] = df['App Term'].map(application_terms).astype(int)

        # Numerical Encoding of Program Choice
        df['Program Choice'] = df['Program Choice'].map(program_choice_mapping).astype(int)

        # Encode Degree Objective
        df['Degree Objective'] = df['Degree Objective'].map(degree_obj_mapping).astype(int)

        # Encode ECE Area of Interest 1 & 2
        mapped1 = df['ECE Area of Interest 1'].map(ece_areas)
        mapped2 = df['ECE Area of Interest 2'].map(ece_areas)

        df['ECE Area of Interest 1'] = mapped1.astype('Int64')
        df['ECE Area of Interest 2'] = mapped2.astype('Int64')
        return df
    
    @staticmethod
    def process_decision_history(df: pd.DataFrame)->pd.DataFrame:
        """
        This function processes the application decision history to label students as rejected, admitted and accepted admission, and admitted but denied admission.
        The decision history of each student is mapped to numerical encoding.
        The following columns are added to the dataframe:
            * Admitted (Binary) - 0 = Rejected from university, 1 = Admitted to University (not same as accepting enrollment)
            * Enrolled (Binary) - 0 = Rejected or denied enrollment, 1 = Accepted admission and currently enrolled, deferred or changed term
        """

        """
        Possible Admission Decision History for each group: 
        1) admitted: those who were admitted to university and accepted, denied, deferred, changed term
        2) rejected: those who were denied from university or whose application expird
        3) accepted: those who were admitted to university and accepted, deferred or changed term
        """

        decision_mapping = {
            'Admitted, Enrollment Accepted':'0', 
            'Admitted, Enrollment Accepted, Change of Term':'1', 
            'Admitted':'2', 
            'Admitted, Enrollment Declined, Enrollment Accepted':'3', 
            'Enrollment Declined':'4', 
            'Denied':'5', 
            'Admitted, Enrollment Accepted, Deferred Admission':'6'
            }
        
        admitted_mapping = {
            'Admitted, Enrollment Accepted':'1', 
            'Admitted, Enrollment Accepted, Change of Term':'1', 
            'Admitted':'1', 
            'Admitted, Enrollment Declined, Enrollment Accepted':'1', 
            'Enrollment Declined':'1', 
            'Denied':'0', 
            'Admitted, Enrollment Accepted, Deferred Admission':'1'
            }
        
        enrolled_mapping = {
            'Admitted, Enrollment Accepted':'1', 
            'Admitted, Enrollment Accepted, Change of Term':'1', 
            'Admitted':'0', 
            'Admitted, Enrollment Declined, Enrollment Accepted':'1', 
            'Enrollment Declined':'0', 
            'Denied':'0', 
            'Admitted, Enrollment Accepted, Deferred Admission':'1'
            }
        
        # Replace Application Expired with Rejection
        df['Decision History'] = df['Decision History'].replace('Application Expired', 'Denied')

        # Merge Decision History Categories
        df['Decision History'] = df['Decision History'].replace('Admitted, Change of Term', 'Admitted, Enrollment Accepted, Change of Term')
        df['Decision History'] = df['Decision History'].replace('Enrollment Declined', 'Admitted, Enrollment Declined')
        df['Decision History'] = df['Decision History'].replace(to_replace=r'^[A-Za-z\_\-, ]*Enrollment Declined$', value='Enrollment Declined', regex=True)
        df['Decision History'] = df['Decision History'].replace('Admitted, Deferred Admission', 'Admitted, Enrollment Accepted, Deferred Admission')
        
        copy_decision = df['Decision History'].copy()
        df['Decision History'] = df['Decision History'].replace(decision_mapping).astype(int) # Takes into account if they deferred or changed term
        df['Admitted (Binary)'] = copy_decision.map(admitted_mapping).astype(int)
        df['Enrolled (Binary)'] = copy_decision.map(enrolled_mapping).astype(int)
        return df
    
class SchoolDataHelpers:
    @staticmethod
    def normalize_school_name(name: str) -> str:
        """
        This function "normalizes" the school names in columns School 1-6 Institution by removing punctuation and putting names in all lowercase
        to prepare the school names for mapping to school tier categorization.

        :param name: School name in original application format
        :type name: str

        :return: School name in normalized (lowercase, no punctuation) format
        :rtype: str
        """
        if pd.isna(name):
            return "Unknown"
        name = name.strip().lower()
        name = name.replace('.', '').replace(',', '').replace('-', ' ').replace('univ ', 'university ').replace('inst ', 'institute ').replace('comm ', 'community ').replace('com ', 'community ').replace(' ca ', ' california ').replace(' cmty ', ' community ').replace(' admissions', '').replace('*','').replace('junior college', 'community').replace('unversity', 'university').replace('tech ', 'technology ').replace('/', ' ').replace('technlgy', 'technology')
        name = ' '.join(name.split())
        # Copied from https://gist.github.com/JeffPaine/3083347
        abbreviations = [
        # https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States#States.
        "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "IA",
        "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO",
        "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK",
        "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI",
        "WV", "WY",
        # https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States#Federal_district.
        "DC",
        # https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States#Inhabited_territories.
        "AS", "GU", "MP", "PR", "VI",
        ]

        abbreviations = [x.lower() for x in abbreviations]
        last_state_index = max(name.rfind(" "+s) for s in abbreviations)
        
        if (last_state_index == len(name) - 3):
            name = name[0:last_state_index]
        return name

    @staticmethod
    def group_schools(school_set: List[str], df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        This function clusters school names by their similarity up to a threshold of 0.2, using the fuzzywuzzy module.
        Then, each school is replaced in the pandas DataFrame - df (which stores the admissions data) with the first school name in the cluster to reduce 
        number of qualitative school names to categorize.
        The school names have been normalized (all lowercase, stripped, stopwords removed) prior to this function.

        :param school_set: unique list of schools that applicants attended (in columns School 1 - 6)
        :type school_set: List[str]
        :param df: pandas DataFrame storing all applicant data
        :type df: pd.DataFrame
        :param cols: column names from pandas DataFrame storing applicant data / fields from applications
        :type cols: List[str]

        :return: applicant pandas DataFrame with schools replaced by first school in their similarity cluster
        :rtype: pd.DataFrame
        """
        n = len(school_set)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                score = fuzz.token_sort_ratio(school_set[i], school_set[j])
                similarity_matrix[i, j], similarity_matrix[i, j] = score, score

        dist_matrix = 100 - similarity_matrix
        cluster = AgglomerativeClustering(
                                        n_clusters=None, 
                                        metric='precomputed',
                                        linkage='average',
                                        distance_threshold=20
                                        )
        labels = cluster.fit_predict(dist_matrix)

        school_cluster_map = {school_set[i]: labels[i] for i in range(n)}
        inverted_school_cluster_map = defaultdict(list)

        for k, v in school_cluster_map.items():
            inverted_school_cluster_map[v].append(k)

        for k, v in inverted_school_cluster_map.items():
            if len(v) > 1:
                for col in cols:
                    df[col] = df[col].replace(v[1:len(v)], v[0])
        return df

    @staticmethod
    def parse_rx_school_status() -> pd.DataFrame:
        """
        This function reads the R1, R2, R3 status of US schools from csv downloaded from: https://carnegieclassifications.acenet.edu/.
        CSV is stored in pandas DataFrame. The schools names are normalized, and the research activity designation is shortened.

        :return: cleaned pandas DataFrame for research designation (includes 3000+ US schools from https://carnegieclassifications.acenet.edu/)
        :rtype: pd.DataFrame
        """
        mapping = {
            'Research 2: High Research Spending and Doctorate Production': 2,
            'Research Colleges and Universities': 3,
            'Research 1: Very High Research Spending and Doctorate Production': 1
        }

        df2 = pd.read_csv('data/reference/ace-institutional-classifications.csv', usecols=['Research Activity Designation', 'name'])
        mapped = df2['Research Activity Designation'].map(mapping)
        df2['Research Activity Designation'] = mapped.where(mapped.notna(), df2['Research Activity Designation'])
        df2['name'] = df2['name'].str.lower()
        return df2

    @staticmethod
    def map_rx_school_status(app_df: pd.DataFrame, carnegie_csv_df: pd.DataFrame, col: str) -> dict[str:List[str]]:
        """
        This function maps the school name from application to research designation R1, R2, R3 if it has exact match in csv from https://carnegieclassifications.acenet.edu/.

        :param app_df: pandas DataFrame containing data from application csv for Fall 2020 - Spring 2024
        :type app_df: pd.DataFrame
        :param carnegie_csv_df: pandas DataFrame containing name of schools and their corresponding R1, R2, R3 research designation from the Carnegie Classifications website csv
        :type carnegie_csv_df: pd.DataFrame
        :param col: column name as string for columns in app_df that correspond to school names
        :type col: str
        
        :return: dictionary mapping of applicant school name to R1, R2, R3 designation if there's exact match of name to that in csv file
        :rtype: dict[str:List[str]]
        """
        for idx, s in app_df[col].items():
            if s in carnegie_csv_df['name'].values:
                app_df.at[idx, col] = carnegie_csv_df.loc[carnegie_csv_df['name'] == s, 'Research Activity Designation'].item()
        return app_df

    @staticmethod
    def query_all_us_schools(schools_left: set, school_dict: dict) -> dict:
        """
        This function scrapes the Carnegie Classification website for each school that was not a 1 to 1 match from the Carnegie Classifiation csv.
        The top result ranking is assigned to the school dictionary for the associated school. If there's no match, None is assigned.

        :param schools_left: Set of school names that don't have an assigned ranking yet (R1, R2, R3, Unknown, etc.)
        :type schools_left: set
        :param school_dict: Set of rankings with associated school names - to be updated in function for US schools
        :type school_dict: dict

        :return: the updated school dictionary with rankings matched to school name
        :rtype: dict
        """
        session = requests_cache.CachedSession(
            'my_cache', 
            expire_after=timedelta(hours=1)
        )

        HEADERS = {
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer":"https://carnegieclassifications.acenet.edu/institutions/",
        }

        session.headers.update(HEADERS)

        # Configure retires and pool size
        adapter = HTTPAdapter(
            pool_connections=100,
            pool_maxsize=100,
            max_retries=Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504]
            )
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        i = 1
        for s in schools_left:
            if ((i % 20) == 0) and (i != 0):
                session.close()
                time.sleep(random.uniform(2.5, 5))
                session = requests_cache.CachedSession('my_cache', expire_after=timedelta(hours=1))
                session.headers.update(HEADERS)
                session.mount("http://", adapter)
                session.mount("https://", adapter)
            research_des = query_carnegie_classifications(i, s, session)
            school_dict[s] = research_des
            time.sleep(random.uniform(2.5, 5))
            i += 1
        return school_dict

    @staticmethod
    def query_all_intl_schools(intl_schools: List[str]) -> dict[str, str]:
        """
        This function scrapes the QS world University rankings website for all non-US schools that were not previously identified from the QS World Rankings csv.

        :param intl_schools: Names of all international schools that have not had rank assigned yet
        :type intl_schools: List[str]

        :return: Mapping of school name to one of the following rankings: Intl Top, Intl Mid, Intl Low, Unknown
        :rtype: dict
        """
        options = webdriver.ChromeOptions()
        options.add_argument("--incognito")
        options.page_load_strategy = 'normal'
        service = Service(executable_path="/home/min/a/rhorwatt/local/chromedriver/usr/lib64/chromium-browser/chromedriver")
        driver = webdriver.Chrome(
            service=service, 
            options=options
            )

        intl_dict = {
            "Unknown":[],
            "Intl Top":[],
            "Intl Mid":[],
            "Intl Low":[]  
        }

        i = 1
        for s in intl_schools:
            research_des = query_qs_website(i, s, driver)
            if research_des == None:
                intl_dict["Unknown"].append(s)
            else:
                intl_dict[research_des] = s
            time.sleep(random.uniform(2.5, 5))
            i += 1
        driver.quit()
        return intl_dict

    @staticmethod
    def query_carnegie_classifications(i: int, s: str, session: requests_cache.CachedSession) -> str:
        """
        This function scrapes the Carnegie Classifications website to return the research designation for the top search result that matches school s
        from the application.

        :param i: index (1-based) of school
        :type i: int
        :param s: 1 school name listed on application
        :type s: str
        :param session: Reused cached session class object used for HTTP GET requests to acquire top search result
        :type session: requests_cache.CachedSession

        :return: Research Classification for school s: "R1,", "R2", "R3", "Community" (if listed on Carnegie website but doesn't have R1, R2, R3 designation), or "None"
        :rtype: str

        Note: Schools that are labeled as "None" by this web scraping may be updated to have a tiered designation for an international school later, as this function
        only assigns research designation / tier to US schools.
        """
        search_url = "https://carnegieclassifications.acenet.edu/institutions/"
        reformatted_school = "+".join(s.split(' '))
        url = search_url+"/?inst="+reformatted_school

        try:
            resp = session.get(url, timeout=(5, 15))
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, 'lxml')
            container = soup.find("div", class_="card-list-archive-institution")

            if container:
                card = container.find("div", class_="card card-archive-institution mb-4")
                top_result = card.find("h4").find("a")
                if top_result:
                    detail_url = urllib.parse.urljoin("https://carnegieclassifications.acenet.edu/", top_result["href"])
                    time.sleep(random.uniform(2.5, 5))

                    try:
                        resp = session.get(detail_url, timeout=(5, 15)) # (connect timeout, read timeout)
                        resp.raise_for_status()
                        soup = BeautifulSoup(resp.text, "lxml")
                        container = soup.find("div", class_="col p-3")
                        
                        if container:
                            card = container.find("div", class_="institution-data-point-icon")
                            research_text = card.get_text(strip=True)
                            research_text = research_text.replace("Research Activity Designation:", "").strip()
                            
                            if "Research 1" in research_text:
                                return "R1"
                            elif "Research 2" in research_text:
                                return "R2"
                            elif "Research Colleges" in research_text:
                                return "R3"
                            else:
                                return None
                        else:
                            return "Community"
                    except requests.exceptions.RequestException as e:
                        print(f"[{i}] Error:", e)
        except requests.exceptions.RequestException as e:
            print(f"[{i}] Error:", e)
        return None

    @staticmethod
    def query_qs_website(i:int, s: str, driver: webdriver.Chrome) -> str:
        """
        This function scrapes the QS World Rankings website to return the research designation for the top search result that matches school s
        from the application.

        :param i: index (1-based) of school
        :type i: int
        :param s: 1 school name listed on application
        :type s: str
        :param driver: Reused selenium webdriver used for HTTP GET requests to acquire top search result
        :type driver: selenium.webdriver.Chrome

        :return: Research Classification / Tier for school s: "Intl Top", "Intl Mid", "Intl Low", or "Unknown"
        :rtype: str

        Note: Schools that are labeled as "None" by this web scraping may be updated to have a tiered designation later with manual search.
        """
        search_url = "https://edurank.org/uni-search?s="
        reformatted_school = "+".join(s.split(' '))
        url = search_url+reformatted_school

        try:
            driver.get(url)
            driver.implicitly_wait(10)

            try:
                iframes = driver.find_elements(By.TAG_NAME, "iframe")
                for i, f in enumerate(iframes):
                    src = f.get_attribute("src")
                html = driver.page_source
                soup = BeautifulSoup(html, 'lxml')
                pretty_soup = soup.prettify()

                with open("edurank.html", "w", encoding="utf-8") as f:
                    f.write(pretty_soup)
                f.close()
            except:
                return None
        except Exception as e:
            print(f"[{i}] Error:", e)
        return None

    @staticmethod
    def encode_one_major(major: str) -> int:
        """
        This function maps each major written on application to 0 - 9 numerical mapping.

        :param major: Lowercase, cleaned major listed in application
        :type major: str
        :return: returns 0-9 mapping for major
        :rtype: int
        """

        # Major Mappings from Str to Int:
        # 0 - Unknown / NaN / High School / Non-degree (NaN on application were already set to 0)
        # 1 - ECE / Robotics
        # 2 - CS / Data Science / Cybersecurity
        # 3 - Mechanical / Aerospace / Materials Engineering
        # 4 - Physics / Math / Statistics
        # 5 - Chemical / Biochemical / Nuclear Engineering
        # 6 - Biology / Biomedical Engineering or Sciences / Neuroscience / Health Sciences
        # 7 - Civil / Environmental / Earth Sciences / Geology / Agriculture
        # 8 - Business / Management / Finance / Economics
        # 9 - Social Sciences / Humanities / Arts / Languages
        # 10 - Higher Advanced Degree like MD, JD

        if not isinstance(major, str) or \
            (major == "assocate") or \
            ("no major" in major) or \
            ("high school" in major) or \
            ("non degree" in major) or \
            ("pre- engineering" in major) or \
            ("pre engineering" in major) or \
            ("not specified" in major) or \
            ("summer session" in major):
            return 0
        
        if any(word in major for word in [
            "general", "----", "other", "guest", "undeclared", "n/a",
            "non-degree", "undecided", "ece", "18.033x", "pre-engineering",
            "undelcared", "undeclaired", "non-matriculated", "transient",
            "certificate", "certification"
        ]):
            return 0
        
        if ("computer engineering" in major) or \
            ("information engineering" in major):
            return 1
        
        if any(word in major for word in [
            "electrical", "electronics", "instrumentation", "robotics", "eet",
            "nanoscience", "microelectronic", "electircal", "nanotechnology",
            "power", "electric", "control", "instrumentation", "compter", "avionics",
            "vlsi", "energy", "battery", "systems", "embedded", "electronic", "ee",
            "electrification", "eletrical", "eletrical"
        ]):
            return 1
        
        if any(word in major for word in [
            "cybersecurity", "it", "software", "computer", "technology"
        ]):
            return 2
        
        if ("artificial intelligence" in major) or \
            ("data science" in major) or \
            ("computer science" in major):
            return 2
        
        if any(word in major for word in [
            "mechanical", "materials", "aerospace", "space", "aeronautics",
            "astronautics", "industrial", "manufacturing", "electromechanic",
            "mechatronics", "astronautical", "mechancial", "aeronautical",
            "mechanics", "automation", "mechnical"
        ]):
            return 3
        
        if ("safety engineering" in major):
            return 3
        
        if any(word in major for word in [
            "physics", "math", "statistics", "mathematics", "electrophysics",
            "optical"
        ]):
            return 4
        
        if any(word in major for word in [
            "chemical", "biochemical", "nuclear", "construction", "architectural",
            "petroleum", "polymer", "bioengineering", "che"
        ]):
            return 5
        
        if ("water resources" in major) or \
            ("biological engineering" in major):
            return 5
        
        if any(word in major for word in [
            "biology", "neuroscience", "biomedical", "chemistry", "ecology", "health",
            "geology", "science"
        ]):
            return 6
        
        if any(word in major for word in [
            "civil", "environmental", "geology", "agriculture", "agricultural"
        ]):
            return 7
        
        if ("traffic engineering" in major):
            return 7
        
        if any(word in major for word in [
            "business", "management", "finance", "economics", "mba", "administration",
            "managment", "fincance", "organizational", "leadership"
        ]):
            return 8

        if any(word in major for word in [
            "sociology", "humanities", "english", "dance", "psychology", "art",
            "biblical", "liberal", "arts", "arabic", "journalism", "vocal", "performing",
            "external", "russian", "communication", "audio", "ethics", "asian", "linguistics",
            "interdisciplinary", "performance", "music", "political", "anthropology",
            "folklore", "christian", "laws", "law", "creative", "writing", "philosophy",
            "french", "international"
        ]):
            return 9
        
        if ("intelligence studies" in major) or \
            ("middle eastern" in major) or \
            ("public policy" in major):
            return 9
        
        if any(word in major for word in [
            "md", "jd", "medicine"
        ]):
            return 10

        if "engineering" in major:
            return 3
        return 0

    @staticmethod
    def encode_one_degree_level(degree: str) -> int:
        """
        This function returns the degree code for degree string.

        # Degree Str to Int Mapping
        # 0 - NaN, Non-Degree, High School, etc.
        # 1 - Undergraduate (BA/BS/BEng)
        # 2 - Graduate (MA, MS, PhD, MD, JD, MBA)
        """
        if degree == 0:
            return 0
        
        if not isinstance(degree, str) or \
            (degree == np.nan) or \
            ("no degree obtained" in degree) or \
            ("other degree" in degree):
            return 0

        if any(word in degree for word in [
            "associates", "bachelor"
        ]):
            return 1
        
        if any(word in degree for word in [
            "master", "juris", "doctor", "graduate"
        ]):
            return 2
        return 0

    @staticmethod
    def encode_school_level(df: pd.DataFrame) -> pd.DataFrame:
        """
        This function encodes the degree level of the school (Undergraduate, Graduate, Unknown).
        """
        school_types = {
            'G':'2', 
            np.nan:'0', 
            'U':'1'
            }

        cols = ['School ' + str(i) + ' Type' for i in range(1, 7)]

        for c in cols:
            df[c] = df[c].replace(school_types).astype(int)
        return df

    @staticmethod
    def encode_school_tier(df: pd.DataFrame) -> pd.DataFrame:
        """
        This function labels every school in School 1-6 Institution columns with tier from 0 to 10.
        Column added 'Purdue (Binary)': 0 = did not attend Purdue Main or Purdue Branch, attended Purdue Main or Purdue Branch
        # School Tier Categorization
        # 1 - US R1
        # 2 - US R2
        # 3 - US R3
        # 4 - US Other
        # 5 - US Community College
        # 6 - International Top
        # 7 - International Mid
        # 8 - International Other
        # 9 - High School
        # 10 - Online
        """
        cols = ['School ' + str(i) + ' Institution' for i in range(1,7)]

        purdue_branches = [
            'purdue university calumet hammond in', 
            'purdue university remote delco electronics kokomo', 
            'purdue university fort wayne', 
            'purdue university global', 
            'purdue university statewide tech in', 
            'indiana purdue university/indpls'
            ]
        
        all_purdue = set([
            'purdue university calumet hammond in', 
            'purdue university remote delco electronics kokomo', 
            'purdue university fort wayne', 
            'purdue unversity', 
            'purduex 69503x', 
            'purdue', 
            'indiana university purdue university indianapolis', 
            'purduex 69501x', 
            'purdue university west lafayette', 
            'purdue university', 
            'purdue university global', 
            'purdue university west lafayette*', 
            'purdue university west lafayette in', 
            'purdue university (simplilearn)', 
            'purdue university statewide tech in', 
            'indiana purdue university/indpls'
            ])
        
        purdue_main = [
            'purdue unversity', 
            'purdue', 
            'purdue university west lafayette', 
            'purdue university', 
            'purdue university west lafayette*', 
            'purdue university west lafayette in'
            ]

        df['Purdue (Binary)'] = 0
        missing_school_cols = ['School ' + str(i) + ' Missing' for i in range(1,7)]
        df2 = SchoolDataHelpers.parse_rx_school_status()  # Stores all Carnegie Classifications for US schools (R1, R2, R3, Other) from Carnegie website

        with open("data/reference/school_to_rank.json", "r", encoding="utf-8") as f:
            school_to_rank = json.load(f)
        f.close()

        # Normalize / clean all school names
        for m, c in zip(missing_school_cols, cols):
            df[c] = df[c].apply(SchoolDataHelpers.normalize_school_name)
            df[c] = df[c].replace(to_replace=r'^[A-Za-z\-,\(\)\/ ]*community[A-Za-z\-\(\)\/ ]*$', value='community', regex=True)
            df[c] = df[c].replace(purdue_branches, 'purdue branch')
            df[c] = df[c].replace(purdue_main, 'purdue')
            df = SchoolDataHelpers.map_rx_school_status(df, df2, c)
            df[m] = 0

            for idx, val in df[c].items():
                if (val in all_purdue) and not(df.at[idx, 'Purdue (Binary)']):
                    # Mark applicant as attending Purdue or Purdue branch campus previously
                    df['Purdue (Binary)'] = 1

                if (not isinstance(val, int)) and (val in school_to_rank):
                    # Map school tier to that from school_to_rank.json reference
                    df.at[idx, c] = school_to_rank[val]
                elif (val == "Unknown"):
                    df.at[idx, m] = 1
                    df.at[idx, c] = 0
                elif (val == "no response") or (val == "sean greene"):
                    df.at[idx, c] = 0
        return df
    
    @staticmethod
    def one_hot_encode_school_lang(df: pd.DataFrame) -> pd.DataFrame:
        """
        This function one hot encodes school language of instruction in columns School 1-6 Language.
        """
        language_cols = ['School ' + str(i) + ' Language' for i in range(1,7)]
        for i, c in enumerate(language_cols):
            df[c] = df[c].fillna('MISSING')
            for idx, val in df[c].items():
                if (val == 'Chinese - Cantonese') or (val == 'Chinese - Mandarin'):
                    df.at[idx, val] = 'Chinese'
                elif (val == 'Persian (Farsi)'):
                    df.at[idx, val] = 'Farsi'
            df = pd.get_dummies(df, columns=[c], prefix=f'School_{i}_Lang')
        return df
    
    @staticmethod
    def encode_all_majors(df: pd.DataFrame) -> pd.DataFrame:
        """
        This function encodes majors for all applicants.
        """
        major_cols = ['School ' + str(i) + ' Major' for i in range(1, 7)]

        for c in major_cols:
            df[c] = df[c].str.lower()
            df[c] = df[c].fillna(0)
            for idx, s in df[c].items():
                df.at[idx, c] = SchoolDataHelpers.encode_one_major(s)
        return df
    
    @staticmethod
    def clean_all_gpa(df: pd.DataFrame) -> pd.DataFrame:
        """
        This function updates all valid GPAs to 4.0 scale. If GPA is unlisted or np.NaN, replaced with 0.
        Default GPA scale is set as 4.0 because it appears all GPAs were GPA scale was unlisted were on 4.0 scale.
        """
        gpa_cols = ['School ' + str(i) + ' GPA' for i in range(1, 7)]
        gpa_scale_cols = ['School ' + str(i) + ' GPA Scale' for i in range(1, 7)]
        gpa_conv_cols = ['School ' + str(i) + ' GPA Converted' for i in range(1,7)]
        confirmed_cols = ['School ' + str(i) + ' Confirmed' for i in range(1,7)]

        for c in gpa_cols:
            # Fill NaN GPA's with 0 - they might not have school information
            df[c] = df[c].fillna(0)

        for i, c in enumerate(gpa_scale_cols):
            # Fill NaN values for GPA scale
            df[c] = df[c].fillna(0)
            for idx, val in df[c].items():
                if val == 0:
                    gpa_col = 'School ' + str(i + 1) + ' GPA'
                    if df.at[idx, gpa_col] != 0:
                        df.at[idx, c] = 4.0

        for i, c in enumerate(gpa_conv_cols):
            df[c] = df[c].fillna(0)
            for j, val in df[c].items():
                if (val == 0) and (df.at[j, gpa_cols[i]] != 0):
                    df.at[j, c] = (df.at[j, gpa_cols[i]] / df.at[j, gpa_scale_cols[i]]) * 4

        df = df.drop(columns=['School 1 GPA Recalculated'])
        df = df.drop(columns= gpa_cols+gpa_scale_cols)

        for c in confirmed_cols:
            df[c] = df[c].fillna(0)
        return df
    
    @staticmethod
    def encode_all_degree_levels(df: pd.DataFrame) -> pd.DataFrame:
        """
        This function numerically encodes degree level attained from schools in columns School 1-6 Institution (ex: Masters, Bachelors, etc).
        """
        degree_cols = ['School ' + str(i) + ' Degree' for i in range(1, 7)]

        for c in degree_cols:
            df[c] = df[c].fillna(0)
            df[c] = df[c].str.lower()
            for idx, s in df[c].items():
                df.at[idx, c] = SchoolDataHelpers.encode_one_degree_level(s)
        return df
    
    @staticmethod
    def compute_last_edit_delta(df: pd.DataFrame, app_deadlines: dict[str, datetime]) -> pd.DataFrame:
        """
        This function drops the columns for school 1-6 creation.
        It also calculates the difference between last time applicant edited school information and the application due date.

        :param df: partially cleaned pandas dataframe of raw applicant data for Fall 2020 - Spring 2024 semesters
        :type df: pd.DataFrame
        :param app_deadlines: dictionary of application semester and corresponding application deadline
        :type app_deadlines: dict[str, datetime]

        :return: application dataframe with difference between last school information edit and application due date updated
        :rtype: pd.DataFrame
        """
        update_cols = ['School ' + str(i) + ' Updated' for i in range(1,7)]
        new_cols = ['School ' + str(i) + ' Update' for i in range(1,7)]

        for c in new_cols:
            df[c] = np.timedelta64(0, 'ns')

        for j, c in enumerate(update_cols):
            df[c] = df[c].fillna(0)
            for i, val in df[c].items():
                if (val != 0):
                    df.at[i, new_cols[j]] = app_deadlines[df.at[i, 'App Term']] - val.to_pydatetime()
        
        df = df.drop(columns=update_cols)
        return df
    
    @staticmethod
    def compute_school_job_duration_recency(df: pd.DataFrame, app_deadlines: dict[str, datetime]) -> pd.DataFrame:
        """
        This function computes the duration applicant attended each school listed on application
        and calculates recency of last degree graduation to deadline of application.

        :param df: partially cleaned pandas dataframe of raw applicant data for Fall 2020 - Spring 2024 semesters
        :type df: pd.DataFrame
        :param app_deadlines: dictionary of application semester and corresponding application deadline
        :type app_deadlines: dict[str, datetime]

        :return: application dataframe with duration of school and job attendance, as well as recency of last attendance and work updated
        :rtype: pd.DataFrame
        """

        s_from_cols = ['School ' + str(i) + ' From' for i in range(1, 7)]
        s_to_cols = ['School ' + str(i) + ' To' for i in range(1, 7)]
        s_duration_cols = ['School ' + str(i) + ' Duration' for i in range(1, 7)]
        s_recency_cols = ['School '+ str(i) + ' Recency' for i in range(1, 7)]

        j_from_cols = ['Job ' + str(i) + ' From' for i in range(1, 7)]
        j_to_cols = ['Job ' + str(i) + ' To' for i in range(1, 7)]
        j_duration_cols = ['Job ' + str(i) + ' Duration' for i in range(1, 7)]
        j_recency_cols = ['Job '+ str(i) + ' Recency' for i in range(1, 7)]
        zero_time_delta = app_deadlines['Fall 2020'] - app_deadlines['Fall 2020']

        for s_from, s_to, s_dur, s_rec, j_from, j_to, j_dur, j_rec in zip(s_from_cols, s_to_cols, s_duration_cols, s_recency_cols, j_from_cols, j_to_cols, j_duration_cols, j_recency_cols):
            df[s_from].fillna(0)
            df[s_to].fillna(0)
            df[j_from].fillna(0)
            df[j_to].fillna(0)

            df[s_dur] = zero_time_delta
            df[s_rec] = zero_time_delta
            df[j_dur] = zero_time_delta
            df[j_rec] = zero_time_delta

            for row, s_from_val in df[s_from].items():
                df.at[row, s_dur] = s_from_val - df.at[row, s_to]
                df.at[row, j_dur] = df.at[row, j_from] - df.at[row, j_to]
                s_recency = app_deadlines[df.at[row, 'App Term']] - df.at[row, s_from]
                j_recency = app_deadlines[df.at[row, 'App Term']] - df.at[row, j_from]

                if s_recency < df.at[row, s_rec]:
                    df.at[row, s_rec] = s_recency
                if j_recency < df.at[row, j_rec]:
                    df.at[row, j_rec] = j_recency

        df = df.drop(columns= s_from_cols + s_to_cols + j_from_cols + j_to_cols)
        return df

    @staticmethod
    def fill_na_class_rank_size(df: pd.DataFrame) -> pd.DataFrame:
        """
        This function replaces np.NaN values in the School 1 - 6 Class Rank and Class Size columns with the median of rank and size.
        Some students who didn't report a class rank or size still listed other school information, so filling with median instead of 0.
        """
        rank_cols = ['School ' + str(i) + ' Class Rank (Numeric)' for i in range(1,7)]
        size_cols = ['School ' + str(i) + ' Class Size (Numeric)' for i in range(1,7)]

        for r, s in zip(rank_cols, size_cols):
            df[s] = df[s].fillna(df[s].median())
            df[r] = df[r].fillna(df[r].median())
        return df
    
    @staticmethod
    def encode_school_job_country(df: pd.DataFrame) -> pd.DataFrame:
        """
        This function encodes the school and job regions (state or non-US city) as Boolean variable - 1 if is in Indiana, 0 if not.
        The school and job countries are encoded as 3 categorical variables: HDI of country, Country Region (0 - 10 on world region), Income Tier (1-4 World Bank scale).
        """
        s_region_cols = ['School ' + str(i) + ' Region' for i in range(1,7)]
        s_country_cols = ['School ' + str(i) + ' Country' for i in range(1,7)]
        j_region_cols = ['Job ' + str(i) + ' Region' for i in range(1,7)]
        j_country_cols = ['Job ' + str(i) + ' Country' for i in range(1,7)]

        s_c_hdi_cols = ['School ' + str(i) + ' Country HDI' for i in range(1,7)]
        s_c_reg_cols = ['School ' + str(i) + ' Country Region' for i in range(1,7)]
        s_c_inc_cols = ['School ' + str(i) + ' Country Income' for i in range(1,7)]
        j_c_hdi_cols = ['Job ' + str(i) + ' Country HDI' for i in range(1,7)]
        j_c_reg_cols = ['Job ' + str(i) + ' Country Region' for i in range(1,7)]
        j_c_inc_cols = ['Job ' + str(i) + ' Country Income' for i in range(1,7)]

        with open('data/reference/country.json', 'r') as f:
            country_features_json = json.load(f)
        f.close()

        hdi_map = {k: v['HDI'] for k, v in country_features_json.items()}
        income_map = {k: v['income_tier'] for k, v in country_features_json.items()}
        region_map = {k: v['region'] for k, v in country_features_json.items()}

        i = 0
        mapped_df = pd.DataFrame(index=df.index)

        for s_r, s_c, j_r, j_c in zip(s_region_cols, s_country_cols, j_region_cols, j_country_cols):
            mapped_df[s_c_hdi_cols[i]] = df[s_c].map(hdi_map).fillna(0)
            mapped_df[s_c_inc_cols[i]] = df[s_c].map(income_map).fillna(0)
            mapped_df[s_c_reg_cols[i]] = df[s_c].map(region_map).fillna(0)

            mapped_df[j_c_hdi_cols[i]] = df[j_c].map(hdi_map).fillna(0)
            mapped_df[j_c_inc_cols[i]] = df[j_c].map(income_map).fillna(0)
            mapped_df[j_c_reg_cols[i]] = df[j_c].map(region_map).fillna(0)

            for j, val in df[s_r].items():
                if (val != 0) and (val == 'IN'):
                    df.at[j, s_r] = 1
                else:
                    df.at[j, s_r] = 0
            for j, val in df[j_r].items():
                if (val != 0) and (val == 'IN'):
                    df.at[j, j_r] = 1
                else:
                    df.at[j, j_r] = 0
            i += 1

        df = pd.concat([df, mapped_df], axis=1)
        df = df.drop(columns=s_country_cols + j_country_cols)
        return df
    
    @staticmethod
    def encode_degree_conferred(df: pd.DataFrame, app_deadlines: dict[str, datetime]) -> pd.DataFrame:
        """
        This function adds a Boolean variable to indicate if student graduated from each school listed on application.
        It also updates the degree conferred field to be the difference between the application deadline for each term
        and the date of degree conferral.
        """
        degree_cols = ['School ' + str(i) + ' Degree Conferred' for i in range(1,7)]
        earned_cols = ['School ' + str(i) + ' Degree Conferred (Binary)' for i in range(1,7)]
        df.loc[:, earned_cols] = 0

        for i, c in enumerate(degree_cols):
            df[c] = df[c].fillna(timedelta(-1, -1, -1, -1, -1))
            for j, val in df[c].items():
                if (val != timedelta(-1, -1, -1, -1, -1)):
                    df.at[j, earned_cols[i]] = 1
                    df.at[j, c] = app_deadlines[df.at[j, "App Term"]] - df.at[j, c]
        return df
    
class JobDataHelpers:
    @staticmethod
    def clean_one_description(j_desc: str, token: BertTokenizer, model: BertModel) -> str:
        """
        This function cleans valid job descriptions by removing punctuation.
        """
        def digit_to_word(match):
            num_str = match.group()
            try:
                return num2words(int(num_str))
            except TypeError:
                return num_str
    
        hidden_size = 768

        if (j_desc != 0) or (j_desc != '0'):
            j_desc = re.sub(r'[\n•\-?*]', ' ', str(j_desc))
            j_desc = re.sub(r'\b\d+\b', digit_to_word, j_desc)
            j_desc = re.sub(r'\s+', ' ', j_desc).strip()
            j_desc = j_desc.lower()

            if j_desc.strip() == "":
                return None, None
            
            inputs = token(j_desc, return_tensors="pt")
            outputs = model(**inputs)
            pooler_output = outputs.pooler_output
            return pooler_output.detach().cpu().numpy(), pooler_output
        return None, None
    
    @staticmethod
    def encode_job_description(df: pd.DataFrame) -> pd.DataFrame:
        """
        This function returns sets the BERT embedding for each job description listed on application.
        """
        desc_cols = ['Job ' + str(i) + ' Description' for i in range(1,7)]
        np_cols = ['Job ' + str(i) + ' Description (np.array)' for i in range(1,7)]
        tensor_cols = ['Job ' + str(i) + ' Description (tensor)' for i in range(1,7)]
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        for desc_col, np_col, tensor_col in zip(desc_cols, np_cols, tensor_cols):
            df[desc_col].fillna(0)
            df[np_col] = [np.zeros(768, dtype=np.float32) for _ in range(1276)]
            df[tensor_col] = [torch.zeros(768) for _ in range(1276)]

            for idx, val in df[desc_col].items():
                np_arr, t_torch = JobDataHelpers.clean_one_description(val, tokenizer, model)

                if np_arr is not None:
                    df.at[idx, np_col] = np_arr
                if t_torch is not None:
                    df.at[idx, tensor_col] = t_torch
        df.drop(columns = desc_cols)
        return df

    @staticmethod
    def embed_text(j_title: str, model: KeyedVectors) -> np.array:
        """
        This function cleans an individual job title listed in the application.
        It removes punctuation and converts ints to strs. It also returns the
        Word2Vec averaged embedding for all words in the job title.

        :param j_title: individual job title applicant listed
        :type j_title: str
        :param model: Google Word2Vec pretrained model
        :type model: gensim.keyedvectors.KeyedVectors

        :return: averaged Word2Vec embeddings of all words in job title
        :rtype: np.array
        """
        if j_title == 0:
            return np.zeros(model.vector_size)
        
        words = str(j_title).split()
        cleaned = []

        for w in words:
            t =  re.sub(r'[^\w]', "", w)
            if t.isdigit():
                t = num2words(int(t))
            cleaned.append(t)

        vecs = [model[w] for w in cleaned if w in model]
        if not vecs:
            return np.zeros(model.vector_size)
        
        return np.mean(vecs, axis=0)
    
    @staticmethod
    def encode_job_title(df: pd.DataFrame) -> pd.DataFrame:
        """
        This function sets the average Word2Vec embedding for each job title listed in application
        using the pretrained Google model.
        """
        title_cols = ['Job ' + str(i) + ' Title' for i in range(1,7)]
        missing_cols = ['Job ' + str(i) + ' Missing' for i in range(1,7)]
        path = 'data/reference/GoogleNews-vectors-negative300.bin.gz'
        model = KeyedVectors.load_word2vec_format(path, binary=True)
        job_enc_cols = ['Job ' + str(i) + ' Title Enc' for i in range(1,7)]

        for c in missing_cols:
            df[c] = 0

        for c in title_cols:
            df[c] = df[c].fillna(0)
            df[c] = df[c].str.lower().replace(',', '').replace('.', '')

        for c in job_enc_cols:
            df[c] = None

        mask = (df[title_cols] == 0).all(axis=1)
        df.loc[mask, missing_cols] = 1

        for c_idx, c in enumerate(title_cols):
            enc_col = job_enc_cols[c_idx]
            df[enc_col] = df[c].apply(lambda x: JobDataHelpers.embed_text(x, model))
        
        df = df.drop(columns=title_cols)
        return df
    
    @staticmethod
    def encode_job_org(df: pd.DataFrame) -> pd.DataFrame:
        """
        This function cleans the job organization strings. Later, before feeding into models,
        job organizations will be transformed into target or learned embedding on training data only.
        """
        org_cols = ['Job ' + str(i) + ' Organization' for i in range(1,7)]
        df['Purdue Internal'] = 0

        for c in org_cols:
            df[c] = df[c].fillna(0)
            df[c] = df[c].str.lower()
        df['Purdue Internal'] = df[org_cols].apply(lambda row: row.astype(str).str.contains('purdue')).any(axis=1).astype(int)
        return df
    
    @staticmethod
    def fill_na_dir_reports(df: pd.DataFrame) -> pd.DataFrame:
        """
        This functions fills in np.NaN direct reports for Jobs 1-6 with 0 if not reported.
        """
        dir_rep_cols = ['Job ' + str(i) + ' Direct Reports' for i in range(1,7)]

        for d in dir_rep_cols:
            df[d] = df[d].fillna(0)
        return df
    
# Main Function
def load_application_data(file_path: str) -> pd.DataFrame:
    """
    This function parses the Purdue's online ECE Master's program application data for Fall 2020 - Spring 2024 semesters
    and stores the data into dictionaries based on admission decision. These dictionaries will later be modified to add course
    behavior data, but before pre-processing for various ML models, they will be flatted into pandas dataframes.

    :param file_path: Path to Excel file storing application data
    :type file_path: str

    :return: All valid Master's applicants' application information (skips those who did not finish application)
    :rtype: pd.DataFrame
    """
    # Store application data for semesters Fall 2020 to Spring 2024
    df = pd.read_excel(file_path)
    
    countries = {'Argentina', 'Nepal', 'Netherlands', 'Mexico', 'Croatia', 'Ecuador', 'Israel', 'Brazil', 'Cayman Islands', 'Iceland', 'Palestine', 'Hong Kong S.A.R.', 'Australia', 
        'Ethiopia', 'China', 'Ghana', 'Turkey', 'United States', 'Bangladesh', 'United Arab Emirates', 'Pakistan', 'Slovenia', 'South Korea', 'Rwanda', 'Zambia', 'Panama', 
        'Russia', 'Egypt', 'Saudi Arabia', 'Spain', 'Iran', 'Japan', 'Chile', 'Finland', 'Canada', 'Germany', 'Taiwan', 'France', 'Vietnam', 'Kuwait', 'New Zealand', 'United Kingdom', 
        'Jordan', 'Nigeria', 'Portugal', 'South Africa', 'India', 'MISSING'}
    
    all_lang = {'Japanese', 'Kazakh', 'Sinahalese', 'Icelandic', 'Persian', 'Chinese', 'Farsi', 'Spanish', 'MISSING', 'Hebrew', 'German', 'Arabic', 'Korean', 'English', 
                'Portuguese', 'Slovenian', 'Kannada', 'French'}
    
    app_deadlines = {
            'Fall 2020': datetime(2020, 7, 31, 11, 59, 59),
            'Fall 2021': datetime(2021, 7, 15, 11, 59, 59),
            'Fall 2022': datetime(2022, 7, 15, 11, 59, 59),
            'Fall 2023': datetime(2023, 7, 15, 11, 59, 59),
            'Fall 2024': datetime(2024, 7, 15, 11, 59, 59),
            'Summer 2024': datetime(2024, 12, 15, 11, 59, 59)
        }

    df = MiscDataHelpers.drop_columns(df)
    df = MiscDataHelpers.process_home_location(df)
    df = MiscDataHelpers.process_program_data(df)
    df = MiscDataHelpers.process_decision_history(df)
    df = SchoolDataHelpers.encode_school_level(df)
    df = SchoolDataHelpers.encode_school_tier(df)
    df = SchoolDataHelpers.one_hot_encode_school_lang(df)
    df = SchoolDataHelpers.encode_all_majors(df)
    df = SchoolDataHelpers.encode_all_degree_levels(df)
    df = SchoolDataHelpers.clean_all_gpa(df)
    df = SchoolDataHelpers.compute_last_edit_delta(df, app_deadlines)
    df = SchoolDataHelpers.compute_school_job_duration_recency(df, app_deadlines)
    df = SchoolDataHelpers.fill_na_class_rank_size(df)
    df = SchoolDataHelpers.encode_school_job_country(df)
    df = SchoolDataHelpers.encode_degree_conferred(df, app_deadlines)
    df = JobDataHelpers.fill_na_dir_reports(df)
    df = JobDataHelpers.encode_job_title(df)
    df = JobDataHelpers.encode_job_org(df)
    #df = JobDataHelpers.encode_job_description(df)

    df.to_excel('data/cleaned/processed_admissions.xlsx', index=False)
    return df

def clean_job_descriptions_looped(file_path: str) -> pd.DataFrame:
    """
    This function encodes the job descriptions to a numpy array and PyTorch tensor using the pre-trained BERT model.

    :return: Dataframe with job descriptions embedded for specified rows
    :rtype: pd.DataFrame
    """
    df = pd.read_excel(file_path)
    # desc_cols = ['Job ' + str(i) + ' Description' for i in range(1,7)]
    # np_cols = ['Job ' + str(i) + ' Description (np.array)' for i in range(1,7)]
    # tensor_cols = ['Job ' + str(i) + ' Description (tensor)' for i in range(1,7)]
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased')
    # hidden_size = 768
    # num_apps = len(df)

    # for desc_col, np_col, tensor_col in zip(desc_cols, np_cols, tensor_cols):
    #     df[desc_col].fillna(0)
    #     df[np_col] = [np.zeros(hidden_size, dtype=np.float32) for _ in range(num_apps)]
    #     df[tensor_col] = [torch.zeros(hidden_size) for _ in range(num_apps)]
    #     i = 0
    #     for idx, val in df[desc_col].items():
    #         np_arr, t_torch = JobDataHelpers.clean_one_description(val, tokenizer, model)

    #         if np_arr is not None:
    #             df.at[idx, np_col] = np_arr
    #         if t_torch is not None:
    #             df.at[idx, tensor_col] = t_torch
    #         i += 1
    #         if i == 100:
    #             i = 0
    #             time.sleep(5)
    #     time.sleep(5)
    # df.drop(columns = desc_cols)
    return df

if __name__ == '__main__':
    _ = clean_job_descriptions_looped('data/cleaned/processed_admissions.xlsx')


