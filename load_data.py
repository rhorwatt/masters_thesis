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

# Dataclasses for parsing application data
@dataclass
class Location:
    """
    Class to hold location of applicant's address, location of each school applicant attended and location of each job applicant worked at

    Attributes:
    city (str):    City for job applicant worked at; NaN / None for applicant's address and school
    state (str):   State of applicant's address; State of school applicant attended; State of job applicant worked at
    region (str):  Region of applicant's address; Region of school applicant attended; Region of job applicant worked at
    country (str): Country applicant lives in; country of school applicant attended; country of job applicant worked at
    """
    city: Optional[str] = None
    state: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None


@dataclass
class School:
    """
    Class to hold information for school(s) that applicant attended previously

    Attributes:
        degree_type (str):             Level of degree from school (U - Undergraduate, G - Graduate)
        name (str):                    Name of school
        ins_lang (str):                Language of instruction at school
        start (pandas.Timestamp):      Start date of attendance (mm/dd/yyy)
        end (pandas.Timestamp):        End date of attendance (mm/dd/yyyy)
        duration (pandas.Timestamp):   Length of attendance of school in days (End - start)
        major (str):                   Applicant's major at school
        gpa (float):                   Applicant's original GPA at school
        gpa_scale (float):             GPA scale at school
        rank (int):                    Applicant's rank at school (X out of Y)
        degree (str):                  Degree pursued at school (MA, JD, BA, MTech, BTech, MEng, Associates, BEng, MD, PhD, BS, MBA, Postgrad, None, Other, BComm)
        grad_date (pandas.Timestamp):  Date degree conferred (if graduated) (mm/dd/yyyy)
        confirmed (bool):              1 if transcript and attendance confirmed by university; 0 or NaN if not confirmed
        gpa_recalc (float):            GPA recalculated on 4.0 scale
        location (Location dataclass): City, Region, Country of school
        class_size (int):              Size of graduating class at school
        num_apps (int):                For 1st school listed on application, number of active applications from school; NaN / None if not school 1
    """
    degree_type: Optional[str] = None
    name: Optional[str] = None
    ins_lang: Optional[str] = None
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    duration: Optional[pd.Timestamp] = None
    major: Optional[str] = None
    gpa: Optional[float] = None
    gpa_scale: Optional[float] = None
    rank: Optional[int] = None
    degree: Optional[str] = None
    grad_date: Optional[pd.Timestamp] = None
    confirmed: Optional[bool] = None
    gpa_recalc: Optional[float] = None
    location: Optional[Location] = None
    class_size: Optional[int] = None
    num_apps: Optional[int] = None


@dataclass
class Job:
    """
    Class to hold applicant's job history information

    Attributes:
        start (pandas.Timestamp):      Job start date (mm/dd/yyyy)
        end (pandas.Timestamp):        Job end date (mm/dd/yyyy)
        duration (pandas.Timestamp):   Duration worked at job (end - start, listed in days)
        company (str):                 Company Name of Job
        location (Location dataclass): Location of job (City, Region, Country)
        title (str):                   Job title
        reports (int):                 Number of direct reports to applicant in said job
        description (str):             Application job description (usually in bullet points describing their accomplishments and responsibilities)
    """
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    duration: Optional[pd.Timestamp] = None
    company: Optional[str] = None
    location: Optional[Location] = None
    title: Optional[str] = None
    reports: Optional[int] = None
    description: Optional[str] = None


@dataclass
class Application_Data:
    """ 
    Class to hold applicant's application data from Fall 2020 to Spring 2024 semester applications to ECE Online Master's Program

    Attributes:
        app_id (int):                     Application Slate ID (every applicant has this, even if rejected)
        puid (int):                       Purdue Student ID assigned if student admitted
        age (int):                        Applicant's age
        location (Location dataclass):    Holds state/county, country of applicant
        citizen (str):                    US Citizenship Status of Applicant (Permanent Resident Non-US Citizen,
                                          International, US Citizen, N/A, Asylee/Refugee)

        sem (int):                        Semester applicant applied (0 = Fall 2020, 1 = Spring 2021, ...)
        program (str):                    Application Program Choice (N/A, 1st Choice, 2nd Choice, 3rd Choice)

        app_areas (list(str)):            Up to 2 ECE Areas of Interest 
                                          (Power & Energy Systems; Automatic Control; 
                                          Communications, Networking, Signal and Image Processing; 
                                          Computer Engineering; Micro Electronics & Nanotechnology;
                                          Biomedical Engineering; VLSI & Circuit Design; Fields & Optics;
                                          Professional Masters - Innovative Technologies; 
                                          Professional Masters - ECE Innovative Technologies)

        decision (str):                   Application Decision (Denied, Admitted, Withdraw / Cancelled or Application Expired) + the following
                                          applicant decisions if they were admitted (Enrollment Accepted, Enrollment Denied, Change of Term,
                                          Deferred Admission)

        schools (list(School dataclass)): List of schools that applicant attended in order of most to least recent (up to 6)
        num_schools (int):                Number of schools applicant attended
        jobs (list(Job dataclass)):       List of jobs that applicant worked in order of most to least recent (up to 6)
        num_jobs (int):                   Number of jobs applicant worked
    """
    app_id: Optional[int] = None
    puid: Optional[int] = None
    age: Optional[int] = None
    location: Optional[Location] = None
    citizen: Optional[str] = None
    sem: Optional[int] = None
    program: Optional[str] = None
    app_areas: List[str] = field(default_factory=list)
    decision: Optional[str] = None
    schools: List[School] = None
    num_schools: Optional[int] = None
    jobs: List[Job] = None
    num_jobs: Optional[int] = None


# Helper functions
def create_school_list(idx: int, row: pd.core.series.Series, df: pd.DataFrame) -> Tuple[List[School], int]:
    """
    This function creates the list of schools (school_idx_schools) that applicant #idx previously attended.

    :param idx: Row number of applicant in pandas dataframe df containing application data.
    :type idx:  int
    :param row: Data from row #idx of pandas dataframe df containing application data
    :type row:  pandas.core.series.Series
    :param df:  Dataframe containing application data for all applicants
    :type df:   pandas.core.frame.DataFrame

    :return:    Returns list of information for schools applicant #idx previously attended, number of schools applicant previously attended
    :rtype:     list(School dataclass), int
    """
    applicant_idx_schools = []

    # Applicant can list up to 6 schools attended on application
    for i in range(1, 7):
        str_i = str(i)

        school_cols = ["School " + str_i + " Type", 
                       "School " + str_i + " Institution",
                       "School " + str_i + " Language", 
                       "School " + str_i + " From", 
                       "School " + str_i + " To",
                       "School " + str_i + " Major", 
                       "School " + str_i + " GPA", 
                       "School " + str_i + " GPA Scale",
                       "School " + str_i + " Class Rank (Numeric)",
                       "School " + str_i + " Degree", 
                       "School " + str_i + " Degree Conferred", 
                       "School " + str_i + " Confirmed",
                       "School " + str_i + " GPA Converted",  
                       "School " + str_i + " Class Size (Numeric)",
                       "School #" + str_i + " Number of Active Applications"]

        school_loc_cols = ["School " + str_i + " City", 
                           "School " + str_i + " Region", 
                           "School " + str_i + " Country"]

        # If applicant #idx doesn't have a school i, break
        if (pd.isna(df.loc[idx, school_cols[2]])):
            break
        
        # Keep track of school location
        if (pd.isna(df.loc[idx, school_loc_cols[-1]])):
            s_loc = Location(None, None, None)
        else:
            s_loc = Location(row[school_loc_cols[0]], row[school_loc_cols[1]], row[school_loc_cols[2]])

        # 1st School has extra data field: Number of applications from that school
        if (i == 1):
            school_i = School(row[school_cols[0]], row[school_cols[1]], row[school_cols[2]], row[school_cols[3]], row[school_cols[4]],
            row[school_cols[4]] - row[school_cols[3]], row[school_cols[5]], row[school_cols[6]], row[school_cols[7]], row[school_cols[8]], row[school_cols[9]], row[school_cols[10]], 
            row[school_cols[11]], row[school_cols[12]], s_loc, row[school_cols[13]], row[school_cols[14]])
        else:
            school_i = School(row[school_cols[0]], row[school_cols[1]], row[school_cols[2]], row[school_cols[3]], row[school_cols[4]], 
            row[school_cols[4]] - row[school_cols[3]], row[school_cols[5]], row[school_cols[6]], row[school_cols[7]], row[school_cols[8]], row[school_cols[9]], row[school_cols[10]], 
            row[school_cols[11]], row[school_cols[12]], s_loc, row[school_cols[13]], None)
        
        applicant_idx_schools.append(school_i)
    return applicant_idx_schools, len(applicant_idx_schools)


def create_job_list(idx: int, row: pd.core.series.Series, df:pd.DataFrame) -> Tuple[List[Job], int]:
    """
    This function creates the list of information about jobs the applicant previously worked at.

    :param idx: Row number of applicant in pandas dataframe df containing application data.
    :type idx:  int
    :param row: Data from row #idx of pandas dataframe df containing application data
    :type row:  pandas.core.series.Series
    :param df:  Dataframe containing application data for all applicants
    :type df:   pandas.core.frame.DataFrame

    :return:    Returns list of information for jobs applicant #idx previously worked at, number of jobs applicant previously worked
    :rtype:     list(Job dataclass), int
    """
    applicant_idx_jobs = []

    # Applicant can list up to 6 jobs worked on application
    for j in range(1, 7):
        str_j = str(j)

        job_cols = ["Job " + str_j + " From", 
                    "Job " + str_j + " To", 
                    "Job " + str_j + " Organization",
                    "Job " + str_j + " City", 
                    "Job " + str_j + " Region", 
                    "Job " + str_j + " Country",
                    "Job " + str_j + " Title", 
                    "Job " + str_j + " Direct Reports", 
                    "Job " + str_j + " Description"]
    
        # If no job title listed, assuming that applicant #idx doesn't have a job j to list
        if (pd.isna(df.loc[idx, job_cols[6]])):
            break
        
        #Store job location
        job_loc = Location(row[job_cols[3]], row[job_cols[4]], row[job_cols[5]])

        job_j = Job(row[job_cols[0]], row[job_cols[1]], row[job_cols[1]] - row[job_cols[0]], row[job_cols[2]], job_loc, row[job_cols[6]], row[job_cols[7]], row[job_cols[8]])
        applicant_idx_jobs.append(job_j)
    return applicant_idx_jobs, len(applicant_idx_jobs)


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


def parse_rx_school_status() -> pd.DataFrame:
    """
    This function reads the R1, R2, R3 status of US schools from csv downloaded from: https://carnegieclassifications.acenet.edu/.
    CSV is stored in pandas DataFrame. The schools names are normalized, and the research activity designation is shortened.

    :return: cleaned pandas DataFrame for research designation (includes 3000+ US schools from https://carnegieclassifications.acenet.edu/)
    :rtype: pd.DataFrame
    """
    df2 = pd.read_csv('ace-institutional-classifications.csv', usecols=['Research Activity Designation', 'name'])
    df2['Research Activity Designation'] = df2['Research Activity Designation'].replace('Research 2: High Research Spending and Doctorate Production', 2)
    df2['Research Activity Designation'] = df2['Research Activity Designation'].replace('Research Colleges and Universities', 3)
    df2['Research Activity Designation'] = df2['Research Activity Designation'].replace('Research 1: Very High Research Spending and Doctorate Production', 1)
    df2['name'] = df2['name'].str.lower()
    return df2


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


# Global/main function in file
def load_application_data(file_path: str):
    """
    This function parses the Purdue's online ECE Master's program application data for Fall 2020 - Spring 2024 semesters
    and stores the data into dictionaries based on admission decision. These dictionaries will later be modified to add course
    behavior data, but before pre-processing for various ML models, they will be flatted into pandas dataframes.

    :param file_path: Path to Excel file storing application data
    :type file_path: str

    :return: All valid Master's applicants' application information (skips those who did not finish application),
             Application information for all applicants who were admitted (including those who do not choose to attend)
             Application information for all applicants who were admitted and accepted their admission (includes those who deferred, changed terms, initially declied admission)
             Application information for all applicants who were rejected,
             Dictionary mapping App ID to PUID for students who were admitted and accepted their admission
    :rtype: dictionary{k:v where k = App ID, v = application information}, dictionary{k:v where k = App ID, v = application information}, 
            dictionary{k:v where k = App ID, v = application information}, dictionary{k:v where k = App ID, v = application information}, 
            dictionary{k:v where k = App ID, v = PUID}
    """
    # Store application data for semesters Fall 2020 to Spring 2024
    df = pd.read_excel(file_path)
    all_applications = defaultdict(Application_Data)
    rejected_list = defaultdict(Application_Data)
    accepted_list = defaultdict(Application_Data)
    admit_list = defaultdict(Application_Data)
    app_to_puid = defaultdict(int)

    """
    Possible Admission Decision History for each group: 
    1) admitted: those who were admitted to university and accepted, denied, deferred, changed term
    2) rejected: those who were denied from university or whose application expird
    3) accepted: those who were admitted to university and accepted, deferred or changed term
    """
    admitted = {'Admitted, Change of Term', 
                'Admitted, Deferred Admission', 
                'Admitted, Enrollment Declined, Enrollment Accepted', 
                'Admitted, Enrollment Declined', 
                'Admitted, Enrollment Accepted, Enrollment Declined', 
                'Admitted, Enrollment Accepted', 
                'Admitted, Enrollment Accepted, Deferred Admission, Enrollment Declined',
                'Admitted, Enrollment Accepted, Change of Term', 
                'Deferred Admission', 'Admitted, Enrollment Accepted, Deferred Admission', 
                'Admitted'
                }

    rejected = {'Application Expired', 
                'Denied'
                }

    accepted = {'Admitted, Change of Term', 
                'Admitted, Deferred Admission', 
                'Admitted, Enrollment Declined, Enrollment Accepted',
                'Admitted, Enrollment Accepted', 
                'Admitted, Enrollment Accepted, Change of Term', 
                'Deferred Admission', 
                'Admitted, Enrollment Accepted, Deferred Admission'
                }

    # Drop all columns full of NaN
    df = df.dropna(axis=1, how='all')

    # Drop School Code columns bc not useful
    df = df[df.columns.drop(list(df.filter(regex=r'School [1-6] Code')))]

    # Rename columns
    df = df.rename(columns={'Application Slate ID': 'App ID', 'App - Official PUID': 'PUID', 'State or County of Residence':'State', 'Active Country':'Continent', 
                            'App - Citizenship Status': 'Citizenship', 'App - Applicant Term/Year':'App Term', 'App - Program Choice':'Program Choice', 
                            'App - ECE Degree Objective':'Degree Objective', 'App - ECE Area of Interest 1':'ECE Area of Interest 1', 'App - ECE Area of Interest 2':'ECE Area of Interest 2',
                            'Decision History (all decisions)':'Decision History', 'School 1 Language of Instruction':'School 1 Language', 'School 2 Language of Instruction':'School 2 Language',
                            'School 3 Language of Instruction':'School 3 Language', 'School 4 Language of Instruction':'School 4 Language', 'School 5 Language of Instruction':'School 5 Language',
                            'School 6 Language of Instruction':'School 6 Language'})
    print(list(df.columns))

    # Drop people who said they were 0, 5, 6 years old (3 entries)
    df = df[df['Age'] >= 18]

    state_mapping={np.nan: '0', "Alabama":'1', "Alaska":'2', "Arizona":'3', "Arkansas":'4', "California":'5', "Colorado":'6', "Connecticut":'7', "Delaware":'8', "Florida":'9', 
                   "Georgia":'10', "Hawaii":'11', "Idaho":'12', "Illinois":'13', "Indiana":'14', "Iowa":'15', "Kansas":'16', "Kentucky":'17', "Louisiana":'18', "Maine":'19', 
                   "Maryland":'20', "Massachusetts":'21', "Michigan":'22', "Minnesota":'23', "Mississippi":'24', "Missouri":'25', "Montana":'26', "Nebraska":'27', "Nevada":'28', 
                   "New Hampshire":'29', "New Jersey":'30', "New Mexico":'31', "New York":'32', "North Carolina":'33', "North Dakota":'34', "Ohio":'35', "Oklahoma":'36', "Oregon":'37', 
                   "Pennsylvania":'38', "Rhode Island":'39', "South Carolina":'40', "South Dakota":'41', "Tennessee":'42', "Texas":'43', "Utah":'44', "Vermont":'45', "Virginia":'46', 
                   "Washington":'47', "West Virginia":'48', "Wisconsin":'49', "Wyoming":'50', 'Dist Of Columbia':'51', "Puerto Rico":'52', "Residents Abroad":'53'}
    
    # Northeast = 1, Midwest = 2, South = 3, West = 4, US Territory = 5, Outside US = 6
    inverse_region_mapping = {'0':[np.nan],
                              '1':["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont", "New Jersey", "New York", "Pennsylvania", "Delaware", "Maryland"],
                              '2':["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin", "Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska", "North Dakota", "South Dakota"],
                              '3':['Dist Of Columbia', "Florida", "Georgia", "North Carolina", "South Carolina", "Virginia", "West Virginia", "Alabama", "Kentucky", "Mississippi", "Tennessee",
                                 "Arkansas", "Louisiana", "Oklahoma", "Texas"],
                              '4':["Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming", "Alaska", "California", "Hawaii", "Oregon", "Washington"],
                              '5':["Puerto Rico"],
                              '6':["Residents Abroad"]}
    region_mapping = {v:k for k,v_list in inverse_region_mapping.items() for v in v_list}

    # Map state to nominal label
    df['State'] = df['State'].replace(to_replace=r'^[A-Za-z ]*IN$', value="Indiana", regex=True)
    copy_state_series = df['State'].copy()
    df['Region of Residence'] = copy_state_series.map(region_mapping).astype(int)
    df['State'] = df['State'].replace(state_mapping).astype(int)

    # Map Country of Residence to Continent
    # 0: Other / Unknown
    # 1: North America
    # 2: South America
    # 3: Europe
    # 4: Asia
    # 5: Africa
    # 6: Australia
    reverse_continent_mapping = {'0':[np.nan],
                                 '1':['United States', 'Panama', 'Cayman Islands', 'Canada', 'Mexico'],
                                 '2':['Ecuador','Argentina', 'Chile', 'Brazil'],
                                 '3':['Croatia', 'Netherlands', 'Iceland', 'Slovenia', 'France', 'United Kingdom', 'Portugal', 'Russia', 'Finland', 'Spain', 'Germany'],
                                 '4':['Nepal', 'Israel', 'South Korea', 'Japan', 'Turkey', 'United Arab Emirates', 'Pakistan', 'Saudi Arabia', 'Palestine', 'Jordan', 'Iran', 'Taiwan', 'Bangladesh',
                                    'India', 'Vietnam', 'China', 'Hong Kong S.A.R.', 'Kuwait'],
                                 '5':['South Africa', 'Zambia', 'Nigeria', 'Ethiopia', 'Congo (Kinshasa)', 'Rwanda', 'Ghana', 'Egypt'],
                                 '6':['New Zealand', 'Australia']}
    continent_mapping = {v:k for k,v_list in reverse_continent_mapping.items() for v in v_list}
    df['Continent'] = df['Continent'].replace(continent_mapping).astype(int)

    # Categorize Citizenship
    citizenship_mapping = {'None of the Above':'0', np.nan:'0', 'Asylee or Refugee':'1', 'Permanent Resident Non-US Ctzn':'2', 'U.S. Citizen':'3', 'International':'4'}
    df['Citizenship'] = df['Citizenship'].replace(citizenship_mapping).astype(int)

    # Categorize App Term - maybe remove later
    application_terms = {'Fall 2020'  : '0',
                         'Fall 2021'  : '0',
                         'Fall 2022'  : '0',
                         'Fall 2023'  : '0',
                         'Fall 2024'  : '0',
                         'Summer 2024': '1'}
    df['App Term'] = df['App Term'].replace(application_terms).astype(int)

    # Numerical Encoding of Program Choice
    program_choice_mapping = {'Third Choice':'3', 'Second Choice':'2', np.nan:'0', 'First Choice':'1'}
    df['Program Choice'] = df['Program Choice'].replace(program_choice_mapping).astype(int)

    # Drop applications with no degree objective listed
    df = df.dropna(subset = ['Degree Objective'])

    # Drop PhD only applications
    df = df[~df['Degree Objective'].str.startswith("Ph.D.", na=False)]
    df = df[~df['Degree Objective'].str.startswith("Direct Ph.D.", na=False)]

    # Categorize Degree Objective
    degree_obj_mapping = {'MS/Ph.D.':'0', 'Professional Masters - Innovative Technologies':'1', 'Masters':'2'}
    df['Degree Objective'] = df['Degree Objective'].replace(degree_obj_mapping).astype(int)

    # Categorize ECE Area of Interest 1 & 2
    ece_areas = {'Professional Masters – ECE Innovative Technologies':'0', 'Micro Electronics and Nanotechnology':'1', 'Automatic Control':'2', 'Computer Engineering':'3', 
                 'Communications Networking Signal and Image Processing':'4', 'VLSI and Circuit Design':'5', np.nan:'6', 'Fields and Optics':'7', 'Power and Energy Systems':8, 
                 'Biomedical Engineering':'9', 'Professional Masters - Innovative Technologies':'0'}
    df['ECE Area of Interest 1'] = df['ECE Area of Interest 1'].replace(ece_areas).astype(int)
    df['ECE Area of Interest 2'] = df['ECE Area of Interest 2'].replace(ece_areas).astype(int)

    # Drop Application Status - don't need it since already have decison history
    df = df[df['Application Status'] != 'Awaiting Submission']
    df = df[df['Application Status'] != 'Awaiting Payment']
    df = df.drop(columns=['Application Status'], axis=1)

    # Drop incomplete applications
    df = df.dropna(subset = ['Decision History'])
    df = df[~df['Decision History'].str.startswith("Awaiting", na=False)]

    # Drop withdrawn / cancelled applications
    df = df[~df['Decision History'].str.startswith("Withdraw", na=False)]

    # Replace Application Expired with Rejection
    df['Decision History'] = df['Decision History'].replace('Application Expired', 'Denied')

    # Merge Decision History Categories
    df['Decision History'] = df['Decision History'].replace('Admitted, Change of Term', 'Admitted, Enrollment Accepted, Change of Term')
    df['Decision History'] = df['Decision History'].replace('Enrollment Declined', 'Admitted, Enrollment Declined')
    df['Decision History'] = df['Decision History'].replace(to_replace=r'^[A-Za-z\_\-, ]*Enrollment Declined$', value='Enrollment Declined', regex=True)
    df['Decision History'] = df['Decision History'].replace('Admitted, Deferred Admission', 'Admitted, Enrollment Accepted, Deferred Admission')

    decision_mapping = {'Admitted, Enrollment Accepted':'0', 'Admitted, Enrollment Accepted, Change of Term':'1', 'Admitted':'2', 'Admitted, Enrollment Declined, Enrollment Accepted':'3', 
                        'Enrollment Declined':'4', 'Denied':'5', 'Admitted, Enrollment Accepted, Deferred Admission':'6'}
    admitted_mapping = {'Admitted, Enrollment Accepted':'1', 'Admitted, Enrollment Accepted, Change of Term':'1', 'Admitted':'1', 'Admitted, Enrollment Declined, Enrollment Accepted':'1', 
                        'Enrollment Declined':'1', 'Denied':'0', 'Admitted, Enrollment Accepted, Deferred Admission':'1'}
    enrolled_mapping = {'Admitted, Enrollment Accepted':'1', 'Admitted, Enrollment Accepted, Change of Term':'1', 'Admitted':'0', 'Admitted, Enrollment Declined, Enrollment Accepted':'1', 
                        'Enrollment Declined':'0', 'Denied':'0', 'Admitted, Enrollment Accepted, Deferred Admission':'1'}
    
    copy_decision = df['Decision History'].copy()
    df['Decision History'] = df['Decision History'].replace(decision_mapping).astype(int) # Takes into account if they deferred or changed term
    df['Admitted (Binary)'] = copy_decision.map(admitted_mapping).astype(int)
    df['Enrolled (Binary)'] = copy_decision.map(enrolled_mapping).astype(int)

    # School Type Categorization
    school_types = {'G':'2', np.nan:'0', 'U':'1'}

    for i in range(1, 7):
        type_i_title = "School "  + str(i) + " Type"
        name_i_title = "School " + str(i) + " Institution"
        df[type_i_title] = df[type_i_title].replace(school_types).astype(int)
        df[name_i_title] = df[name_i_title].apply(normalize_school_name)
        df[name_i_title] = df[name_i_title].replace(to_replace=r'^[A-Za-z\-,\(\)\/ ]*community[A-Za-z\-\(\)\/ ]*$', value='community', regex=True)

    cols = [col for col in df.columns if re.match(r'^School [1-6] Institution$', col)]
    all_schools = set()

    for col in cols:
        all_schools.update(df[col])

    purdue_branches = ['purdue university calumet hammond in', 'purdue university remote delco electronics kokomo', 'purdue university fort wayne', 'purdue university global', 
                       'purdue university statewide tech in', 'indiana purdue university/indpls']
    all_purdue = set(['purdue university calumet hammond in', 'purdue university remote delco electronics kokomo', 'purdue university fort wayne', 'purdue unversity', 'purduex 69503x', 
                  'purdue', 'indiana university purdue university indianapolis', 'purduex 69501x', 'purdue university west lafayette', 'purdue university', 'purdue university global', 
                  'purdue university west lafayette*', 'purdue university west lafayette in', 'purdue university (simplilearn)', 'purdue university statewide tech in', 'indiana purdue university/indpls'])
    purdue_main = ['purdue unversity', 'purdue', 'purdue university west lafayette', 'purdue university', 'purdue university west lafayette*', 'purdue university west lafayette in']

    for col in cols:
        copy_schools = df[col].copy()
        df[col] = df[col].replace(purdue_branches, 'purdue branch')
        df[col] = df[col].replace(purdue_main, 'purdue')
        df['Purdue (Binary)'] = copy_schools.apply(lambda x: 1 if x in all_purdue else 0) # Added variable to check if they attended Purdue or Purdue branch campus previously

    df2 = parse_rx_school_status()
    for col in cols:
        df = map_rx_school_status(df, df2, col)

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
    with open("school_to_rank.json", "r", encoding="utf-8") as f:
        school_to_rank = json.load(f)

    count, non_count = 0, 0
    unmapped = set()
    for col in cols:
        for idx, s in df[col].items():
            if (not isinstance(s, int)) and (s in school_to_rank):
                count += 1
                df.at[idx, col] = school_to_rank[s]
            elif s == "Unknown" or s == "no response" or "sean greene":
                df.at[idx, col] = 0
                count += 1
            else:
                non_count += 1
                if (not isinstance(s, int)):
                    unmapped.add(s)
            # else:
            #     if (not isinstance(s, int)):
            #         print(s)
    for s in unmapped:
        print(s)
    print(len(unmapped))

    # Merge repeat languages
    df['School 1 Language'] = df['School 1 Language'].replace(to_replace=r'^Chinese[A-Za-z\_\- ]*$', value='Chinese', regex=True)
    df['School 1 Language'] = df['School 1 Language'].replace(to_replace=r'^[A-Za-z\_\-\(\) ]*Farsi\)*$', value='Persian', regex=True)

    # Clean up school majors
    df['School 1 Major'] = df['School 1 Major'].str.lower()
    # TODO: Remove degree level (Bachelor, Master etc) from major bc already in separate variable
        # some say bachelor at end, some at beginning or in () or after ,
        # bs, bachelor's, bachelor of, master of 
        # remove words "focus in"
        # remove "graduate certificate"
    # TODO: Group together similar degree names
    # TODO: fix typo - noticed "compter engineering", "engineernig", "informatino"

    # Fill NaN GPA's with 0 - they might not have school information
    df['School 1 GPA'] = df['School 1 GPA'].fillna(0)

    # TODO: fix NaN values for GPA scale

    # Note School 1 Class Rank X out of Y column not listed for any applicants

    # School 1 Degree NaN filled with 0, want to be separate category than no degree obtained bc this shows they have 0 school attendance which is slightly different
    df['School 1 Degree'] = df['School 1 Degree'].fillna(0)

    # Note: School x Confirmed category only useful for admitted & enrolled students
    # TODO: Need to fill in converted GPA category once figure out NaN values for GPA scale

    # TODO: see if need to combine school cities
    # TODO: what to do with NaN for school country

    # TODO: Fuzzy wuzzy on school 2

    # Merge School 2 Languagaes
    df['School 2 Language'] = df['School 2 Language'].replace(to_replace=r'^Chinese[A-Za-z\_\- ]*$', value='Chinese', regex=True)
    df['School 2 Language'] = df['School 2 Language'].replace(to_replace=r'^[A-Za-z\_\-\(\) ]*Farsi\)*$', value='Persian', regex=True)
    df['School 2 Language'] = df['School 2 Language'].fillna(0)

    # Leaving out school x created and updated timestamps bc it overcomplicates model currently
    # but could potentially show how long they worked on their application

    # School 3 Data Cleaning
    df['School 3 Type'] = df['School 3 Type'].fillna(0)
    
    # Merge School 3 Language
    df['School 3 Language'] = df['School 3 Language'].replace(to_replace=r'^Chinese[A-Za-z\_\- ]*$', value='Chinese', regex=True)
    df['School 3 Language'] = df['School 3 Language'].fillna(0)

    # School 4 Data Cleaning
    df['School 4 Type'] = df['School 4 Type'].fillna(0)
    df['School 4 Language'] = df['School 4 Language'].fillna(0)

    # Iterate through all applicants in Excel file
    for idx, row in df.iterrows():
        # Create school list for applicant #idx
        applicant_idx_schools, num_schools_idx = create_school_list(idx, row, df)

        # Create job list for applicant #idx
        applicant_idx_jobs, num_jobs_idx = create_job_list(idx, row, df)

        # Create application location from Location dataclass
        applicant_idx_loc = Location(None, row['State'], row['Continent'])
        
        # Create applicant #idx application data from Application_Data dataclass
        app_id = row['App ID']

        applicant_idx = Application_Data(row['App ID'], 
                                         row['PUID'], 
                                         row['Age'],
                                         applicant_idx_loc, 
                                         row['Citizenship'], 
                                         row['App Term'], 
                                         row['Program Choice'],
                                         [row['ECE Area of Interest 1'], 
                                         row['ECE Area of Interest 2']],  
                                         row['Decision History'],
                                         applicant_idx_schools, 
                                         num_schools_idx, 
                                         applicant_idx_jobs, 
                                         num_jobs_idx)

        all_applications[app_id] = applicant_idx

        decision = row['Decision History']

        # Admitted Students
        if (decision in admitted):
            app_to_puid[app_id] = row['PUID']
            admit_list[app_id] = applicant_idx

        # Attending Students
        if (decision in accepted):
            accepted_list[app_id] = applicant_idx

        # Rejected Students
        elif (decision in rejected):
            rejected_list[app_id] = applicant_idx

    return all_applications, admit_list, accepted_list, rejected_list, app_to_puid

if __name__ == '__main__':
    _, _, _, _, _ = load_application_data("data/app_data/Fall20thru24_MSECEOnline_All.xlsx")
   
    # intl_schools = []
    # fp = open('schools_left.txt','r')
    # lines = fp.readlines()
    # for l in lines:
    #     name = l.replace('\t', '')
    #     name = name.replace('\n', '')
    #     intl_schools.append(name)
    # fp.close()

    # TODO: Following data is categorical still for each 
    # location: Optional[Location] = None
    # schools: List[School] = None - Make all schools the same
    # jobs: List[Job] = None

    # start: Optional[pd.Timestamp] = None
    # end: Optional[pd.Timestamp] = None
    # duration: Optional[pd.Timestamp] = None
    # company: Optional[str] = None
    # location: Optional[Location] = None
    # title: Optional[str] = None
    # description: Optional[str] = None

    # name: Optional[str] = None
    # start: Optional[pd.Timestamp] = None
    # end: Optional[pd.Timestamp] = None
    # duration: Optional[pd.Timestamp] = None
    # major: Optional[str] = None
    # degree: Optional[str] = None
    # grad_date: Optional[pd.Timestamp] = None
    # location: Optional[Location] = None

    # city: Optional[str] = None
    # region: Optional[str] = None
    # country: Optional[str] = None
