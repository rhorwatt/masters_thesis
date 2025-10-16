import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from collections import defaultdict

# Dataclasses for parsing application data
@dataclass
class Location:
    """
    Class to hold location of applicant's address, location of each school applicant attended and location of each job applicant worked at

    Attributes:
    city (str):    City for job applicant worked at; NaN / None for applicant's address and school
    region (str):  State or County (if lives in Indiana) of applicant's address; State / region of school applicant attended; State / region of job applicant worked at
    country (str): Country applicant lives in; country of school applicant attended; country of job applicant worked at
    """
    city: Optional[str] = None
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

        app_status (str):                 Application Status (Awaiting Payment, Awaiting Decision, Decision Released,
                                          Awaiting Submission) - Note: Removing applicants with Awaiting Payment or
                                          Awaiting Submission

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
    app_status: Optional[str] = None
    decision: Optional[str] = None
    schools: List[School] = None
    num_schools: Optional[int] = None
    jobs: List[Job] = None
    num_jobs: Optional[int] = None


# Helper functions
def create_school_list(idx, row, df):
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
                       "School " + str_i + " Language of Instruction", 
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


def create_job_list(idx, row, df):
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


# Global/main function in file
def load_application_data(file_path):
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
             Dictionary mapping application ID to PUID for students who were admitted and accepted their admission
    :rtype: dictionary{k:v where k = application ID, v = application information}, dictionary{k:v where k = application ID, v = application information}, 
            dictionary{k:v where k = application ID, v = application information}, dictionary{k:v where k = application ID, v = application information}, 
            dictionary{k:v where k = application ID, v = PUID}
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

    application_terms = {'Fall 2020'  : 0,
                         'Fall 2021'  : 1,
                         'Fall 2022'  : 2,
                         'Fall 2023'  : 3,
                         'Fall 2024'  : 4,
                         'Summer 2024': 5
                        }

    # Iterate through all applicants in Excel file
    for idx, row in df.iterrows():
        semester.add(row['App - Applicant Term/Year'])
        # Skip students who didn't finish their application
        if (pd.isna(df.loc[idx, 'Decision History (all decisions)'])):
            continue

        # Skip PhD only applications
        if (df.loc[idx, 'App - ECE Degree Objective'] == ("Direct Ph.D. (students without an MS)" or "Ph.D. (students with an MS)")):
            continue

        # Skip students who withdrew / cancelled their application
        if (df.loc[idx, 'Decision History (all decisions)'] == "Withdraw/Cancelled"):
            continue
            
        # Create school list for applicant #idx
        applicant_idx_schools, num_schools_idx = create_school_list(idx, row, df)

        # Create job list for applicant #idx
        applicant_idx_jobs, num_jobs_idx = create_job_list(idx, row, df)

        # Create application location from Location dataclass
        applicant_idx_loc = Location(None, row['State or County of Residence'], row['Active Country'])
        
        # Create applicant #idx application data from Application_Data dataclass
        app_id = row['Application Slate ID']

        applicant_idx = Application_Data(row['Application Slate ID'], 
                                         row['App - Official PUID'], 
                                         row['Age'],
                                         applicant_idx_loc, 
                                         row['App - Citizenship Status'], 
                                         application_terms[row['App - Applicant Term/Year']], 
                                         row['App - Program Choice'],
                                         [row['App - ECE Area of Interest 1'], row['App - ECE Area of Interest 2']], 
                                         row['Application Status'], 
                                         row['Decision History (all decisions)'],
                                         applicant_idx_schools, 
                                         num_schools_idx, 
                                         applicant_idx_jobs, 
                                         num_jobs_idx)

        all_applications[app_id] = applicant_idx

        decision = row['Decision History (all decisions)']

        # Admitted Students
        if (decision in admitted):
            app_to_puid[app_id] = row['App - Official PUID']
            admit_list[app_id] = applicant_idx

        # Attending Students
        if (decision in accepted):
            accepted_list[app_id] = applicant_idx

        # Rejected Students
        elif (decision in rejected):
            rejected_list[app_id] = applicant_idx

    print(f"""Number of Total Applications: {len(all_applications)}\nNumber of Admitted Students: {len(admit_list)}\nNumber of Rejected Students: {len(rejected_list)}\n
    Number of Attending Students: {len(accepted_list)}""")

    return all_applications, admit_list, accepted_list, rejected_list, app_to_puid

if __name__ == '__main__':
    _, _, _, _, _ = load_application_data("data/app_data/Fall20thru24_MSECEOnline_All.xlsx")

    # TODO: Following data is categorical still for each 
    # location: Optional[Location] = None
    # citizen: Optional[str] = None
    # program: Optional[str] = None
    # app_areas: List[str] = field(default_factory=list)
    # app_status: Optional[str] = None
    # decision: Optional[str] = None
    # schools: List[School] = None
    # jobs: List[Job] = None

    # start: Optional[pd.Timestamp] = None
    # end: Optional[pd.Timestamp] = None
    # duration: Optional[pd.Timestamp] = None
    # company: Optional[str] = None
    # location: Optional[Location] = None
    # title: Optional[str] = None
    # description: Optional[str] = None

    # degree_type: Optional[str] = None
    # name: Optional[str] = None
    # ins_lang: Optional[str] = None
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