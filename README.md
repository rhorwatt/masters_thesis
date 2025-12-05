# Master's Thesis: Early Prediction of Student Attrition and Admissions Decisions

## Folder Structure
/data
    - /cleaned
        - app_to_puid.json: maps application ID to PUID
        - data.parquet: preprocesssed admissions data
    - /reference
        - school_to_rank.json: maps school name mentioned in admissions application to numerical tier for preprocessing of data
        - ace-institutional-classifications.xlsx: from Carnegie Classifications website, used to tier US schools
        - GoogleNews-vectors-negative300.bin.gz: downloaded WordVectors model, used to embed job titles
        - qs_rankings.xlsx: downloaded from QS website, used to tier international schools
        - country.json: maps country name listed in application to HDI, income tier (1-4 from World Bank), and region in world
/model
    - /regression
        - /linear_regression
        - /lasso_regression
    Each model directory includes script to build model, as well as .json with model results with and without feature extraction.
- scripts/
    - load_data.py: script to preprocess admissions data
