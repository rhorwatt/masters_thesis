# Master's Thesis: Early Prediction of Student Attrition and Admissions Decisions

## Folder Structure

/data  
&nbsp;&nbsp;&nbsp;&nbsp;/cleaned  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- app_to_puid.json: maps application ID to PUID  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- data.parquet: preprocessed admissions data  

&nbsp;&nbsp;&nbsp;&nbsp;/reference  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- school_to_rank.json: maps school name to numerical tier  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- ace-institutional-classifications.xlsx: Carnegie Classifications data  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- GoogleNews-vectors-negative300.bin.gz: Word2Vec model for job title embeddings  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- qs_rankings.xlsx: QS world university rankings  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- country.json: maps country to HDI, income tier, region  

/model  
&nbsp;&nbsp;&nbsp;&nbsp;/regression  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/linear_regression  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/lasso_regression  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Each model directory includes build scripts and `.json` result files  

/scripts  
&nbsp;&nbsp;&nbsp;&nbsp;- load_data.py: preprocesses admissions data
