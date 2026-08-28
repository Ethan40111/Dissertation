#Data cleaning to be used before fitting a model
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
from sklearn.preprocessing import LabelEncoder

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build path relative to script location
csv_path = os.path.join(script_dir, '..', 'Resources', 'Overall.csv')
df = pd.read_csv(csv_path, encoding='latin-1')
df = df.loc[:, df.nunique() > 1]


#Set values above thresholds to NA
#errors.coerce = NA already converts non numeric values (ie value = 'Unknown') to NA
numeric_columns = ['Total number of benign/borderline tumors for patient',
                   'Record number recode',
                   'Total number of in situ/malignant tumors for patient',
                   'Regional nodes positive (1988+)',
                   'Regional nodes examined (1988+)',
                   'Survival months',
                   'Tumor Size Summary (2016+)',
                   'Tumor Size Over Time Recode (1988+)']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

#Set unknown values/encodings to 0
df.loc[df['Tumor Size Summary (2016+)'] > 988, 'Tumor Size Summary (2016+)'] = pd.NA
df.loc[df['Survival months'] > 9998, 'Survival months'] = pd.NA
df.loc[df['Regional nodes examined (1988+)'] > 89, 'Regional nodes examined (1988+)'] = pd.NA
df.loc[df['Regional nodes positive (1988+)'] > 89, 'Regional nodes positive (1988+)'] = pd.NA
df.loc[df['Total number of in situ/malignant tumors for patient'] > 98, 'Total number of in situ/malignant tumors for patient'] = pd.NA
df.loc[df['Race recode (W, B, AI, API)'] =='Unknown', 'Race recode (W, B, AI, API)'] = pd.NA
df.loc[df['Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)'] =='Non-Hispanic Unknown Race', 'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)'] = pd.NA
df.loc[df['Marital status at diagnosis'] =='Unknown', 'Marital status at diagnosis'] = pd.NA
df.loc[df['Rural-Urban Continuum Code'] =='Unknown/missing/no match (Alaska or Hawaii - Entire State)', 'Rural-Urban Continuum Code'] = pd.NA
df.loc[df['Rural-Urban Continuum Code'] =='Unknown/missing/no match/Not 1990-2023', 'Rural-Urban Continuum Code'] = pd.NA
df.loc[df['AYA site recode 2020 Revision'] =='Unclassified', 'AYA site recode 2020 Revision'] = pd.NA
df.loc[df['Grade Recode (thru 2017)'] =='Unknown', 'Grade Recode (thru 2017)'] = pd.NA
df.loc[df['Grade Recode (thru 2017)'] =='Blank(s)', 'Grade Recode (thru 2017)'] = pd.NA
df.loc[df['Derived Summary Grade 2018 (2018+)'] =='Blank(s)', 'Derived Summary Grade 2018 (2018+)'] = pd.NA
df.loc[df['Grade Pathological (2018+)'] =='Blank(s)', 'Grade Pathological (2018+)'] = pd.NA
df.loc[df['Grade Pathological (2018+)'] =='Unknown', 'Grade Pathological (2018+)'] = pd.NA
df.loc[df['Schema ID (2018+)'] =='Blank(s)', 'Schema ID (2018+)'] = pd.NA
df.loc[df['Site recode - rare tumors'] =='69 Not Classified', 'Site recode - rare tumors'] = pd.NA
df.loc[df['Tumor Size Over Time Recode (1988+)'] =='Unknown or size unreasonable (includes any tumor sizes 401-989)', 'Tumor Size Over Time Recode (1988+)'] = pd.NA

df.loc[df['SEER Combined Mets at DX-bone (2010+)'] =='Blank(s)', 'SEER Combined Mets at DX-bone (2010+)'] = pd.NA
df.loc[df['SEER Combined Mets at DX-bone (2010+)'] =='Unknown', 'SEER Combined Mets at DX-bone (2010+)'] = pd.NA
df.loc[df['SEER Combined Mets at DX-brain (2010+)'] =='Blank(s)', 'SEER Combined Mets at DX-brain (2010+)'] = pd.NA
df.loc[df['SEER Combined Mets at DX-brain (2010+)'] =='Unknown', 'SEER Combined Mets at DX-brain (2010+)'] = pd.NA
df.loc[df['SEER Combined Mets at DX-liver (2010+)'] =='Blank(s)', 'SEER Combined Mets at DX-liver (2010+)'] = pd.NA
df.loc[df['SEER Combined Mets at DX-liver (2010+)'] =='Unknown', 'SEER Combined Mets at DX-liver (2010+)'] = pd.NA
df.loc[df['SEER Combined Mets at DX-lung (2010+)'] =='Blank(s)', 'SEER Combined Mets at DX-lung (2010+)'] = pd.NA
df.loc[df['SEER Combined Mets at DX-lung (2010+)'] =='Unknown', 'SEER Combined Mets at DX-lung (2010+)'] = pd.NA
df.loc[df['Mets at DX-Distant LN (2016+)'] =='Blank(s)', 'Mets at DX-Distant LN (2016+)'] = pd.NA
df.loc[df['Mets at DX-Distant LN (2016+)'] =='Unknown', 'Mets at DX-Distant LN (2016+)'] = pd.NA
df.loc[df['Mets at DX-Other (2016+)'] =='Blank(s)', 'Mets at DX-Other (2016+)'] = pd.NA
df.loc[df['Mets at DX-Other (2016+)'] =='Unknown', 'Mets at DX-Other (2016+)'] = pd.NA
df.loc[df['Grade Clinical (2018+)'] =='Blank(s)', 'Grade Clinical (2018+)'] = pd.NA
df.loc[df['Diagnostic Confirmation'] =='Unknown', 'Diagnostic Confirmation'] = pd.NA
df.loc[df['SEER Combined Mets at DX-bone (2010+)'] =='2', 'SEER Combined Mets at DX-bone (2010+)'] = pd.NA
#replace Age bins column with mid point age
age_bins = df['Age recode with <1 year olds and 90+'].astype(str)
lo = age_bins.str.extract(r'(\d{1,3})')[0].astype(float)
hi = age_bins.str.extract(r'-(\d{1,3})')[0].astype(float)
is_plus = age_bins.str.contains(r'\d+\+')
#treat 90+ as 90-94
hi = hi.where(~is_plus,lo+4)
new_col = (lo+hi)/2.0
df['Age recode with <1 year olds and 90+'] = (lo+hi)/2.0


df.loc[df['Median household income inflation adj to 2023'] =='Unknown/missing/no match/Not 1990-2023', 'Median household income inflation adj to 2023'] = pd.NA
#replace bins in Median household income column with midpoints
median_income_bins = df['Median household income inflation adj to 2023'].astype(str)

# Remove $ and commas first
median_income_bins = median_income_bins.str.replace('$', '').str.replace(',', '')
print(median_income_bins.value_counts()) #works
lo = median_income_bins.str.extract(r'(\d{1,6})')[0].astype(float)
print(lo.value_counts())
is_less = median_income_bins.str.contains(r'<')
lo = lo.where(~is_less, pd.NA)
hi = median_income_bins.str.extract(r'-\s*(\d+)')[0].astype(float)
print(hi.value_counts())
is_plus = median_income_bins.str.contains(r'\d+\+')
hi = hi.where(~is_plus,lo+4000)
print(hi.value_counts())
new_col = (lo+hi)/2.0
print(new_col.value_counts())
df['Median household income inflation adj to 2023'] = (lo+hi)/2.0

#convert time column to numeric
df['Survival months'] = pd.to_numeric(df['Survival months'],errors="coerce")

#convert event column to numeric
#.eq("Dead") means we replace Dead with 1
df['Vital status recode (study cutoff used)'] = df['Vital status recode (study cutoff used)'].eq("Dead")

#drop rows with missing y_time and event values
mask = df['Survival months'].notna() & df['Vital status recode (study cutoff used)'].notna()
#only keep rows where mask is True
df = df.loc[mask].reset_index(drop=True)

#drop redundant column (duplicate of primary site) and columns with no impact on survival
df = df.drop(columns=['Histologic Type ICD-O-3',"PRCDA 2020","Record number recode","Total number of benign/borderline tumors for patient","Primary Site",'Sentinel Lymph Nodes Examined (2018+)','Sentinel Lymph Nodes Positive (2018+)'])

#Drop columns containing AJCC, TNM, EOD - we want our own staging system
staging_keywords = ['AJCC', 'TNM', 'EOD', 'CS','Stage',"ICD-O-3"]
cols_to_drop = [col for col in df.columns 
                if any(keyword in col for keyword in staging_keywords)]

print(f"dropping {cols_to_drop}")
df = df.drop(columns=cols_to_drop)

csv_path = os.path.join(script_dir, '..', 'Resources', 'Overall_cleaned.csv')
df.to_csv(csv_path, index=False)