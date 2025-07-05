# Mexico covid eda 

This is an eda analysis carried out on a dataset for mexico covid 

 The following dataset is used: [COVID-19 Patient](https://www.kaggle.com/datasets/meirnizri/covid19-dataset/data)

The dataset was provided by the Mexican government, and contains the following information:
- usmer: Indicates whether the patient treated medical units of the first, second or third level
- medical_unit: Type of institution of the National Health System that provided the care
- sex: 1 - female; 2 - male
- patient_type: Type of care the patient received in the unit. 1 for returned home and 2 for hospitalization
- date_died: If the patient died indicate the date of death, and 9999-99-99 otherwise
- intubed: Whether the patient was connected to the ventilator
- pneumonia: Whether the patient already have air sacs inflammation or not
- age: Patient's age
- pregnant: Whether the patient is pregnant or not
- diabetes: Whether the patient has diabetes or not
- copd: Whether the patient has Chronic obstructive pulmonary disease or not
- asthma: Whether the patient has asthma or not
- inmsupr: Whether the patient is immunosuppressed or not
- hipertension: Whether the patient has hypertension or not
- other_disease: Whether the patient has other disease or not
- cardiovascular: Whether the patient has heart or blood vessels related disease
- obesity: Whether the patient is obese or not
- renal_chronic: Whether the patient has chronic renal disease or not
- tobacco: Whether the patient is a tobacco user
- clasiffication_final: Covid test results. Values 1-3 mean that the patient was diagnosed with covid in different degrees. 4 or higher means that the patient is not a carrier of covid or that the test is inconclusive.
- icu: whether the patient had been admitted to an Intensive Care Unit

*NOTE*: In the boolean features, 1 means "yes" and 2 means "no". values as 97 and 99 are missing data 

## Business Question
What factors are associated with increased risk of death among COVID-19 patients in Mexico?

## Extraction and initial look at data

Code:
```
!head /home/chuhao/dsai_sctp/3m-data-assignment-1.2/data/Covid_Data.csv 
```
Code:
```
import numpy as np
import pandas as pd

covid= pd.read_csv("/home/chuhao/dsai_sctp/3m-data-assignment-1.2/data/Covid_Data.csv")
covid.head()
```
Output:
```
	USMER	MEDICAL_UNIT	SEX	PATIENT_TYPE	DATE_DIED	INTUBED	PNEUMONIA	AGE	PREGNANT	DIABETES	...	ASTHMA	INMSUPR	HIPERTENSION	OTHER_DISEASE	CARDIOVASCULAR	OBESITY	RENAL_CHRONIC	TOBACCO	CLASIFFICATION_FINAL	ICU
0	2	1	1	1	03/05/2020	97	1	65	2	2	...	2	2	1	2	2	2	2	2	3	97
1	2	1	2	1	03/06/2020	97	1	72	97	2	...	2	2	1	2	2	1	1	2	5	97
2	2	1	2	2	09/06/2020	1	2	55	97	1	...	2	2	2	2	2	2	2	2	3	2
3	2	1	1	1	12/06/2020	97	2	53	2	2	...	2	2	2	2	2	2	2	2	7	97
4	2	1	2	1	21/06/2020	97	2	68	97	1	...	2	2	1	2	2	2	2	2	3	97
5 rows × 21 columns
```

 Code:
 ```
#Due to how data is stored in binary for float, numbers rounded off to improve readibility
covid.describe().round(2)
```
Output:
```
	USMER	MEDICAL_UNIT	SEX	PATIENT_TYPE	INTUBED	PNEUMONIA	AGE	PREGNANT	DIABETES	COPD	ASTHMA	INMSUPR	HIPERTENSION	OTHER_DISEASE	CARDIOVASCULAR	OBESITY	RENAL_CHRONIC	TOBACCO	CLASIFFICATION_FINAL	ICU
count	1048575	1048575	1048575	1048575	1048575	1048575	1048575	1048575	1048575	1048575	1048575	1048575	1048575	1048575	1048575	1048575	1048575	1048575	1048575	1048575
mean	1.63	8.98	1.5	1.19	79.52	3.35	41.79	49.77	2.19	2.26	2.24	2.3	2.13	2.44	2.26	2.13	2.26	2.21	5.31	79.55
std	0.48	3.72	0.5	0.39	36.87	11.91	16.91	47.51	5.42	5.13	5.11	5.46	5.24	6.65	5.19	5.18	5.14	5.32	1.88	36.82
min	1	1	1	1	1	1	0	1	1	1	1	1	1	1	1	1	1	1	1	1
25%	1	4	1	1	97	2	30	2	2	2	2	2	2	2	2	2	2	2	3	97
50%	2	12	1	1	97	2	40	97	2	2	2	2	2	2	2	2	2	2	6	97
75%	2	12	2	1	97	2	53	97	2	2	2	2	2	2	2	2	2	2	7	97
max	2	13	2	2	99	99	121	98	98	98	98	98	98	98	98	98	98	98	7	99

```

interesting to note that max ages is claimed to be 121 in this dataset

Code:
```
#check for null rows
covid[covid.isna().all(axis="columns")]

```
Output:
```
USMER	MEDICAL_UNIT	SEX	PATIENT_TYPE	DATE_DIED	INTUBED	PNEUMONIA	AGE	PREGNANT	DIABETES	...	ASTHMA	INMSUPR	HIPERTENSION	OTHER_DISEASE	CARDIOVASCULAR	OBESITY	RENAL_CHRONIC	TOBACCO	CLASIFFICATION_FINAL	ICU
0 rows × 21 columns
```

no checks for duplicates were made in this case since no unique identification number was in the database. Duplicate rows can still be records of different patients. It has been stated that patients have been kept anonymous.

## Data Cleaning
### Create `covid_positive` as binary column; 1 = covid-positive, 0 = covid-negative

As per source file, clasiffication_final refers to covid test results. Values 1-3 mean that the patient was diagnosed with covid in different degrees. 4 or higher means that the patient is not a carrier of covid or that the test is inconclusive.

Code:
```

def covid_cat(x):
    if x in [1, 2, 3]:
        return 1
    else:
        return 0
covid["covid_positive"] = covid["CLASIFFICATION_FINAL"].apply(covid_cat)
covid["covid_positive"].value_counts()
```
Output:
```
0    656596
1    391979
Name: covid_positive, dtype: int64

```

### Create `died` as a binary column; 1 = died, 0 = survived

#source file states: If the patient died indicate the date of death, and 9999-99-99 otherwise.

Code:
```
def died_cat(x):
    if x == np.NaN or x == "9999-99-99":
        return 0
    else:
        return 1
    
covid["DIED"] = covid["DATE_DIED"].apply(died_cat)
covid["DIED"].value_counts()
```
Output:
```
0    971633
1     76942
Name: DIED, dtype: int64

```

### Convert all boolean columns into `1`, `0` or `NaN`

Code:
```
covid.info()
```
Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1048575 entries, 0 to 1048574
Data columns (total 23 columns):
 #   Column                Non-Null Count    Dtype 
---  ------                --------------    ----- 
 0   USMER                 1048575 non-null  int64 
 1   MEDICAL_UNIT          1048575 non-null  int64 
 2   SEX                   1048575 non-null  int64 
 3   PATIENT_TYPE          1048575 non-null  int64 
 4   DATE_DIED             1048575 non-null  object
 5   INTUBED               1048575 non-null  int64 
 6   PNEUMONIA             1048575 non-null  int64 
 7   AGE                   1048575 non-null  int64 
 8   PREGNANT              1048575 non-null  int64 
 9   DIABETES              1048575 non-null  int64 
 10  COPD                  1048575 non-null  int64 
 11  ASTHMA                1048575 non-null  int64 
 12  INMSUPR               1048575 non-null  int64 
 13  HIPERTENSION          1048575 non-null  int64 
 14  OTHER_DISEASE         1048575 non-null  int64 
 15  CARDIOVASCULAR        1048575 non-null  int64 
 16  OBESITY               1048575 non-null  int64 
 17  RENAL_CHRONIC         1048575 non-null  int64 
 18  TOBACCO               1048575 non-null  int64 
 19  CLASIFFICATION_FINAL  1048575 non-null  int64 
...
 21  covid_positive        1048575 non-null  int64 
 22  DIED                  1048575 non-null  int64 
dtypes: int64(22), object(1)
memory usage: 184.0+ MB

```
#source_data states: In the Boolean features, 1 means "yes" and 2 means "no". values as 97 and 99 are missing data
Code:
```
#declare columns in bool_col
bool_col =[
    "INTUBED", "PNEUMONIA", "PREGNANT", "DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION", "OTHER_DISEASE", "CARDIOVASCULAR", "OBESITY", "RENAL_CHRONIC", "TOBACCO", "ICU"
]
covid[bool_col].value_counts()
```
Output:
```
INTUBED  PNEUMONIA  PREGNANT  DIABETES  COPD  ASTHMA  INMSUPR  HIPERTENSION  OTHER_DISEASE  CARDIOVASCULAR  OBESITY  RENAL_CHRONIC  TOBACCO  ICU
97       2          2         2         2     2       2        2             2              2               2        2              2        97     274082
                    97        2         2     2       2        2             2              2               2        2              2        97     250299
                    2         2         2     2       2        2             2              2               1        2              2        97      33872
                    97        2         2     2       2        2             2              2               1        2              2        97      27291
                                                                                                            2        2              1        97      26356
                                                                                                                                                     ...  
2        1          2         1         1     2       1        1             2              1               2        2              1        2           1
97       1          2         1         2     2       2        1             98             1               1        1              2        97          1
2        1          2         1         1     2       1        1             2              2               2        2              1        2           1
97       1          2         1         2     2       2        1             2              2               2        1              1        97          1
99       99         98        98        98    98      98       98            98             98              98       98             98       99          1
Length: 8728, dtype: int64

```
Code:
```
covid["DIABETES"].value_counts()

Output:
2     920248
1     124989
98      3338
Name: DIABETES, dtype: int64

```

#It appears that there is a special value 98 in bool_col, transforming to np.NaN, as the desired transformed values are 1, 0, or NaN. (It will be good to clarify with data collection team on fields such as pregnant, diabetes, copd, hyperension, other_disease, cardio_vascular, obesity, renal_chronic. tobacco)

Code:
```
bool_transform ={
    1 : 1,
    2 : 0,
    97 : np.NaN,
    98: np.NaN,
    99 :np.NaN
}
covid[bool_col] = covid[bool_col].replace(bool_transform)
covid.describe().round(2)
#quick check on transformed dataframe

```
Output:
```
USMER	MEDICAL_UNIT	SEX	PATIENT_TYPE	INTUBED	PNEUMONIA	AGE	PREGNANT	DIABETES	COPD	...	HIPERTENSION	OTHER_DISEASE	CARDIOVASCULAR	OBESITY	RENAL_CHRONIC	TOBACCO	CLASIFFICATION_FINAL	ICU	covid_positive	DIED
count	1048575.00	1048575.00	1048575.0	1048575.00	192706.00	1032572.00	1048575.00	521310.00	1045237.00	1045572.00	...	1045471.00	1043530.00	1045499.00	1045543.00	1045569.00	1045355.00	1048575.00	192543.00	1048575.00	1048575.00
mean	1.63	8.98	1.5	1.19	0.17	0.14	41.79	0.02	0.12	0.01	...	0.16	0.03	0.02	0.15	0.02	0.08	5.31	0.09	0.37	0.07
std	0.48	3.72	0.5	0.39	0.38	0.34	16.91	0.12	0.32	0.12	...	0.36	0.16	0.14	0.36	0.13	0.27	1.88	0.28	0.48	0.26
min	1.00	1.00	1.0	1.00	0.00	0.00	0.00	0.00	0.00	0.00	...	0.00	0.00	0.00	0.00	0.00	0.00	1.00	0.00	0.00	0.00
25%	1.00	4.00	1.0	1.00	0.00	0.00	30.00	0.00	0.00	0.00	...	0.00	0.00	0.00	0.00	0.00	0.00	3.00	0.00	0.00	0.00
50%	2.00	12.00	1.0	1.00	0.00	0.00	40.00	0.00	0.00	0.00	...	0.00	0.00	0.00	0.00	0.00	0.00	6.00	0.00	0.00	0.00
75%	2.00	12.00	2.0	1.00	0.00	0.00	53.00	0.00	0.00	0.00	...	0.00	0.00	0.00	0.00	0.00	0.00	7.00	0.00	1.00	0.00
max	2.00	13.00	2.0	2.00	1.00	1.00	121.00	1.00	1.00	1.00	...	1.00	1.00	1.00	1.00	1.00	1.00	7.00	1.00	1.00	1.00
8 rows × 22 columns
```
Code:
```
#checking for rows with all na in bool_col categories
covid[covid[bool_col].isna().all(axis="columns")]

```
Output:
```
USMER	MEDICAL_UNIT	SEX	PATIENT_TYPE	DATE_DIED	INTUBED	PNEUMONIA	AGE	PREGNANT	DIABETES	...	HIPERTENSION	OTHER_DISEASE	CARDIOVASCULAR	OBESITY	RENAL_CHRONIC	TOBACCO	CLASIFFICATION_FINAL	ICU	covid_positive	DIED
444753	1	12	1	2	03/02/2020	NaN	NaN	20	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	7	NaN	0	1
661869	1	12	2	2	9999-99-99	NaN	NaN	41	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	5	NaN	0	0
723972	1	12	1	1	9999-99-99	NaN	NaN	30	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	7	NaN	0	0
3 rows × 23 columns
```
Code:
```
#dropping rows with all na in bool_col categories since they would not contribute to both descriptive and inferential stats
covid = covid[covid[bool_col].notna().any(axis="columns")]
covid[covid[bool_col].isna().all(axis="columns")]

```
Output:
```
USMER	MEDICAL_UNIT	SEX	PATIENT_TYPE	DATE_DIED	INTUBED	PNEUMONIA	AGE	PREGNANT	DIABETES	...	HIPERTENSION	OTHER_DISEASE	CARDIOVASCULAR	OBESITY	RENAL_CHRONIC	TOBACCO	CLASIFFICATION_FINAL	ICU	covid_positive	DIED
0 rows × 23 columns
```


Code:
```
#dropping redundant columns since there are 2 new columns (covid_positive, died) transformed from these 2
covid = covid.drop(["CLASIFFICATION_FINAL", "DATE_DIED"], axis = 1)
covid.info()

```
Output:
```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1048572 entries, 0 to 1048574
Data columns (total 21 columns):
 #   Column          Non-Null Count    Dtype  
---  ------          --------------    -----  
 0   USMER           1048572 non-null  int64  
 1   MEDICAL_UNIT    1048572 non-null  int64  
 2   SEX             1048572 non-null  int64  
 3   PATIENT_TYPE    1048572 non-null  int64  
 4   INTUBED         192706 non-null   float64
 5   PNEUMONIA       1032572 non-null  float64
 6   AGE             1048572 non-null  int64  
 7   PREGNANT        521310 non-null   float64
 8   DIABETES        1045237 non-null  float64
 9   COPD            1045572 non-null  float64
 10  ASTHMA          1045596 non-null  float64
 11  INMSUPR         1045171 non-null  float64
 12  HIPERTENSION    1045471 non-null  float64
 13  OTHER_DISEASE   1043530 non-null  float64
 14  CARDIOVASCULAR  1045499 non-null  float64
 15  OBESITY         1045543 non-null  float64
 16  RENAL_CHRONIC   1045569 non-null  float64
 17  TOBACCO         1045355 non-null  float64
 18  ICU             192543 non-null   float64
 19  covid_positive  1048572 non-null  int64  
 20  DIED            1048572 non-null  int64  
dtypes: float64(14), int64(7)
memory usage: 176.0 MB
```


## Exploratory Data Analysis & Visualisation
### Amongst COVID-positive patients, what is the average age? As well as the youngest and oldest?

Code:
```
#dataframe for covid positive
covid_pos_patients = covid[covid["covid_positive"] == 1]
covid_pos_patients.describe().round(2)

```
Output:
```
USMER	MEDICAL_UNIT	SEX	PATIENT_TYPE	INTUBED	PNEUMONIA	AGE	PREGNANT	DIABETES	COPD	...	INMSUPR	HIPERTENSION	OTHER_DISEASE	CARDIOVASCULAR	OBESITY	RENAL_CHRONIC	TOBACCO	ICU	covid_positive	DIED
count	391979.00	391979.00	391979.00	391979.00	109779.00	391975.00	391979.00	181107.00	390539.00	390666.00	...	390530.00	390591.0	389843.00	390588.00	390626.00	390629.00	390545.00	109771.00	391979.0	391979.00
mean	1.62	8.70	1.53	1.28	0.22	0.22	45.19	0.02	0.16	0.02	...	0.01	0.2	0.03	0.02	0.19	0.02	0.07	0.10	1.0	0.14
std	0.48	3.76	0.50	0.45	0.41	0.41	16.46	0.12	0.37	0.12	...	0.11	0.4	0.16	0.15	0.39	0.14	0.26	0.29	0.0	0.35
min	1.00	1.00	1.00	1.00	0.00	0.00	0.00	0.00	0.00	0.00	...	0.00	0.0	0.00	0.00	0.00	0.00	0.00	0.00	1.0	0.00
25%	1.00	4.00	1.00	1.00	0.00	0.00	33.00	0.00	0.00	0.00	...	0.00	0.0	0.00	0.00	0.00	0.00	0.00	0.00	1.0	0.00
50%	2.00	12.00	2.00	1.00	0.00	0.00	44.00	0.00	0.00	0.00	...	0.00	0.0	0.00	0.00	0.00	0.00	0.00	0.00	1.0	0.00
75%	2.00	12.00	2.00	2.00	0.00	0.00	56.00	0.00	0.00	0.00	...	0.00	0.0	0.00	0.00	0.00	0.00	0.00	0.00	1.0	0.00
max	2.00	13.00	2.00	2.00	1.00	1.00	120.00	1.00	1.00	1.00	...	1.00	1.0	1.00	1.00	1.00	1.00	1.00	1.00	1.0	1.00
8 rows × 21 columns

```
Initial Findings:
```
Among COVID posititive patients:
Mean age = 45.19
median age = 44
max age= 120 (an area worth exploring if dataset had error entries: intial checks could not verify max age in 2019)
min age= 0

```
Code:
```
import matplotlib.pyplot as plt
import seaborn as sns


plt.hist(covid_pos_patients["AGE"], bins = 60, alpha = 0.3)
plt.axvline(x = covid_pos_patients["AGE"].median(), linestyle = "--", color = "red", label = "median")
plt.axvline(x = covid_pos_patients["AGE"].mean(), linestyle = "--", color = "k", label = "mean")
plt.xlabel("AGE")
plt.ylabel("Count")
plt.legend()
plt.title("age distribution of covid positive patients")
plt.show()
```
![image](https://github.com/user-attachments/assets/aaddb8bc-74a6-44f4-812f-6b7800ff2529)
```
Code:
covid_pos_patients["AGE"].skew().round(3)
```
Output:
```
0.269
```

Additional findings: 
```
The distribution of covid positive patients by age is slightly skewed to the right, with mean slightly higher than median, but very close to median The skew is slightly positive but very close to zero.

A significant proportion of missing data is observed for intubed and ICU which could have been an area of interest if it had helped in the recovery of patients, especially when ventitllators were claimed to have aided in helping patients.


```

Plot the age distribution of COVID-positive patient and their death rate. Any trend between age and death rate?

Code:
```
covid.corrwith(covid["covid_positive"], numeric_only=True).sort_values(ascending=False)

```
Output:
```
covid_positive    1.000000
DIED              0.192567
PNEUMONIA         0.191651
PATIENT_TYPE      0.183201
AGE               0.155059
INTUBED           0.124145
DIABETES          0.093945
HIPERTENSION      0.086889
OBESITY           0.071783
SEX               0.054363
ICU               0.031106
RENAL_CHRONIC     0.013197
CARDIOVASCULAR    0.010583
COPD              0.008350
PREGNANT         -0.002301
OTHER_DISEASE    -0.005601
INMSUPR          -0.008921
USMER            -0.012078
ASTHMA           -0.015993
TOBACCO          -0.020152
MEDICAL_UNIT     -0.058647
dtype: float64

```
Comments:
The various data categories are not showing significant or moderate correleation with the probability of being affected by covid initially

Code:
```
covid_pos_patients.corrwith(covid_pos_patients["DIED"], numeric_only=True).sort_values(ascending=False)

Output:
DIED              1.000000
PATIENT_TYPE      0.553517
PNEUMONIA         0.513440
INTUBED           0.424788
AGE               0.397807
DIABETES          0.238162
HIPERTENSION      0.235274
RENAL_CHRONIC     0.129281
COPD              0.101041
SEX               0.093187
ICU               0.089021
CARDIOVASCULAR    0.085780
OBESITY           0.061442
OTHER_DISEASE     0.061015
INMSUPR           0.050870
TOBACCO           0.011864
ASTHMA           -0.016129
PREGNANT         -0.032850
USMER            -0.159976
MEDICAL_UNIT     -0.170018
covid_positive         NaN
dtype: float64
```
Findings:
```
Patient_type, pneumonia is showing moderate correlation with the mortality rate of patients who are affected with covid. Patient type indicates whether a patient is returned home or hospitalised
 
This implies that there is a moderate linear relationship between patient type, penumonia with mortality rate.

Age is showing weak correlation with mortality rate of covid patients.

Among the three, the correlation data implies that patient type has the most significant linear relationship with mortality rate.
This implies a benefit for ensuring there are sufficient capacity for patients in hospitals or triage facilities during covid.
```
Code:
```
fig, axes = plt.subplots(2, sharex=True,  gridspec_kw={'hspace':0} )

axes[0].hist(covid_pos_patients["AGE"], bins = 60, label = "covid positive", alpha = 0.3)
axes[0].axvline(x = covid_pos_patients["AGE"].median(), linestyle = "--", color = "red", label = "median")
axes[0].legend(loc = "upper right", fontsize = 10)
axes[0].set_ylabel("Covid positive count")
axes[1].hist(covid_pos_patients[covid_pos_patients['DIED'] == 1]['AGE'], bins = 60, color="y", label = "dead covid", alpha = 0.3)
axes[1].axvline(x = covid_pos_patients[covid_pos_patients['DIED'] == 1]['AGE'].median(), linestyle = "--", color = "red", label = "median")
plt.xlabel("AGE")
axes[1].set_ylabel("covid dead count")
plt.legend(fontsize = 10)
fig.suptitle("Age distribution of covid positive Vs dead covid patients")
plt.show()

```
![image](https://github.com/user-attachments/assets/f51a35a1-773b-479d-8c63-ab46a60cec49)


Additional code:
```
covid_pos_patients[covid_pos_patients['DIED'] == 1]['AGE'].median()
```
Output:
62.0
```
Code:
covid_pos_patients[covid_pos_patients['DIED'] == 1]['AGE'].mean().round(2)
```
Output:
61.53
```
Code:
covid_pos_patients[covid_pos_patients['DIED'] == 1]['AGE'].skew().round(3)
```
Output:
-0.364

Findings:
```
The distribution of dead covid patients is shifted towards the older patients as compared to the distribution of covid patients. 

This can be seen from the fact that the distribution of dead covid patients had a negative skew, with median higher than mean.

In comparison, the distribution of covid positive patients had a slightly postive skew, with median lower than mean.

The median in dead covid patients was higher at 62, as compared to 44 for covid positive patients.

This suggests a higher mortality rate for older patients.
```

### Do comorbidities contribute to a higher risk of death? Which specific conditions are associated with the greatest increase in mortality?

Code:
```
covid_pos_patients.info()
```
Output:
```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 391979 entries, 0 to 1047937
Data columns (total 21 columns):
 #   Column          Non-Null Count   Dtype  
---  ------          --------------   -----  
 0   USMER           391979 non-null  int64  
 1   MEDICAL_UNIT    391979 non-null  int64  
 2   SEX             391979 non-null  int64  
 3   PATIENT_TYPE    391979 non-null  int64  
 4   INTUBED         109779 non-null  float64
 5   PNEUMONIA       391975 non-null  float64
 6   AGE             391979 non-null  int64  
 7   PREGNANT        181107 non-null  float64
 8   DIABETES        390539 non-null  float64
 9   COPD            390666 non-null  float64
 10  ASTHMA          390670 non-null  float64
 11  INMSUPR         390530 non-null  float64
 12  HIPERTENSION    390591 non-null  float64
 13  OTHER_DISEASE   389843 non-null  float64
 14  CARDIOVASCULAR  390588 non-null  float64
 15  OBESITY         390626 non-null  float64
 16  RENAL_CHRONIC   390629 non-null  float64
 17  TOBACCO         390545 non-null  float64
 18  ICU             109771 non-null  float64
 19  covid_positive  391979 non-null  int64  
 20  DIED            391979 non-null  int64  
dtypes: float64(14), int64(7)
memory usage: 65.8 MB

```
Code:
```
#declare como categories
comorbidity_list = [
  'PNEUMONIA',
  'DIABETES',
  'COPD',
  'ASTHMA',
  'INMSUPR',
  'HIPERTENSION',
  'OTHER_DISEASE',
  'CARDIOVASCULAR',
  'OBESITY',
  'RENAL_CHRONIC'
]
#build dictionary for death rate of patients grouped by comorbidity type
death_rate_como_dict = {}
for como in comorbidity_list:
  death_rate_como_dict[como] = covid_pos_patients.groupby(como)['DIED'].mean()

#transform to table with named index, column
death_rate_comorbidity_df1 = pd.DataFrame(death_rate_como_dict).T
#initial column name was just 0,  1
death_rate_comorbidity_df1 = death_rate_comorbidity_df1.rename(columns = {0:"Without", 1 : "With"})
death_rate_comorbidity_df1 = death_rate_comorbidity_df1.rename_axis("Comorbidity").reset_index()
#additional col added for with vs without comparison
death_rate_comorbidity_df1['with_vs_without'] = (death_rate_comorbidity_df1['With']/death_rate_comorbidity_df1['Without']).round(2)
# sort before melting
death_rate_como_sorted = death_rate_comorbidity_df1.sort_values(by="with_vs_without", ascending= False)

#dataframe melted to long form table for ease in charting in seaborn, original dataframe kept separate on purpose
death_rate_comorbidity_melt= death_rate_como_sorted.melt(id_vars="Comorbidity", value_vars=["Without", "With"], var_name= "status", value_name="Death_rate")

```
Code:
```
#plotting chart
g = sns.catplot(
    data=death_rate_comorbidity_melt,
    kind="bar",
    hue="status",
    x="Comorbidity",
    y="Death_rate",
    height=6,
    aspect=1,
)

g.set_xticklabels(rotation = 45)
g._legend.set_bbox_to_anchor((1, 1))
g.fig.suptitle("COVID Death rate by Comorbidity Type", fontsize =14)
g.fig.tight_layout()

plt.show()
```
Output:

![image](https://github.com/user-attachments/assets/cacef76f-9efc-4b30-b5bc-50d4c13c9e01)


Code:
```
death_rate_como_sorted

```
Output:

![image](https://github.com/user-attachments/assets/a325ae6d-2696-47c9-89a8-f716f09cfd6f)


Findings:
```
As seen from the graph, there is a significant increase in death rate for patients with existing comorbidity status, except for asthma.

The data implies that if a patient has pneumonia when being covid positive, the death rate probability is increased at 10.66 times, as compared to another covid positive patient who does not have pneumonia. 

This difference observed in pneumonia cases is the highest among all comorbidity types in the dataset.

Recall that pneumonia has the highest correlation with mortality rate at 0.51 among comorbidity types, implying a linear relationship between pneumonia and death rate.

Patient type has the highest correlation with mortality rate at 0.55 but is not a comorbidity type.

Patient type indicates the type of care received: if the patient was returned home or hospitalized.

This suggests that healthcare facilities can increase survival rates of covid patients from hospitalising covid patients and focusing on pneumonia covid positive patients

```
## Additional data processing
Code:
```
covid_pos_patients[bool_col].isna().sum()
#check total number of n/a left in individual rows. rows with all na has been removed previously
```
Output:
```
INTUBED           282200
PNEUMONIA              4
PREGNANT          210872
DIABETES            1440
COPD                1313
ASTHMA              1309
INMSUPR             1449
HIPERTENSION        1388
OTHER_DISEASE       2136
CARDIOVASCULAR      1391
OBESITY             1353
RENAL_CHRONIC       1350
TOBACCO             1434
ICU               282208
dtype: int64
```
Code:
```
#percentage of null values left
#rounded off to improve readability
(covid_pos_patients[bool_col].isna().sum()/len(covid_pos_patients[bool_col])*100).round(0)
```
Output:
```
INTUBED           72.0
PNEUMONIA          0.0
PREGNANT          54.0
DIABETES           0.0
COPD               0.0
ASTHMA             0.0
INMSUPR            0.0
HIPERTENSION       0.0
OTHER_DISEASE      1.0
CARDIOVASCULAR     0.0
OBESITY            0.0
RENAL_CHRONIC      0.0
TOBACCO            0.0
ICU               72.0
dtype: float64
```
Findings:
```
The trans_col columns are binary(2 values)
This means that they are categorical in nature. There is no meaningful sequential order within 2 values. 
This is why mode, instead of median is chosen as the value for imputing for n/a values here.
```
Code:
```

#INTUBED, PREGNANT, ICU  has a signficant percentage of N/A value
#filling N/A before this stage would affect correlation values and skew
#making copy of dataframe to create new break point for data before imputing values for na 

covid_pos_clean3 = covid_pos_patients.copy()
trans_col = ["PNEUMONIA", "DIABETES", "COPD", "ASTHMA", "INMSUPR", 
             "HIPERTENSION", "OTHER_DISEASE", "CARDIOVASCULAR", 
             "OBESITY", "RENAL_CHRONIC", "TOBACCO"]

#mode is chosen as the value for imputing na values for these columns
#mode has to be converted to series before inserting to fillna

for col in trans_col:
    covid_pos_clean3[col].fillna(covid_pos_clean3[col].mode().iloc[0], inplace=True)
covid_pos_clean3[bool_col].isna().sum()

```
Code:
```
INTUBED           282200
PNEUMONIA              0
PREGNANT          210872
DIABETES               0
COPD                   0
ASTHMA                 0
INMSUPR                0
HIPERTENSION           0
OTHER_DISEASE          0
CARDIOVASCULAR         0
OBESITY                0
RENAL_CHRONIC          0
TOBACCO                0
ICU               282208
dtype: int64

```
Comments:
```
For intubed, pregnant, ICU has a significant percentage of N/A values (exceeeding 50 percent). 

Imputing values for n/a in these categories will affect the distribution for these columns.

Recall that these 3 categories, like the rest of bool_col columns, are currently binary (2 values). 

This also means that using the median or mode for filling up missing values is in fact equivalent to using min/max values and will distort the distribution significantly.

If analysis is required on these 3 variables, dropping/ignoring the N/A on another dataframe copy and carrying out the analysis on the remaining data can be considered.

In comparison, the other columns, such as pneumonia, where mode was used to fill in missing data, had a low percentage of N/A values.

The imputation of values for N/A would be ideally be carried out on the train set, but the test data should be kept intact.
```

