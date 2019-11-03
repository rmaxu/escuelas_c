# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:06:07 2019

@author: uhrma
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#-----------------------Read CEMABE 1------------------------------------------
df1 = pd.read_csv("tr_inmuebles.csv")

#Transform attributes

df1['P12'] = df1['P12'].map({1:1, 2:0, 3:0, 4:1, 9:0, np.nan:0})
df1['P13A'] = df1['P13A'].map({1:1, 2:0, 3:0, 4:0, 9:0, 5:0, np.nan:0})
df1['P14'] = df1['P14'].map({1:1, 2:0, 3:0, 4:0, 9:0, 5:0, 6:0, np.nan:0})
df1['P15'] = df1['P15'].map({1:1, 2:0, 3:0, 4:0, 9:0, 5:0, 6:0, np.nan:0})
df1['P16'] = df1['P16'].map({1:1, 2:1, 3:0, 9:0, np.nan:0})
df1['P17A'] = df1['P17A'].map({1:1, 2:1, 3:1, 4:0, 5:0, 6:0, 9:0, np.nan:0})
df1['P18A'] = df1['P18A'].map({1:1, 2:1, 3:1, 4:1, 5:0, 9:0, np.nan:0})
df1['P19'] = df1['P19'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P20'] = df1['P20'].map({1:0, 2:1, 9:1, np.nan:1})
df1['P21'] = df1['P21'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P22'] = df1['P22'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P23'] = df1['P23'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P24'] = df1['P24'].map({1:1, 2:0, 9:0, np.nan:0})

df1['P25'].replace([999,np.nan],[0,0],inplace=True)
df1['P42'].replace([999,np.nan],[0,0],inplace=True)
df1['P43'].replace([999,np.nan],[0,0],inplace=True)
df1['P44'].replace([999,np.nan],[0,0],inplace=True)


"""
df1['P25'] = df1['P25'].map({999:0, np.nan:0})
df1['P42'] = df1['P42'].map({999:0, np.nan:0})
df1['P43'] = df1['P43'].map({999:0, np.nan:0})
df1['P44'] = df1['P44'].map({999:0, np.nan:0})
"""
df1['P92'] = df1['P92'].map({1:0, 2:1, 9:0, np.nan:0})
df1['P102'] = df1['P102'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P103'] = df1['P103'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P104'] = df1['P104'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P105'] = df1['P105'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P106'] = df1['P106'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P107'] = df1['P107'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P108'] = df1['P108'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P109'] = df1['P109'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P112'] = df1['P112'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P113'] = df1['P113'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P114'] = df1['P114'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P116'] = df1['P116'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P117'] = df1['P117'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P118'] = df1['P118'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P119'] = df1['P119'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P122'] = df1['P122'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P123'] = df1['P123'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P124'] = df1['P124'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P125'] = df1['P125'].map({1:1, 2:0, 9:0, np.nan:0})
#df1['P126'] = df1['P126'].map({999:0, np.nan:0})
df1['P126'].replace([999,np.nan],[0,0],inplace=True)

df1['P133'] = df1['P133'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P134'] = df1['P134'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P135'] = df1['P135'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P136'] = df1['P136'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P137'] = df1['P137'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P138'] = df1['P138'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P139'] = df1['P139'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P140'] = df1['P140'].map({1:1, 2:0, 9:0, np.nan:0})
df1['P141'] = df1['P141'].map({1:1, 2:0, 3:0, 9:0, np.nan:0})
df1['P142'] = df1['P142'].map({1:1, 2:0, 3:0, 9:0, np.nan:0})
df1['P143'] = df1['P143'].map({1:1, 2:0, 3:0, 9:0, np.nan:0})

# ------------------------Read CEMABE 2 ---------------------------------------
df2 = pd.read_csv("tr_centros.csv", dtype={'CLAVE_CT': 'str', 'TURNO': 'str'})

"""
df2['P169'] = df2['P169'].map({999:0, np.nan:0})
df2['P170'] = df2['P170'].map({999:0, np.nan:0})
df2['P171'] = df2['P171'].map({999:0, np.nan:0})
df2['P172'] = df2['P172'].map({999:0, np.nan:0})
df2['P173'] = df2['P173'].map({999:0, np.nan:0})
df2['P179'] = df2['P179'].map({999:0, np.nan:0})
df2['P180'] = df2['P180'].map({999:0, np.nan:0})
df2['P181'] = df2['P181'].map({999:0, np.nan:0})

"""
df2['P169'].replace([999,np.nan],[0,0],inplace=True)
df2['P170'].replace([999,np.nan],[0,0],inplace=True)
df2['P171'].replace([999,np.nan],[0,0],inplace=True)
df2['P172'].replace([999,np.nan],[0,0],inplace=True)
df2['P173'].replace([999,np.nan],[0,0],inplace=True)
df2['P179'].replace([999,np.nan],[0,0],inplace=True)
df2['P180'].replace([999,np.nan],[0,0],inplace=True)
df2['P181'].replace([999,np.nan],[0,0],inplace=True)


df2['P186'] = df2['P186'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P189'] = df2['P189'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P191'] = df2['P191'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P192'] = df2['P192'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P195'] = df2['P195'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P196'] = df2['P196'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P202'] = df2['P202'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P205'] = df2['P205'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P206'] = df2['P206'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P207'] = df2['P207'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P208'] = df2['P208'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P209'] = df2['P209'].map({1:1, 2:0, 9:0, np.nan:0})


df2['P216'] = df2['P216'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P219'] = df2['P219'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P222'] = df2['P222'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P225'] = df2['P225'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P228'] = df2['P228'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P231'] = df2['P231'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P234'] = df2['P234'].map({1:1, 2:0, 9:0, np.nan:0})
#df2['P237'] = df2['P237'].map({999:0, np.nan:0})
df2['P237'].replace([999,np.nan],[0,0],inplace=True)
df2['P265'] = df2['P265'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P266'] = df2['P266'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P268'] = df2['P268'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P269'] = df2['P269'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P270'] = df2['P270'].map({1:1, 2:0, 9:0, np.nan:0})
df2['P271'] = df2['P271'].map({1:1, 2:0, 9:0, np.nan:0})
"""
df2['P276'] = df2['P276'].map({999:0, np.nan:0})
df2['P277'] = df2['P277'].map({999:0, np.nan:0})
df2['P278'] = df2['P278'].map({999:0, np.nan:0})
df2['P279'] = df2['P279'].map({999:0, np.nan:0})
df2['P280'] = df2['P280'].map({999:0, np.nan:0})
df2['P281'] = df2['P281'].map({999:0, np.nan:0})
df2['P282'] = df2['P282'].map({999:0, np.nan:0})
df2['P283'] = df2['P283'].map({999:0, np.nan:0})
"""
df2['P276'].replace([9999,np.nan],[0,0],inplace=True)
df2['P277'].replace([9999,np.nan],[0,0],inplace=True)
df2['P278'].replace([9999,np.nan],[0,0],inplace=True)
df2['P279'].replace([9999,np.nan],[0,0],inplace=True)
df2['P280'].replace([9999,np.nan],[0,0],inplace=True)
df2['P281'].replace([9999,np.nan],[0,0],inplace=True)
df2['P282'].replace([9999,np.nan],[0,0],inplace=True)
df2['P283'].replace([9999,np.nan],[0,0],inplace=True)

#Merge both datasets
dfc = pd.merge(df1, df2, left_on=['ID_INM'], right_on=['ID_INM'], how='right')

print(dfc.isnull().any()) #Chech for nan values

dfc = dfc.drop(['ID_INM','TURNO'], 1)

dfc.to_csv("cemabe.csv", encoding='utf-8-sig', index=False)


#------------------------- Secundarias ----------------------------------------

#Read enlace_sec
df1= pd.read_csv("enlace_secundarias.csv")

df1.replace('S/D', np.nan, inplace=True) #Replace S/D by nan

df1['TURNO'] = df1['TURNO'].map({'MATUTINO':1, 'VESPERTINO':2, 
                               'NOCTURNO':3, 'DISCONTINU':4})

df1 = df1.astype({'ESPAÑOL_G1':'float32', 'MATEMÁTICAS_G1':'float32', 
            'ESPAÑOL_G2':'float32', 'MATEMÁTICAS_G2':'float32', 
            'F. C. y E. G2':'float32', 'ESPAÑOL_G3':'float32',
            'MATEMÁTICAS_G3':'float32', 'F. C. y E. G3':'float32',
            'TURNO':'str'})

#Create the key for merging with cemabe dataset
df1['CLAVE DE LA ESCUELA'] = df1['CLAVE DE LA ESCUELA'].astype(str) + df1['TURNO'].astype(str)


#Impute missing values

values = df1.iloc[:, 6:].values
imputer = IterativeImputer(max_iter = 100)
transformed_values = imputer.fit_transform(values)

cols1 = df1.iloc[:, 6:].columns

df_temp = pd.DataFrame(transformed_values, columns= cols1)

df1 = df1.iloc[:,:6].join(df_temp)

print(df1.isnull().any()) #Chech for nan values

#Merge enlace_sec and cemabe datasets. This is the final dataset for secundarias
df_sec = pd.merge(df1, dfc, left_on=['CLAVE DE LA ESCUELA'], right_on=['CLAVE_CT'], how='inner')

df_sec = df_sec.drop(['TURNO', 'CLAVE_CT'], 1)

df_sec.to_csv("secundarias.csv", encoding='utf-8-sig', index=False)

#----------------------- Primarias --------------------------------------------

#Read enlace_prim
df2 = pd.read_csv("enlace_primarias_1.csv")
df3 = pd.read_csv("enlace_primarias_2.csv")

df2 = pd.concat([df2, df3], ignore_index=True) #Concatenate both datasets

df2.replace('S/D', np.nan, inplace=True) #Replace S/D by nan

df2['TURNO'] = df2['TURNO'].map({'MATUTINO':1, 'VESPERTINO':2, 
                               'NOCTURNO':3, 'DISCONTINU':4})

df2 = df2.astype({'ESPAÑOL_G3':'float32', 'MATEMÁTICAS_G3':'float32', 
            'F. C. y E. G3':'float32', 'ESPAÑOL_G4':'float32', 
            'MATEMÁTICAS_G4':'float32', 'F. C. y E. G4':'float32', 
            'ESPAÑOL_G5':'float32', 'MATEMÁTICAS_G5':'float32', 
            'F. C. y E. G5':'float32', 'ESPAÑOL_G6':'float32',
            'MATEMÁTICAS_G6':'float32', 'F. C. y E. G6':'float32','TURNO':'str'})

#Create the key for merging with cemabe dataset
df2['CLAVE DE LA ESCUELA'] = df2['CLAVE DE LA ESCUELA'].astype(str) + df2['TURNO'].astype(str)

#Impute missing values
values = df2.iloc[:, 6:].values
imputer = IterativeImputer(max_iter = 100)
transformed_values = imputer.fit_transform(values)

cols1 = df2.iloc[:, 6:].columns

df_temp = pd.DataFrame(transformed_values, columns= cols1)

df2 = df2.iloc[:,:6].join(df_temp)

print(df2.isnull().any()) #Chech for nan values

#Merge enlace_sec and cemabe datasets. This is the final dataset for secundarias
df_prim = pd.merge(df2, dfc, left_on=['CLAVE DE LA ESCUELA'], right_on=['CLAVE_CT'], how='inner')

df_prim = df_prim.drop(['TURNO', 'CLAVE_CT'], 1)

df_prim.to_csv("primarias.csv", encoding='utf-8-sig', index=False)



