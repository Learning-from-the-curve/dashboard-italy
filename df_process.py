import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from process_functions import write_log, applyDiff
from pickle_functions import picklify, unpicklify
from operator import add



######################################
# Retrieve data
######################################

# Paths
path_population = Path.cwd() / 'input' / 'italy_population_istat.csv'
path_geo = Path.cwd() / 'input'/ 'province.geojson'

path_regioni = Path.cwd() / 'input' / 'dpc-covid19-ita-regioni.csv'
path_province = Path.cwd() / 'input' / 'dpc-covid19-ita-province.csv'
path_nazione = Path.cwd() / 'input' / 'dpc-covid19-ita-andamento-nazionale.csv'
#path_cod_province = Path.cwd() / 'input' / 'cod_province.CSV'

# get data directly from github. The data source provided by Johns Hopkins University.
url_regioni = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv'
url_province = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv'
url_nazione = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'

###########################################################################

pop = pd.read_csv(path_population)
#cod_province = pd.read_csv(path_cod_province, sep=";", encoding="latin-1")
#print(cod_province)

#load old data
df_nazione_backup = pd.read_csv(path_nazione, encoding="latin-1")
old_df_nazione = df_nazione_backup['data']
df_regioni_backup = pd.read_csv(path_regioni, encoding="latin-1")
old_df_regioni = df_regioni_backup[['data','denominazione_regione']]
df_province_backup = pd.read_csv(path_province, encoding="latin-1")
old_df_province = df_province_backup['codice_provincia']

#load new data
df_nazione = pd.read_csv(url_nazione, encoding="latin-1")
new_df_nazione = df_nazione['data']
df_regioni = pd.read_csv(url_regioni, encoding="latin-1")
new_df_regioni = df_regioni[['data','denominazione_regione']]
df_province = pd.read_csv(url_province, encoding="latin-1")
new_df_province = df_province['codice_provincia']

#print(list(df_province['denominazione_provincia']))
tot_province_casi = df_province[['data','denominazione_provincia','totale_casi']]
dates = set(df_province['data'])   # convert to dates   .apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').date()))
last_date = max(dates)
dates.remove(last_date)
second_to_last_date = max(dates)
#print( last_date )
#print(second_to_last_date )

df_province_second_to_last = tot_province_casi.loc[df_province['data'] == second_to_last_date]
tot_province_casi = tot_province_casi.loc[df_province['data'] == last_date]
tot_province_casi = tot_province_casi.drop(['data'], axis=1)
df_province_second_to_last = df_province_second_to_last.drop(['data'], axis=1)
tot_province_casi = tot_province_casi.loc[tot_province_casi['denominazione_provincia'] != 'In fase di definizione/aggiornamento']
tot_province_casi = tot_province_casi.loc[tot_province_casi['denominazione_provincia'] != 'Fuori Regione / Provincia Autonoma']
df_province_second_to_last = df_province_second_to_last.loc[df_province_second_to_last['denominazione_provincia'] != 'In fase di definizione/aggiornamento']
df_province_second_to_last = df_province_second_to_last.loc[df_province_second_to_last['denominazione_provincia'] != 'Fuori Regione / Provincia Autonoma']
tot_province_casi['daily'] = tot_province_casi['totale_casi']
tot_province_casi = tot_province_casi.apply( applyDiff,args = [df_province_second_to_last], axis =1)

#compute difference of rows and columns
nazione_date_diff = set(new_df_nazione).symmetric_difference(set(old_df_nazione))
regioni_date_diff = set(new_df_regioni['data']).symmetric_difference(set(old_df_regioni['data']))
regioni_name_diff = set(new_df_regioni['denominazione_regione']).symmetric_difference(set(old_df_regioni['denominazione_regione']))
province_cod_diff = set(new_df_province).symmetric_difference(set(old_df_province))

#print(regioni_date_diff,regioni_name_diff)

#write log and load the backup df if there is new data until the next update
#for nazione
write_log('--- nazione file check'.upper())

if len(nazione_date_diff) > 1:
    write_log('multiple new dates added: ' + str(nazione_date_diff))
elif len(nazione_date_diff) == 1:
    write_log('new date added: ' + str(nazione_date_diff))
else:
    write_log('no new date added')

#for province
write_log('--- province code file check'.upper())
if not bool(province_cod_diff):
    write_log('no new province code added')
else:
    write_log('new province code added:\n' + str(province_cod_diff))
    df_province = df_province_backup

#for regioni
write_log('--- regioni file check'.upper())
if not bool(regioni_name_diff):
    write_log('no new regioni added')
else:
    write_log('new regioni added:\n' + str(regioni_name_diff))
    df_regioni = df_regioni_backup

if len(regioni_date_diff) > 1:
    write_log('multiple new dates added: ' + str(regioni_date_diff))
elif len(regioni_date_diff) == 1:
    write_log('new date added: ' + str(regioni_date_diff))
else:
    write_log('no new date added')

df_regioni.to_csv(path_regioni, index = None)
df_province.to_csv(path_province, index = None)
df_nazione.to_csv(path_nazione, index = None)

#########################################################################################
# Data preprocessing for getting useful data and shaping data compatible to plotly plot
#########################################################################################
# for national counts
df_nazione['data'] = pd.to_datetime(df_nazione.data)
df_nazione['data'] = df_nazione['data'].dt.strftime('%Y/%m/%d')
df_regioni['data'] = pd.to_datetime(df_regioni.data)
df_regioni['data'] = df_regioni['data'].dt.strftime('%Y/%m/%d')

tot_nazione_ospedalizzati = df_nazione[['data', 'totale_ospedalizzati']]
tot_nazione_dimessi_guariti = df_nazione[['data', 'dimessi_guariti']]
tot_nazione_casi = df_nazione[['data', 'totale_casi']]
tot_nazione_deceduti = df_nazione[['data', 'deceduti']]
tot_nazione = df_nazione[['data', 'totale_casi', 'totale_ospedalizzati', 'dimessi_guariti', 'deceduti']]
#print(tot_nazione)
stats = list(df_regioni.drop(['data', 'stato', 'codice_regione', 'denominazione_regione', 'lat', 'long', 'note'], axis =1).columns)
#print(stats)
for stat in stats:   
    sumlist = list( map(add, list(df_regioni.loc[df_regioni['denominazione_regione'] == 'P.A. Trento', stat]), list(df_regioni.loc[df_regioni['denominazione_regione'] == 'P.A. Bolzano', stat])) )
    df_regioni.at[df_regioni['denominazione_regione'] == 'P.A. Trento', stat] = sumlist
    
df_regioni.at[df_regioni['denominazione_regione'] == 'P.A. Trento', 'denominazione_regione'] = 'Trentino Alto Adige'
df_regioni = df_regioni[~(df_regioni['denominazione_regione'] == 'P.A. Bolzano')]

#for tab card left and plots
tot_regioni_ospedalizzati = df_regioni[['data','denominazione_regione', 'totale_ospedalizzati']]
tot_regioni_dimessi_guariti = df_regioni[['data','denominazione_regione', 'dimessi_guariti']]
tot_regioni_casi = df_regioni[['data','denominazione_regione', 'totale_casi']]
tot_regioni_deceduti = df_regioni[['data','denominazione_regione', 'deceduti']]
tot_regioni = df_regioni[['data', 'denominazione_regione', 'totale_casi', 'totale_ospedalizzati', 'dimessi_guariti', 'deceduti']]
tot_regioni.rename(columns={
    'data': 'date', #
    'denominazione_regione': 'Region',
    'totale_ospedalizzati': 'in Hospital', 
    'dimessi_guariti': 'Discharged Healed',
    'deceduti': 'Deceased', 
    'totale_casi': 'Total Cases', 

    }, inplace=True)

tot_regioni.drop(tot_regioni[tot_regioni['Total Cases'] < 1].index, inplace=True)
tot_regioni = tot_regioni.reset_index(drop=True)

#sorted versions
sorted_regioni_casi = tot_regioni_casi.copy().groupby('denominazione_regione').last()
sorted_regioni_casi = sorted_regioni_casi.drop(['data'], axis=1)
sorted_regioni_casi = sorted_regioni_casi.sort_values(by=['totale_casi'], ascending = False)

sorted_regioni_deceduti = tot_regioni_deceduti.copy().groupby('denominazione_regione').last()
sorted_regioni_deceduti = sorted_regioni_deceduti.drop(['data'], axis=1)
sorted_regioni_deceduti = sorted_regioni_deceduti.sort_values(by=['deceduti'], ascending = False)

sorted_regioni_ospedalizzati = tot_regioni_ospedalizzati.copy().groupby('denominazione_regione').last()
sorted_regioni_ospedalizzati = sorted_regioni_ospedalizzati.drop(['data'], axis=1)
sorted_regioni_ospedalizzati = sorted_regioni_ospedalizzati.sort_values(by=['totale_ospedalizzati'], ascending = False)

sorted_regioni_dimessi_guariti = tot_regioni_dimessi_guariti.copy().groupby('denominazione_regione').last()
sorted_regioni_dimessi_guariti = sorted_regioni_dimessi_guariti.drop(['data'], axis=1)
sorted_regioni_dimessi_guariti = sorted_regioni_dimessi_guariti.sort_values(by=['dimessi_guariti'], ascending = False)

# for tab card right
tab_right_df = df_nazione.drop(['stato','note'], axis=1)[-1:]
tab_right_df.rename(columns={
    'data': 'date', #
    'ricoverati_con_sintomi': 'hospitalized_with_symptoms',
    'terapia_intensiva': 'intensive_care',
    'totale_ospedalizzati': 'total_hospitalized', 
    'isolamento_domiciliare': 'home_isolation',
    'totale_positivi': 'total_positives',
    'variazione_totale_positivi': 'total_positives_variation',
    'nuovi_positivi': 'new_positives',
    'dimessi_guariti': 'discharged_healed',
    'deceduti': 'deceased', 
    'casi_da_sospetto_diagnostico': 'cases_of_diagnostic_suspicion',
    'casi_da_screening' : 'screening cases',
    'totale_casi': 'total_cases', 
    'tamponi': 'swabs',
    'casi_testati': 'cases_tested',
    }, inplace=True)

#for province map

tot_province_casi.at[tot_province_casi['denominazione_provincia'] == 'Massa Carrara','denominazione_provincia'] = 'Massa-Carrara'
tot_province_casi.at[tot_province_casi['denominazione_provincia'] == "Reggio nell'Emilia",'denominazione_provincia'] = 'Reggio Emilia'
tot_province_casi.at[tot_province_casi['denominazione_provincia'] == "La Spezia",'denominazione_provincia'] = 'La spezia'
tot_province_casi.at[tot_province_casi['denominazione_provincia'] == "Reggio di Calabria",'denominazione_provincia'] = 'Reggio Calabria'
tot_province_casi.at[tot_province_casi['denominazione_provincia'] == 'Aosta','denominazione_provincia'] = "Valle d'Aosta"
tot_province_casi.at[tot_province_casi['denominazione_provincia'] == "ForlÃ¬-Cesena",'denominazione_provincia'] = "ForlÂ\x8d-Cesena"

tot_province_casi = tot_province_casi.append(pd.Series(['Carbonia-Iglesias', int(tot_province_casi.loc[tot_province_casi['denominazione_provincia'] == 'Sud Sardegna','totale_casi']), int(tot_province_casi.loc[tot_province_casi['denominazione_provincia'] == 'Sud Sardegna','daily'])], index=tot_province_casi.columns), ignore_index=True)
tot_province_casi = tot_province_casi.append(pd.Series(['Medio Campidano', int(tot_province_casi.loc[tot_province_casi['denominazione_provincia'] == 'Sud Sardegna','totale_casi']), int(tot_province_casi.loc[tot_province_casi['denominazione_provincia'] == 'Sud Sardegna','daily'])], index=tot_province_casi.columns), ignore_index=True)
tot_province_casi = tot_province_casi.append(pd.Series(['Ogliastra', int(tot_province_casi.loc[tot_province_casi['denominazione_provincia'] == 'Nuoro','totale_casi']), int(tot_province_casi.loc[tot_province_casi['denominazione_provincia'] == 'Nuoro','daily'])], index=tot_province_casi.columns), ignore_index=True)
tot_province_casi = tot_province_casi.append(pd.Series(['Olbia-Tempio', int(tot_province_casi.loc[tot_province_casi['denominazione_provincia'] == 'Sassari','totale_casi']), int(tot_province_casi.loc[tot_province_casi['denominazione_provincia'] == 'Sassari','daily'])], index=tot_province_casi.columns), ignore_index=True)
tot_province_casi = tot_province_casi[tot_province_casi['denominazione_provincia'] != 'Sud Sardegna']
tot_province_casi = tot_province_casi.reset_index(drop=True)

with open(path_geo, encoding='latin-1') as f:
    geo = json.load(f)


#lista_province = []
#for i in range(len(geo['features'])):
#    lista_province.append(geo['features'][i]['properties']["NOME_PRO"])

#print(lista_province)

#province_in_geo = []
#for i in lista_province:
#    if i not in set(tot_province_casi['denominazione_provincia']):
#        province_in_geo.append(i)


#print(province_in_geo)

#province_in_csv = []
#for i in set(tot_province_casi['denominazione_provincia']):
#    if i not in lista_province:
#        province_in_csv.append(i)
#
#print(province_in_csv)


####################################################################

#store the pickles for all the df needed
dataframe_list = [
    [tot_nazione_ospedalizzati, 'tot_nazione_ospedalizzati'],
    [tot_nazione_dimessi_guariti, 'tot_nazione_dimessi_guariti'],
    [tot_nazione_casi, 'tot_nazione_casi'],
    [tot_nazione_deceduti, 'tot_nazione_deceduti'],
    [tot_regioni_ospedalizzati, 'tot_regioni_ospedalizzati'],
    [tot_regioni_dimessi_guariti, 'tot_regioni_dimessi_guariti'],
    [tot_regioni_casi, 'tot_regioni_casi'],
    [tot_regioni_deceduti, 'tot_regioni_deceduti'],
    [tab_right_df, 'tab_right_df'],
    [tot_province_casi, 'tot_province_casi'],
    [sorted_regioni_casi, 'sorted_regioni_casi'],
    [sorted_regioni_deceduti, 'sorted_regioni_deceduti'],
    [sorted_regioni_ospedalizzati, 'sorted_regioni_ospedalizzati'],
    [sorted_regioni_dimessi_guariti, 'sorted_regioni_dimessi_guariti'],
    [geo,'geo'],
    [tot_nazione, 'tot_nazione'],
    [tot_regioni, 'tot_regioni'],
    [pop, 'pop'],
    
]

for dataframe, name in dataframe_list:
    picklify(dataframe, name)
