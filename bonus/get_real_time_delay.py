#!/usr/bin/env python3
import requests
import pandas as pd

old_df = pd.read_csv("cleaned_dataset.csv")

url = "https://data.sncf.com/api/explore/v2.1/catalog/datasets/regularite-mensuelle-tgv-aqst/records"
ma_cle_api = "7c873e047288843cfa85770fa55714819c63c89c920f84f5c7b08e6f"
headers = {'Authorization': f'Apikey {ma_cle_api}'}

new_data = []
limite = 100
offset = 0

print("Start get data")

while True:
    parametres = {
        'select': 'date, service, gare_depart, gare_arrivee, duree_moyenne, '
                  'nb_train_prevu, retard_moyen_tous_trains_arrivee',
        'where': 'date >= "2025-12"',
        'limit': limite,
        'offset': offset
    }

    reponse = requests.get(url, headers=headers, params=parametres)
    if reponse.status_code == 200:
        donnees = reponse.json()
        nb_page = donnees.get('results', [])
        if len(nb_page) == 0:
            break
        new_data.extend(nb_page)
        print(f"{len(new_data)} new lines")
        offset += limite
    else:
        print("Error :", reponse.status_code)
        break

if len(new_data) > 0:
    new_df = pd.DataFrame(new_data)
    new_df = new_df.rename(columns={"date": "Date", "service": "Service",
                           "gare_depart": "Departure station",
                  "gare_arrivee": "Arrival station", "duree_moyenne":
                               "Average journey time", "nb_train_prevu":
                               "Number of schedueld trains",
                  "retard_moyen_tous_trains_arrivee": "Average delay of all trains at arrival"})
    print(new_df)
    df_final = pd.concat([old_df, new_df])
    print(f"{len(df_final)} new lines.")
    df_final.to_csv("bonus/real_time_dataset.csv", index=False)
else:
    print("Nothing new")