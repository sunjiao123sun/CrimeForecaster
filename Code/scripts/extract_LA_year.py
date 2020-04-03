import numpy as np
import pandas as pd
import pickle

# please download the LA_dataset_full data from the (https://drive.google.com/open?id=1nbqr0gdp_bO2QaIRN6qPx-7Tvd5pKa0P)
file_path = '/home/users/jiaosun/crime/data/LA_dataset_full'
la_crime = pickle.load(open(file_path, 'rb'), encoding='utf-8')

category_dict_1 = {
    'theft': ['property crime, theft', 'theft, property crime'],
    'vehicle_theft': ['property crime, vehicle theft'],
    'burglary': ['burglary, property crime'],
    'fraud': ['fraud, property crime', 'property crime, fraud'],
    'assault': ['assault, violent crime'],
    'vandalism': ['vandalism, property crime', 'property crime, vandalism'],
    'robbery': ['violent crime, robbery'],
    'sexual_offenses': ['assault, sexual offenses, violent crime', 'quality of life crime, sexual offenses', 'sexual offenses, quality of life crime', 'sexual offenses, violent crime', 'sexual offenses, violent crime, assault', 'violent crime, sexual offenses', 'violent crime, sexual offenses, assault'],
}

all_new_categories = list()
for key, value in category_dict_1.items():
    this_category_crime = la_crime[la_crime['big_bundle_name'].isin(value)]
    this_category_crime['new_category'] = key
    all_new_categories.append(this_category_crime)

#keep 89% of the original data
df_la = pd.concat(all_new_categories)
# keep the data of year 2019
df_la["year"] = [x.year for x in df_la["date_occ"]]
df_la = df_la[df_la.year == 2018]
print("length of the la dataframe: ", len(df_la))

distance_df = pd.read_csv('../../Geographical-LA/neighborhood_dis.csv')
neigh_list = distance_df["neighborhood"].unique()
df_la = df_la[df_la["neighborhood"].isin(neigh_list)]
pickle.dump(df_la, open('../../Geographical-LA/8_category_la_2018', 'wb'))


