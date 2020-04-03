from datetime import datetime, timedelta
import pandas as pd
import random
import itertools as it
import time
import numpy as np
import pickle

store_path = './'

# the time slot is per day
time_frequency = 60 * 24
# choose_neighborhood_id = 26
#chunk_size is also sample_length, which indicates n-1 intervals, then n-2 train, 1 label
chunk_size = 10
# chunk_size = 6
# chunk_size = 8
# chunk_size = 12
# chunk_size = 14
# chunk_size = 16

def convert24(str1):

    # Checking if last two elements of time
    # is AM and first two elements are 12
    return_string = ''
    if str1[-2:] == "AM" and str1[11:13] == "12":
        return_string = str1[:11]+ '00' + str1[13:-2]
        return return_string.strip()

    # remove the AM
    elif str1[-2:] == "AM":
        return str1[:-2].strip()

        # Checking if last two elements of time
    # is PM and first two elements are 12
    elif str1[-2:] == "PM" and str1[11:13] == "12":
        return str1[:-2].strip()

    else:
        # add 12 to hours and remove PM
        return_string = ''
        return_string = str1[:11]+ str(int(str1[11:13])+ 12) + str1[13:-2]
        return return_string.strip()

def convert(date_time):
    data_format = '%m/%d/%Y %H:%M:%S' # The format
    datetime_str = datetime.strptime(date_time, data_format)

    return datetime_str

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

def generate_time_list(start, end):
    dts = [dt.strftime('%Y-%m-%dT%H:%MZ') for dt in
           datetime_range(start, end, timedelta(minutes=time_frequency))]
    return dts

def moving_window(x, length, step=1):
    streams = it.tee(x, length)
    return zip(*[it.islice(stream, i, None, step) for stream, i in zip(streams, it.count(step=step))])


class GenerateData:
    def __init__(self):
        file_path = '/tank/users/jiaosun/crime/crime-prediction/data/8-crime-chicago-2015'
        crime = pickle.load(open(file_path, 'rb'))

        # crime = pd.read_csv(file_path, sep=",", header=None, index_col=None)
        self.crime = crime

    def preprocess_time(self):
        crime_useful = self.crime
        crime_useful["time"] = crime_useful["time"].apply(convert24)
        crime_useful["combine_time"] = crime_useful["time"].apply(convert)
        self.crime = crime_useful

    def mapping_neighborhood_crime(self):
        crime_df = self.crime
        crime_dict = {k:v for v,k in enumerate(crime_df["crime_type"].unique())}
        crime_df['crime_type_id'] = crime_df["crime_type"].map(crime_dict)
        self.crime = crime_df

    def choose_target_generate_fllist(self):
        '''
        :param choose_neighborhood_id: the id of the neighborhood we want to target at
        :param chunk_size: sample size
        :return:
        '''

        # please change this and make it unified throughout the code
        crime = self.crime
        # cuz here we do not filter out any of the rows in the crime dataset
        sheroaks_crime = crime
        # ID of the chosen neighborhood
        crime_type = len(crime["crime_type_id"].unique())
        neighborhood_type = len(crime["neighborhood"].unique())

        # sheroaks_crime = crime[crime["neighborhood_id"] == choose_neighborhood_id]

        start_time_so = min(sheroaks_crime["combine_time"])
        end_time_so = max(sheroaks_crime["combine_time"])
        time_list_so = [dt.strftime('%Y-%m-%dT%H:%MZ') for dt in datetime_range(start_time_so, end_time_so, timedelta(minutes=time_frequency))]
        x_=list(moving_window(time_list_so, chunk_size))

        final_list_so = []
        label_list_so = []
        for i in range(0, len(x_)):
            feature_time_frame = x_[i][:chunk_size-1]
            feature_list = []
            ##fix a bug here
            for index_fea in range(0, len(feature_time_frame) - 1):
                start_so = feature_time_frame[index_fea]
                end_so = feature_time_frame[index_fea + 1]
                df_so_middle = sheroaks_crime.loc[(sheroaks_crime['combine_time'] >= start_so) & (sheroaks_crime['combine_time'] < end_so)]
                crime_record = np.zeros((neighborhood_type, crime_type))
                for index, row in df_so_middle.iterrows():
                    crime_record[int(row["neighborhood"]) - 1][int(row["crime_type_id"])] = 1
                feature_list.append(crime_record)
            final_list_so.append(feature_list)

            label_time_frame = x_[i][chunk_size-2:]
            label_time_slots = sheroaks_crime.loc[(sheroaks_crime['combine_time'] >= label_time_frame[0]) & (sheroaks_crime['combine_time'] < label_time_frame[1])]
            crime_record = np.zeros((neighborhood_type, crime_type))
            for index_label, row_label in label_time_slots.iterrows():
                crime_record[int(row_label["neighborhood"])-1][int(row_label["crime_type_id"])-1] = 1
            label_list_so.append(crime_record)

        print("the shape of feature list is {}, and the shape of label list is {} ".format(np.shape(final_list_so),
                                                                                           np.shape(label_list_so)))
        ts = time.time()
        # st = datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

        np.array(final_list_so).dump(open(store_path + 'feature' + '_chicago_8_nei_77_14' +'.npy', 'wb'))
        np.array(label_list_so).dump(open(store_path + 'label' + '_chicago_8_nei_77_14' +'.npy', 'wb'))
        print("Successfully stored the data at "+ store_path + " !")
        return final_list_so, label_list_so


if __name__ == '__main__':
    test_object = GenerateData()
    test_object.preprocess_time()
    # print(test_object.crime.columns)
    test_object.mapping_neighborhood_crime()
    test_object.choose_target_generate_fllist()