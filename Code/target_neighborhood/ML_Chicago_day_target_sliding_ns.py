'''
Simple machine learning models for training, when we have a certain target, no spatial information involved
data format: (data size, training size, neighborhood information)
'''
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import itertools as it
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from itertools import cycle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics_sk
from sklearn.multiclass import OneVsRestClassifier
import statistics
import os
import argparse

import random
import itertools as it
import time

data_folder = '../data/simple_ML/Chicago/'


#local file name format for training data and test data: timeFrequency_neighID_chunksize_train
#timeFrequency_neighID_chunksize_train
time_frequency = 60 * 24

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


def evaluate_micro_plot(Y_test, y_score, n_classes):
    '''
    Y_test: the label for test data;
    y_scrore: the predict result for rest data;
    n_classses: the amount of classes for the classifier;
    '''
    Y_test = np.array(Y_test)
    y_score = np.array(y_score)
    # For each class

    average_precision = average_precision_score(Y_test, y_score, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision))

    macro_f1 = metrics_sk.f1_score(Y_test, y_score, average = 'macro')
    micro_f1 = metrics_sk.f1_score(Y_test, y_score, average = 'micro')
    print("The average macro-F1 score is {}, average micro-F1 score is {}".format(macro_f1, micro_f1))
    return average_precision

class ML_LA_target_sliding:
    def __init__(self, neighborhood_info = 'no', choose_neighborhood_id = 2, choose_model = 'LR'):
        file_path = '../data/8-crime-chicago-2015'
        crime = pickle.load(open(file_path, 'rb'))
        print(crime["neighborhood"].unique())
        chicago_crime = crime[~crime["Longitude"].isnull()]

        self.crime = chicago_crime
        self.model = choose_model
        self.start_time = ''
        self.end_time = ''
        self.threshold_rate = 0
        self.neighborhood_info = neighborhood_info
        self.crime_type = len(self.crime["crime_type"].unique())

        # the time slot is per day
        self.time_frequency = 60 * 24
        self.choose_neighborhood_id = choose_neighborhood_id #for the new category dataset
        self.feature = ''
        self.label = ''
        #chunk_size is also sample_length, which indicates n-1 intervals, then n-2 train, 1 label
        self.chunk_size = 10
        if self.neighborhood_info == 'yes':
            print("neighborhood_info YES")
            self.save_name = str(self.crime_type)+'_'+str(self.time_frequency)+'_'+str(self.choose_neighborhood_id)+'_'+str(
                self.chunk_size)+'_neigh'
        elif self.neighborhood_info == 'no':
            print("neighborhood_info NO")
            self.save_name = str(self.crime_type)+'_'+str(self.time_frequency)+'_'+str(self.choose_neighborhood_id)+'_'+str(
                self.chunk_size)+'_noneigh'
        return

    def preprocess_time(self):
        crime_useful = self.crime
        crime_useful["time"] = crime_useful["time"].apply(convert24)
        crime_useful["combine_time"] = crime_useful["time"].apply(convert)
        self.crime = crime_useful

    def mapping_neighborhood_crime(self):
        crime = self.crime
        # map the neighborhood and crime type information to IDs
        crime_dict = {k:v for v,k in enumerate(crime["crime_type"].unique())}
        crime['crime_type_id'] = crime["crime_type"].map(crime_dict)
        self.crime = crime
        return

    def choose_target_generate_fllist(self):
        '''
        :param choose_neighborhood_id: the id of the neighborhood we want to target at
        :param chunk_size: sample size
        :return:
        '''
        print("You are not using the target's neighborhood information.")
        choose_neighborhood_id, chunk_size, time_frequency = self.choose_neighborhood_id, self.chunk_size, self.time_frequency

        # please change this and make it unified throughout the code
        crime = self.crime
        # ID of the chosen neighborhood
        crime_type = len(crime["crime_type_id"].unique())

        sheroaks_crime = crime[crime["neighborhood"] == choose_neighborhood_id]
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
                crime_record = np.zeros(crime_type)
                for index, row in df_so_middle.iterrows():
                    crime_record[int(row["crime_type_id"])] = 1
                feature_list.append(crime_record)
            final_list_so.append(feature_list)

            label_time_frame = x_[i][chunk_size-2:]
            label_time_slots = sheroaks_crime.loc[(sheroaks_crime['combine_time'] >= label_time_frame[0]) & (sheroaks_crime['combine_time'] < label_time_frame[1])]
            crime_record_label = np.zeros(crime_type)
            for index_label, row_label in label_time_slots.iterrows():
                crime_record_label[int(row_label["crime_type_id"])] = 1
            label_list_so.append(crime_record_label)

        self.feature = np.array(final_list_so)
        self.label = np.array(label_list_so)

        pickle.dump(self.feature, open(data_folder +self.save_name + '_feature', 'wb'))
        pickle.dump(self.label, open(data_folder+self.save_name + '_label', 'wb'))

        print("Stored the feature data and the label data!")


        return final_list_so, final_list_so

    def generate_fllist_neighbor(self):
        print("Using the target's neighborhood information.")
        choose_neighborhood_id, chunk_size, time_frequency = self.choose_neighborhood_id, self.chunk_size, self.time_frequency
        print("the columns in the crime dataset is: {}".format(self.crime.columns) )
        crime_type = len(self.crime["crime_type"].unique())
        neighbors_crime = self.get_neighbor_crime_df()
        start_time_nbs = min(neighbors_crime["combine_time"])
        end_time_nbs = max(neighbors_crime["combine_time"])
        time_list_nbs = [dt.strftime('%Y-%m-%dT%H:%MZ') for dt in datetime_range(start_time_nbs, end_time_nbs, timedelta(minutes=time_frequency))]
        x_timechunks_nbs = list(moving_window(time_list_nbs, chunk_size))

        final_list_nbs = []
        label_list_nbs = []
        for i in range(0, len(x_timechunks_nbs)):
            feature_time_frame = x_timechunks_nbs[i][:chunk_size-1]
            feature_list = []
            for index_fea in range(0, len(feature_time_frame) - 1):
                start_nbs = feature_time_frame[index_fea]
                end_nbs = feature_time_frame[index_fea + 1]
                df_so_middle = neighbors_crime.loc[(neighbors_crime['combine_time'] >= start_nbs) & (neighbors_crime['combine_time'] < end_nbs)]
                crime_record = np.zeros(crime_type)
                for index, row in df_so_middle.iterrows():
                    crime_record[int(row["crime_type_id"])] = 1
                feature_list.append(crime_record)
            final_list_nbs.append(feature_list)

            label_time_frame = x_timechunks_nbs[i][chunk_size-2:]
            label_time_slots = neighbors_crime.loc[(neighbors_crime['combine_time'] >= label_time_frame[0]) & (neighbors_crime['combine_time'] < label_time_frame[1])]
            crime_record_label = np.zeros(crime_type)
            for index_label, row_label in label_time_slots.iterrows():
                crime_record_label[int(row_label["crime_type_id"])] = 1
            label_list_nbs.append(crime_record_label)

        self.feature = np.array(final_list_nbs)
        self.label = np.array(label_list_nbs)

        pickle.dump(self.feature, open(data_folder+self.save_name + '_feature', 'wb'))
        pickle.dump(self.label, open(data_folder+self.save_name + '_label', 'wb'))

        print("Stored the feature data and the label data!")
        return final_list_nbs, label_list_nbs

    def load_data(self):
        self.feature = np.array(pickle.load(open(data_folder+self.save_name + '_feature', 'rb')))
        self.label = np.array(pickle.load(open(data_folder+self.save_name + '_label', 'rb')))


    def choose_model(self):
        '''
        :param model: the name of models:random_forest, decision_tree, extra_tree, KNN, LR
        :return:
        '''
        feature = self.feature
        label = self.label
        model = self.model
        result_save_path = './'

        list_shape = np.shape(feature)
        print("feature shape: ", list_shape)
        second_dim = list_shape[1] * list_shape[2]
        shaped_final_list_so = np.reshape(feature, (list_shape[0], second_dim))
        num_samples = shaped_final_list_so.shape[0]
        num_test = round(num_samples * 0.2)
        num_train = round(num_samples * 0.7)

        X_train, X_test, y_train, y_test = shaped_final_list_so[:num_train], shaped_final_list_so[-num_test:], \
                                           label[:num_train], label[-num_test:]

        print("the shape of X_train: ", np.shape(X_train))
        print("the shape of y_train: ", np.shape(y_train))
        print("the shape of X_test: ", np.shape(X_test))
        print("the shape of y_test: ", np.shape(y_test))

        if model == 'random_forest':
            print("MODEL: RANDOM FOREST")
            clf_rf = RandomForestClassifier(max_depth=100, random_state=1, n_estimators=100)
            clf_rf.fit(X_train, y_train)
            y_predict = clf_rf.predict(X_test)
            average_precision_rfc = evaluate_micro_plot(y_test, y_predict, self.crime_type)
            print(average_precision_rfc)
            np.savez_compressed(result_save_path +self.save_name + '_rf', predict=y_predict, label=y_test)
            return average_precision_rfc

        elif model == 'decision_tree':
            print("MODEL: DECISION TREE")
            clf_dt = DecisionTreeClassifier(random_state=0)
            clf_dt.fit(X_train, y_train)
            y_predict_dt = clf_dt.predict(X_test)
            average_precision_dtc = evaluate_micro_plot(y_test, y_predict_dt, self.crime_type)
            print(average_precision_dtc)
            np.savez_compressed(result_save_path +self.save_name + '_dt', predict=y_predict_dt, label=y_test)
            return average_precision_dtc

        elif model == 'extra_tree':
            print("MODEL: EXTRA TREE")
            clf_et = ExtraTreesClassifier(random_state=0)
            clf_et.fit(X_train, y_train)
            y_predict_et = clf_et.predict(X_test)
            average_precision_etc = evaluate_micro_plot(y_test, y_predict_et, self.crime_type)
            print(average_precision_etc)
            np.savez_compressed(result_save_path +self.save_name + '_et', predict=y_predict_et, label=y_test)
            return average_precision_etc

        elif model == 'KNN':
            print("MODEL: KNN")
            neigh = KNeighborsClassifier(n_neighbors=3)
            neigh.fit(X_train, y_train)
            y_predict_neigh = neigh.predict(X_test)
            average_precision_neigh = evaluate_micro_plot(y_test, y_predict_neigh, self.crime_type)
            print(average_precision_neigh)
            np.savez_compressed(result_save_path +self.save_name + '_KNN', predict=y_predict_neigh, label=y_test)
            return average_precision_neigh

        elif model == 'LR':
            print("MODEL: LR")

            average_precision_neigh = {}
            average_precision_list = []
            LogReg_pipeline = Pipeline([('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),])

            prediction_list = []
            for category in range(0, self.crime_type):
                print('**Processing {} crimes...**'.format(category))
                # Training logistic regression model on train data
                LogReg_pipeline.fit(X_train, np.array(y_train)[:, category])
                # calculating test accuracy
                prediction = LogReg_pipeline.predict(X_test)
                average_precision_neigh[category] = average_precision_score(np.array(y_test)[:, category], prediction)
                average_precision_list.append(average_precision_score(np.array(y_test)[:, category], prediction))
                prediction_list.append(prediction)

            prediction_list = np.transpose(np.array(prediction_list))
            # y_test, prediction_list
            print(np.shape(y_test), np.shape(prediction_list))
            y_test = np.array(y_test)
            average_precision = average_precision_score(y_test, prediction_list, average="micro")
            print('Average precision score, micro-averaged over all classes: {0:0.2f}'
                  .format(average_precision))
            np.savez_compressed(result_save_path +self.save_name + '_LR', predict=prediction_list, label=y_test)

            macro_f1 = metrics_sk.f1_score(y_test, prediction_list, average = 'macro')
            micro_f1 = metrics_sk.f1_score(y_test, prediction_list, average = 'micro')

            print("The average macro-F1 score is {}, average micro-F1 score is {}".format(macro_f1, micro_f1))

            return average_precision_neigh

    def select_near_neighbors(self, threshold_rate):
        la_crime = self.crime
        # load the data from the file
        neighborhood_path = '../data/chicago_neighborhood.csv'
        neighborhood_info = pd.read_csv(open(neighborhood_path, 'r'), encoding='utf-8')
        distance_list = sorted(neighborhood_info["st_distance"])
        threshold_dis = distance_list[int(len(distance_list)*threshold_rate)]
        filtered_neighbod_df = neighborhood_info[neighborhood_info["st_distance"] <= threshold_dis]

        information_exist_list = filtered_neighbod_df["neighborhood"].unique()
        all_neighborhood_count = len(la_crime["neighborhood"].unique())
        neighbor_adja = np.zeros((all_neighborhood_count, all_neighborhood_count))
        for index_nei, row_nei in filtered_neighbod_df.iterrows():
            this_nei_name = row_nei["neighborhood"]
            if  this_nei_name in information_exist_list:
                neighbor_adja[int(this_nei_name)-1][int(row_nei["neighborhood-2"])-1] = 1
            else:
                raise Exception("This neighborhood does not exist in our database, exception raised.")
        self.threshold_rate = threshold_rate
        return neighbor_adja

    def get_neighbor_crime_df(self):
        choose_neighborhood_id = self.choose_neighborhood_id

        print("We are using the neighborhood information, the threshold rate is {}".format(self.threshold_rate))
        neighbor_adja = self.select_near_neighbors(self.threshold_rate)
        la_crime = self.crime
        neighbor_target_list = neighbor_adja[choose_neighborhood_id]
        neighbor_index_list = [in_n for in_n, e in enumerate(neighbor_target_list) if e != 0]
        if len(neighbor_index_list) > 5:
            print("Neighborhood {} has {} neighborhoods as neighbords, sampling needed!".format(
                choose_neighborhood_id, len(neighbor_index_list)))
            neighbor_index_list = random.sample(neighbor_index_list, 5)
        print("Now the amount of the neighbors we picked is {}".format(len(neighbor_index_list)))
        # the neighborhood id in the dict would be the index in matrix + 1
        nei_in_dict = [nid+1 for nid in neighbor_index_list]
        neighbors_crime = la_crime[la_crime["neighborhood"].isin(nei_in_dict)]
        return neighbors_crime

    def run(self):

        if os.path.exists(data_folder+self.save_name + '_feature'):
            print("the data is already precomputed, load the data")
            print("----------------------------------------------")
            self.load_data()
            print("finish loading the data")
            print("----------------------------------------------")
        else:
            print("the data is not precomputed, calculation starts")
            print("----------------------------------------------")
            self.preprocess_time()
            self.mapping_neighborhood_crime()
            if self.neighborhood_info == 'yes':
                self.generate_fllist_neighbor()
            elif self.neighborhood_info == 'no':
                self.choose_target_generate_fllist()
            print("calculation done")
            print("----------------------------------------------")

        # self.choose_model(model='random_forest')
        self.choose_model()
        # self.choose_model(model='decision_tree')
        # self.choose_model(model='extra_tree')
        # self.choose_model(model='KNN')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neighbor_id", type=int, default=37, help="The neighborhood ID you are interested in, Auburn Gresham: 71, "
                                                    "Fuller Park: 37, RIVERDALE: 54.")
    parser.add_argument(
        "--use_neighbor", type=str, default='no', help="if you want to use its neighbor information. "
    )

    parser.add_argument(
        "--model", type=str, default='random_forest', help='available models: random_forest, decision_tree, LR, KNN, '
                                                           'extra_tree'
    )

    args = parser.parse_args()
    print(args.use_neighbor, args.neighbor_id, args.model )
    ML_LA_target_sliding(neighborhood_info=args.use_neighbor, choose_neighborhood_id=args.neighbor_id,
                         choose_model=args.model).run()


