# CrimeForecaster

The Dataset and code description for the paper titled "CrimeForecaster: Crime Prediction by Exploiting the Geographical Neighborhoods’ Spatiotemporal Dependencies", accepted at ECML-PKDD 2020.

If you want to use our code/data or find CrimeForecaster insightful to your research, please cite our work:

```
@inproceedings{Sun2020CrimeForecasterCP,
  title={CrimeForecaster: Crime Prediction by Exploiting the Geographical Neighborhoods' Spatiotemporal Dependencies},
  author={Jiao Sun and Mingxuan Yue and Zongyu Lin and Xiaochen Yang and L. Nocera and Gabriel Kahn and C. Shahabi},
  booktitle={ECML/PKDD},
  year={2020}
}
```

## Dataset Description

We release a 10-year crime dataset we collected in Los Angeles, the two dataset we used for the experiment (crime data of Los Angeles in 2018, crime data of Chicago in 2015), together with the geographical dataset for Los Angeles and Chicago.

### Crime Data

1. **[10-year LA data]** The clean version of 10-year LA crime record data from 2010-01-01 to 2019-10-07 is available  here (https://drive.google.com/open?id=1nbqr0gdp_bO2QaIRN6qPx-7Tvd5pKa0P), If you want to use it for other purposes, please use ``pickle.load(open(file_path), 'rb')`` to read it, and cite our paper.
2. **[Los Angeles 2018 crime data for experiment]**  You can access the data here (https://drive.google.com/open?id=13xXr22bMNnjBDis7IR6TpISzcveyY-_E)
3. **[Chicago 2015 crime data for experiment]** You can access the data here (https://drive.google.com/open?id=1a3veMTk1sULWD1l8w_MZ_1qtjFrOOD-y)

### Geographical Data

4. **[LA Geographical data]**  ``Geographical-LA`` directory contains the information of 133 neighborhoods in the City of Los Angeles as ``neighborhood_info.csv`` and their connectivity information as ``neighborhood_dis.csv``.
5. **[Chicago Geographical data]** ``Geographical-Chicago`` directory contains the connectivity information of 77 neighborhoods in Chicago as ``chicago_neighborhood.csv`` and their boudary information ``Boundaries - Neighborhoods.zip``.



## Code Description

### 1. Code for CrimeForecaster 

`cd CrimeForecaster`

1. Prerequisite: scipy>=0.19.0; numpy>=1.12.1; pandas>=0.19.2; pyyaml; statsmodels; tensorflow==1.5.0

   You can install all of them by 

   `pip install -r requirement.txt`

2. We give an example of using the data from January 1st to July 15th as training data, July 15th to July 31th as validation data and August as test data in CrimeForecaster.

   - For Chicago 

     `python cf_train.py --config_filename=CRIME-CHICAGO/chicago_crime.yaml`

   - For LA 

     `python cf_train.py --config_filename=CRIME-LA/LA_crime.yaml`

**NOTICE**: please notice that you need to generate the adjacency matrix for both the cities first, we offer our matrix in the ``CrimeForecaster/graph`` folder,  if you want to regenerate them. Please use the scripts in `scripts/gen_adj_mx.py`. Here we give an example:

````shell
python scripts/gen_adj_mx.py --sensor_ids_filename=graph/neighborhood_ids_la.txt --distances_filename=graph/distance_neighborhood_la.csv
````

### 2. Code for Baseline Methods

- We implement the current state-of-the-art method MiST in ``baseline/MiST.py``
- We offer the code of most of the baselines we metion in the paper
  -  ``baseline/arima.py``: the time series model ARIMA
  - ``baseline/basic_ML.py`` contains the code for LR/Decision tree/Logistic Regression/Extra Tree/KNN/SVR/Random Forest and etc.

### 3. Miscellaneous: scripts we use for generating the training data

1. Code for extracting the 2018 Los Angeles data from the full dataset

   ``code/scripts/extract_LA_year.py``

2. Code for generating the feature and labels for Chicago data: ``code/scripts/generate_feature_label_Chicago.py``

   Please pay attention to the comment in the script 

   ````python
   #Note that please change the directory to your directory of the Chicago crime data. You can download it from https://drive.google.com/open?id=1a3veMTk1sULWD1l8w_MZ_1qtjFrOOD-y
   ````

3. Code for splitting the data to train/validation/test: ``code/scripts/CrimeForecaster_preparation.py``

   ````python
   #replace the "feature_generated.npy" and "label_generated.npy" by the files you generated using generate_feature_label_Chicago.py
   ````

### 4. Code for Appendix: when we have a target neighborhood

- Available models: Logistic Regression, random forest, decision tree, extra tree, K-nearest-neighbor classifiers

- **LA example**

  ```Shell
  python target_neighborhood/ML_LA_day_target_sliding_ns.py --neighbor_id=36 --use_neighbor='no' model='LR'
  ```

  The neighborhood ID journalists whom we collaborated with are interested in: sherman-oaks: 33, studio-city: 36, echo-park: 31

  **Chicago example**

  ```Shell
  python target_neighborhood/ML_Chicago_day_target_sliding_ns.py --neighbor_id=36 --use_neighbor='no' model='LR'
  ```

  The neighborhood ID journalists you may be interested in: Auburn Gresham: 71, Fuller Park: 37, RIVERDALE: 54

 
