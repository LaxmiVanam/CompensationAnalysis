# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:23:31 2020

@author: laxmi
"""
#import packages
import numpy as np
import pandas as pd

#Preprocessing
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

#modeling
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

#Import modules
import Dataprep.py
import PreprocessingCatCols.py
import Addinggroupstats.py
import ModelContainer.py


if __name__ == '__main__':
    #Define input files path
    input_file_path = 'C:/Users/laxmi/CompAnalysis/rawdata/'

    #Define input files
    train_feature_file = input_file_path + 'train_features.csv'
    train_target_file =  input_file_path + 'train_salaries.csv'
    test_feature_file =  input_file_path + 'test_features.csv'

    #Define other parameters for Data class
    category_cols = ['companyId','jobType','degree','major','industry']
    numeric_cols = ['yearsExperience','milesFromMetropolis']
    target = 'salary'
    id_col = 'jobId'
    
    
    data = Dataprep(train_feature_file,train_target_file,test_feature_file, target, id_col)
    
    data.print_dataframe_shape(data.train_df, 'train')
    data.print_dataframe_shape(data.test_df, 'test')


    PreprocessingCatCols = PreprocessingCatCols( data.train_df,data.test_df,  category_cols)
    
    
    
    encoded_train_df = PreprocessingCatCols.label_encode_check(data.train_df,   PreprocessingCatCols.category_cols)
    
    encoded_test_df = PreprocessingCatCols.label_encode_check(data.test_df,   PreprocessingCatCols.category_cols)
    final_train_df = PreprocessingCatCols.concat_encodedcat_numericfields(encoded_train_df,data.train_df[numeric_cols])
    final_test_df = PreprocessingCatCols.concat_encodedcat_numericfields(encoded_test_df,data.train_df[numeric_cols])
    
    #Label for turning feature engineering on/off
    Addinggroupstats_label = True

    #Parameters for ModelContainer class
    num_procs = 4   #Number of processes for parallel runs
    verbose_lvl = 0 #Verbose level for modeling

    if Addinggroupstats_label:
        feature_generator = Addinggroupstats(data)
        feature_generator.add_group_statistics()
        
        
    # modelcontainer = ModelContainer()
    modelcontainer.add_model(LinearRegression(normalize = True))

    modelcontainer.add_model(RandomForestRegressor(n_estimators = 100,n_jobs = num_procs, max_depth = 15, min_samples_split = 80,   verbose = verbose_lvl))

    modelcontainer.add_model(GradientBoostingRegressor(n_estimators = 200, max_depth = 15, loss = 'ls', min_samples_split = 80,   verbose = verbose_lvl))

    modelcontainer.cross_validate(feature_generator.merged_data,data.train_df.iloc[:,-1])
    modelcontainer.scores   
    modelcontainer.select_best_model()
    modelcontainer.best_model_fit(data.train_df.iloc[:,1:-1],data.train_df.iloc[:,-1])
    
    modelcontainer.best_model_predict(data.test_df.iloc[:,1:])
    modelcontainer.print_summary()