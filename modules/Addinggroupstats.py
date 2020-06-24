# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:19:08 2020

@author: laxmi
"""

class Addinggroupstats:
    
    def __init__ (self,data, cols_to_filter = None):
        '''To initialize instance variSEz2ables passed through object'''
        self.data = data
        self.target_col = 'salary'
        self.cols_to_filter = ['yearsExperience','milesFromMetropolis']
        self.categoies_for_grouping = ['jobType','degree','major','industry']
        self.groups = data.train_df.groupby(self.categoies_for_grouping)
        self.train_features = data.train_df.iloc[:,2:-1]
    
    def add_group_statistics(self):        
        '''To add group statistics for each of the categorical columns'''
        self.group_stats_df = self._get_group_statistics()  
        self.merged_data = self._merge_derived_columns_to_original(final_train_df, self.group_stats_df, self.categoies_for_grouping)
        return self.merged_data
    
    def  _get_group_statistics(self):
        '''To calculate various statistics of target salary'''
        #target_col = self.data.target
        group_stats_df = pd.DataFrame({ 'group_mean': self.groups[self.target_col].mean(),
                                      'group_median': self.groups[self.target_col].median(),
                                         'group_max': self.groups[self.target_col].max(),
                                         'group_min': self.groups[self.target_col].min(),
                                         'group_std': self.groups[self.target_col].std()})
        group_stats_df.reset_index(inplace = True)
        return group_stats_df
    
    def _merge_derived_columns_to_original(self,df1, df2, keys , how = 'left',fillna = False):
        '''To merge the statistics to the original columns in train data frame'''
        merged_df = pd.merge(df1, df2, on=keys)
        merged_df.fillna(0, inplace = True)
        return merged_df
        