class Dataprep:    
    
    def __init__ (self, train_feature_file, train_target_file, test_feature_file,  
                    target, id_col):
        '''To initialize instance variables passed through object'''
        
        self.id_col = id_col
        self.target = target
        self.train_df = self._create_train_dataframe(train_feature_file,train_target_file)      
        self.test_df = self._create_test_dataframe(test_feature_file)
        
   
    
    def _create_train_dataframe(self,train_feature_file,train_target_file,validate_train_files = True,clean_dataset = True
                                ):
        '''Private method to create train dataframe by preprocessing and label encoding categorical columns'''
        self.train_feature_df = self._load_data(train_feature_file)
        self.train_target_df = self._load_data(train_target_file)    
        
        if validate_train_files:
            self.validate_train_records( self.train_feature_df, self.train_target_df,self.id_col)
            train_df = self._merge_dfs(self.train_feature_df,self.train_target_df,self.id_col)    
            
        if clean_dataset:
            train_df = self._cleandataframe(train_df,id_col,target)
            train_df = self._shuffledataframe(train_df)
            
             
        return train_df    
    
    def _create_test_dataframe(self,test_file ):
        '''Private method to create test dataframe by label encoding categorical columns'''
     
        test_df = self._load_data(test_file)   
        
        #if label_encode:            
            #self._label_encode_check(test_df,self.category_cols)            
        return test_df    
    
    def print_dataframe_shape(self, df, df_name):
        '''To print the shape of the train/ test dataset'''
        print('The shape of the %s dataset is %s' %(df_name, df.shape))     
           
    
    def _load_data(self,file):
        '''Private method to load input files to pandas dataframes'''
        return pd.read_csv(file)
    
    def validate_train_records(self,train_feature_df, train_target_df, id_col):  
        if ((train_feature_df[id_col].nunique() != len(train_feature_df)) 
            | (train_target_df[id_col].nunique() != len(train_target_df))):
            print ('Duplicate ID records exist in the dataframes')
        print('The number of records in features file that are not in target file are',len(set(train_feature_df[id_col])) - len(set(train_target_df[id_col])))
        print('The number of records in target file that are not in features file are',len(set(train_target_df[id_col])) - len(set(train_feature_df[id_col])))
        
    
    def _merge_dfs(self,dataframe1,dataframe2, key , how = 'inner',left_index = False, right_index = False):
        '''Private method to merge train features and the target column for train dataframe'''
        return pd.merge(dataframe1,dataframe2,on = key)#.reset_index(drop = True, inplace = True)
    
   
    def _cleandataframe(self,df,id_col, target): 
        '''Private method to drop duplicates and also the records with no salary from training dataframe'''
        df = df.drop_duplicates(subset = id_col)
        df = df[df[target] >0]
        return df
    
    def _shuffledataframe(self,df): 
        '''Private method to shuffle train dataset for modeling'''
        df = shuffle(df).reset_index(drop = True)
        return df    
        