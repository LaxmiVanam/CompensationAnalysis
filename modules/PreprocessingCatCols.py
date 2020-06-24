# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:18:19 2020

@author: laxmi
"""

class PreprocessingCatCols:
    ''''''
    def __init__ (self,df_train, df_test, category_cols):
            self.df_train = df_train
            self.df_test= df_test
            self.category_cols = list(category_cols)        
            self.existing_label_encoded_cols = {}
                   
    def label_encode_check(self,df,cols):
        '''Private method to check for existence of label encoders for categorical columns and uses it. Otherwise, it creates new label encoders'''
        for col in cols:                    
            if col in self.existing_label_encoded_cols:
                self._encode_now(df,col,self.existing_label_encoded_cols[col].all())                 
            else:
                self._encode_now(df,col)     
        return self.encoded_df        
                
    def inverse_encode_check(self,cols):
        '''Method to inverse label encoded values to the original values'''
        for col in cols:                    
            if col in self.existing_label_encoded_cols:
                self._inverselabel_encode_now(df,col)                 
            else:
                raise ValueError("Label incoders must be defined before calling inverse function")    
                
                                  
    def _inverse_encode_now(self,col):
        '''Private method to create label encoders for the given categorical column and adds them to the dictionary'''    
        le = self.existing_label_encoded_cols[col]
        df[col]  = le.inverse_transform(df[col])       
                                
    def _encode_now(self,df,col, le = None):
        '''Private method to create label encoders for the given categorical column and adds them to the dictionary'''    
        if le:     
            df[col] = le.transform(df[col])
        else:                
            le = LabelEncoder()
            le.fit(df[col])
            df[col] = le.transform(df[col])
            self.existing_label_encoded_cols[col] = df[col] 
            self.encoded_df =  pd.DataFrame(self.existing_label_encoded_cols)
            
    def concat_encodedcat_numericfields(self, df1, df2) :
        return pd.concat([df1,df2], axis=1)
        