# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:19:32 2020

@author: laxmi
"""

class ModelContainer:
   
    def __init__(self,modellist = []):
        ''' To initialize instance variables passed through object'''    
        self.modellist = modellist
        self.scores = {} 
                
    def add_model(self, modelname):
        '''To add a new model to the list'''
        self.modellist.append(modelname)
        
    def cross_validate(self,  features,target,  k = 5, num_procs = 1):
        '''To cross validate models using given data, k value and the number of processes'''
        self.trainfeatures = features#data.train_df.iloc[:,2:-1]
        self.traintarget = target#data.train_df.iloc[:,-1]
        for model in self.modellist:
            self.scores[model] = cross_val_score(model,self.trainfeatures, self.traintarget,cv = k,scoring=('neg_mean_squared_error'))     
            self.scores[model] =  -1*np.mean(self.scores[model])
           
    def select_best_model(self):
        '''To choose the best model of all the given ones by the score'''
        self.best_model = min(self.scores, key = self.scores.get)
        return self.best_model
    
        
    def best_model_fit(self,traifeatures,traintarget):
        '''To fit the best model to the train dataset'''  
        self.best_model.fit(traifeatures, traintarget)
        
    def best_model_predict(self,testfeatures):
        '''To predict the target value for the test dataset'''
        self.predictions = self.best_model.predict(testfeatures)    
        
    def save_results(self):
        ''' To save the test results if needed'''
        pass
    
    @staticmethod     # use as decorator   
    def get_feature_importance(model, cols):
        '''Static method which can be accessed outside the class with no self/ cls parameter as well'''
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            feature_imp_df = pd.DataFrame({'Features':cols,'Importance' : feature_importance})         
            feature_imp_df.sort_values(by = 'Importance', ascending = False,inplace = True )
            feature_imp_df.set_index('Features', inplace=True, drop=True)
            feature_imp_df.plot.bar()
            plt.show()
        else:
            print("Feature Importance does not exist for the current model")
        
    def print_summary(self):
        '''To print summary of the final model'''
        print('\nModel Summary:\n')
        self.get_feature_importance (self.best_model,data.train_df.iloc[:,2:-1].columns) 
        
      
            