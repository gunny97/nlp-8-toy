import pandas as pd
import numpy as np


class STS_Ensemble_module:
    def  __init__(self,model_list,score_list=None):
        self.model_list=model_list
        self.score_list=score_list if score_list is not None else [0] * len(model_list)
        self.ensemble_df=None
        
    def load_model_predictions(self):
        self.ensemble_df = pd.DataFrame()
        for model in self.model_list:
            df = pd.read_csv(model + ".csv")
            self.ensemble_df = pd.concat([self.ensemble_df, df['target']], axis=1)
        self.ensemble_df.columns = self.model_list

    def calculate_average(self):
        average = round(self.ensemble_df.sum(axis=1) / len(self.ensemble_df.columns), 1)
        average_df=pd.DataFrame(average,columns=['average'])
        return average_df
        

    def calculate_weighted_average(self):
        weighted_df = self.ensemble_df.multiply(self.score_list, axis=1)
        weighted_average = round(weighted_df.sum(axis=1) / sum(self.score_list), 1)
        weighted_average_df=pd.DataFrame(weighted_average,columns=['weighted_average'])
        return weighted_average_df

    def calculate_scaled_weighted_average(self):
        min_weight = min(self.score_list)
        max_weight = max(self.score_list)
        scaled_weights = 0.8 + ((np.array(self.score_list) - min_weight) / (max_weight - min_weight)) * (1.2 - 0.8)
        scaled_weighted_df = self.ensemble_df.multiply(scaled_weights, axis=1)
        scaled_weighted_average = round(scaled_weighted_df.sum(axis=1) / sum(scaled_weights), 1)
        scaled_weighted_average_df=pd.DataFrame(scaled_weighted_average,columns=['scaled_weighted_average'])
        return scaled_weighted_average_df