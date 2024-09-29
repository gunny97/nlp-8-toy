import pandas as pd
import numpy as np

model_list=['model_A','model_B','modle_C']

ensemble_df=pd.DataFrame()
for model in model_list:
    df=pd.read_csv(model+".csv")
    ensemble_df=pd.concat([ensemble_df,df['target']],axis=1)
    
ensemble_df.columns=model_list

# 1. average
average=round(ensemble_df.sum(axis=1)/len(model_list),1)

print(average)
average.to_csv("submission_a.csv",encoding='utf-8-sig')
# 2. weight average
score_list=[0.9190,0.9110,0.9164]
weighted_df=ensemble_df.multiply(score_list,axis=1)
weighted_average=round(weighted_df.sum(axis=1)/sum(score_list),1)

print(weighted_average)
weighted_average.to_csv("submission_b.csv",encoding='utf-8-sig')

# 3. scaled average
min_weight = min(score_list)
max_weight = max(score_list)

scaled_weights = 0.8 + ((np.array(score_list) - min_weight) / (max_weight - min_weight)) * (1.2 - 0.8)

scaled_weighted_df = ensemble_df.multiply(scaled_weights, axis=1)
scaled_weighted_average = round(scaled_weighted_df.sum(axis=1) / sum(scaled_weights), 1)
print(scaled_weighted_average)
scaled_weighted_average.to_csv("submission_c.csv",encoding='utf-8-sig')