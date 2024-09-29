import argparse
from utils.ensemble_module import STS_Ensemble_module

def run_ensemble(config):
    models=config.model_path_list.split(',')
    scores=list(map(float,config.score_list.split(',')))
    ensemble=STS_Ensemble_module(
        model_list=models,
        score_list=scores
    )
    ensemble.load_model_predictions()

    average=ensemble.calculate_average()
    average.to_csv("average_ensemble.csv",index=False,encoding='utf-8-sig')
    weight_average=ensemble.calculate_weighted_average()
    weight_average.to_csv("weight_average_ensemble.csv",index=False,encoding='utf-8-sig')
    scaled_weight_average=ensemble.calculate_scaled_weighted_average()
    scaled_weight_average.to_csv("scaled_weight_average_ensemble.csv",index=False,encoding='utf-8-sig')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path_list", required=True, help="e.g. 'model_A,model_B,model_C'")
    parser.add_argument("--score_list", required=False, help="e.g. '0.9190,0.9110,0.9164'")

    args = parser.parse_args()

    run_ensemble(config=args)