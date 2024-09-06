import os
import argparse
from argparse import ArgumentParser


def add_options():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument("--train_data", default="resources/sample/sample_train.csv", type=str, help='상대 경로를 적어주세요!')
    parser.add_argument("--test_data", default="resources/sample/sample_test.csv", type=str, help='상대 경로를 적어주세요!')
    parser.add_argument("--target_name", default="topic", type=str, choices=['topic', 'keyword', 'speach_act'], help='topic과 keywrod는 같은 파일이고, speach_act는 또 다른 파일입니다.')
    
    # Model
    parser.add_argument("--pretrained_model", default="beomi/gemma-ko-2b", type=str, help='7b도 시험해보면 좋을 것 같아요!')
    parser.add_argument("--num_classes", default=9, type=int, help='데이터마다 다르게 설정해야 해요!')

    # Training
    parser.add_argument("--num_workers", default=os.cpu_count(), type=int)
    parser.add_argument("--lr", default=0.1, type=float, help='여러 단계로 실험해보면 좋을 것 같아요!') 
    parser.add_argument("--max_length", default=256, type=int, help='텍스트 길이에 따라 조정해야할 것 같아요.')
    parser.add_argument("--batch_size", default=8, type=int, help='학습할 때 병렬처리 최적화를 위해서 8, 16, 32 등 2의 제곱으로 지정해주세요')
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--min_delta", default=0.05, type=int, help='EarlyStopping 파라미터에요! 자세한 사항은 document 확인해주세요')
    parser.add_argument("--patience", default=4, type=int, help='EarlyStopping 파라미터에요! 자세한 사항은 document 확인해주세요')
    parser.add_argument("--model_checkpoint_dir", default="resources/log/sample", type=str, help='학습된 모델을 저장할 장소를 지정하는거에요. log 폴더 하위에 새로운 폴더 생성해서 실험해주세요!')
    parser.add_argument("--exp_name", default="test", type=str)
    

    args = parser.parse_args()
    return args