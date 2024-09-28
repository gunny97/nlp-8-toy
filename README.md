# 🔥 네이버 AI Tech NLP 8조 The AIluminator 🌟
## Level 1 Project - Semantic Text Similarity (STS)

## 목차
1. [프로젝트 소개](#1-프로젝트-소개)
2. [팀원 소개](#2-팀원-소개)
3. [사용한 데이터셋](#3-사용한-데이터셋)
4. [모델](#4-모델)
5. [프로젝트 구조](#5-프로젝트-구조)
6. [Installation and Quick Start](#6-installation-and-quick-start)

## 1. 프로젝트 소개
(1) 주제 및 목표
- 부스트캠프 AI Tech NLP 트랙 level 1 기초 대회
- 주제 : 문장 간 유사도 측정 (Semantic Text Similarity, STS)    
      STS 데이터셋을 활용해 두 문장의 유사도를 0 ~ 5 사이의 점수로 예측한다.  <br>

(2) 평가지표
- 피어슨 상관 계수(Pearson Correlation Coefficient ,PCC) <br>

(3) 개발 환경 <br>
- GPU : Tesla V100 * 4 <br>

(4) 협업 환경
- 노션 - 팀 노션 페이지에 해야할 일, 상황 공유    
- 슬랙 - 허들, DM을 활용해 팀원 간 실시간 소통   
- 깃허브 - 코드 공유

## 2. 팀원 소개
|김동한|김성훈|김수아|김현욱|송수빈|신수환|
|:--:|:--:|:--:|:--:|:--:|:--:|
|<img src="https://github.com/user-attachments/assets/c7d1807e-ef20-4c82-9a88-bc0eb5a700f4" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/62829d6a-13c9-40dd-807a-116347c1de11" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/5933a9e6-b5b8-41df-b050-c0a89ec19607" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/c90f4226-3bea-41d9-8b28-4d6227c1d254" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/65a7e762-b018-41fc-88f0-45d959c0effa" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/8d806852-764d-499b-a780-018b6cf32b8d" width="100" height="100" />|
|[Github]()|[Github]()|[Github](https://github.com/tndkkim)|[Github](https://github.com/hwk9764)|[Github](https://github.com/suvinn)|[Github]()|

### 맡은 역할
|**member**|**role**|
|:--|--|
|**김동한**|**EDA**(데이터 셋 특성 분석), **데이터 증강**(back translation), **모델링 및 튜닝**(Bert, Roberta, Albert, SBERT, WandB)|
|**김성훈**|**Model Exploration & Training**, **Modeling(Second-stream with GNN**, **Contrastive Learning, Soft Voting Ensemble)**, **Model Tuning(deliciouscat/kf-deberta-base-cross-sts, snunlp/KR-ELECTRA-discriminator)**|
|**김수아**|**EDA**(label 분포 및 문장 길이 분석)|
|**김현욱**|**EDA**(label 분포 분석), **데이터 증강**(Sentence Swap/Adverb Augmentation/BERT-Mask Insertion)|
|**송수빈**|**데이터 증강**(Downsampling/Sentence Swap/BERT-Mask Insertion/hanspell)|
|**신수환**|**모델링 및 튜닝**(RoBERTa, T5, SBERT), **모델 경량화**(Roberta-large with deepspeed)|
<br>

## 3. 사용한 데이터셋
데이터는 train.csv / dev.csv / test.csv의 3개의 파일로 되어있으며 각 파일의 column은 다음과 같이 구성되어있다. <br>
![Alt text](./markdownimg/data_column.png)  

**id** : 문장 고유 id <br>
**source** : 문장 출처 <br>
**sentence_1, sentence_2** : 유사성을 비교할 두 문장 <br>
**label** : 문장 쌍의 유사도. 0~5점 사이 값으로 소수점 첫째 자리까지 표현됨 <br>
**binary-label** : label 2.5점을 기준으로 0과 1로 변환한 값 <br>

### 데이터 분포
train data의 경우 label 0.0에 데이터가 쏠린 반면 dev data의 경우 비교적 균등하게 데이터가 분포되어있음을 알 수 있다. <br>
<img src="./markdownimg/train_dev_state.png" width="600" height="450"/> <br>
train data의 불균형을 해소하기 위해 label 0.0에 해당하는 데이터 수를 줄이고 여러 증강 기법들을을 활용하였다. <br>
<br>

### 데이터 증강
|**Version**|**Abstract**|**num**|
|:--:|--|:--:|
|**V1_Downsampling**|label 0.0 데이터 1000개 downsampling|8,324|
|**V2_augmentation_biased**|`AugmentationV1` + `BERT-Token Insertion`|9,994|
|**V3_augmentation_uniform**|`AugmentationV2` + `Adverb Augmentation` + `Sentence Swap` + `BERT-Token Insertion`|15,541|
|**V4_augmentation_spellcheck**|`AugmentationV2` + `hanspell` + `Sentence Swap` |17,313|

### 증강 데이터 버전 설명
|**Version**|**Description**|
|:--:|--|
|**V1_Downsampling** |Downsampling된 1000개의 문장으로 V2에서 (4.0, 5.0] label의 data augmentation을 진행할 것이기 때문에, label이 0.0인 데이터셋에서 문장 내 token 수가 3개 이상이면서, K-TACC 증강 방법 중 random_masking_insertion을 진행했을 때 증강이 되는 문장을 선별했습니다. sentence_1과 sentence_2 모두 증강된 index만 고려하면서, sentence_1을 기준으로 유사도가 높은 상위 1000개의 index를 선별했습니다. 문장 간 유사도가 고려되지 못한 sentence_2 데이터셋에 대해서는 추후 data filtering을 거쳤습니다.|
|**V2_augmentation_biassed**|V1에서 Downsampling된 1000개 데이터셋을 증강한 데이터셋 중에서도 label이 5.0인 데이터셋은 큰 차이가 없어야 한다고 판단하여, 불용어를 제거하면 같은 문장인 데이터를 label 5.0에 할당했습니다. label이 (4.0, 5.0)인 데이터셋은 라벨 간의 비율을 직접 조정하면서, 유사도가 높은 순서대로 개수에 맞게 할당했습니다.|
|**V3_augmentation_uniform**| label 분포를 균형있게 맞추어 전체적인 데이터 분포를 고르게 하기 위해 **라벨별 증강 비율을 조정**하여 총 3단계에 걸쳐 증강했고 매 단계마다 데이터의 개수가 적은 label들을 집중적으로 증강했습니다. <br> 1단계로 label이 `0.5, 1.5, 1.6, 2.2, 2.4, 2.5, 3.5` 데이터에 대해 Adverb Augmentation 수행했습니다. 2단계로 label이 `0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.8, 2.6, 2.8, 3, 3.2, 3.4, 3.5` 데이터에 대해 Sentence Swap 수행하였습니다. 3단계로 `1.5, 2.5, 3.5` 데이터에 대해 random_masking_insertion을 수행하였으며 추가로 `1.5, 2.5` 데이터 중 Masking Insertion한 증강 데이터에 대해 Sentence Swap을 수행했습니다.|
|**V4_augmentation_spellcheck**|label이 0.0인 데이터셋 중 맞춤법 교정 라이브러리 hanspell이 sentence_1과 sentence_2 모두에 적용된 index 776개를 뽑고, 증강된 데이터셋들을 label 4.8에 493개, label 5.0에 1059개 할당하였습니다. label이 (0.0, 4.4]인 데이터셋은 sentence swapping을 진행하였습니다. V2의 데이터셋 중 500개를 뽑아와 label 4.6에 450개, 4.5에 50개 할당하여 라벨 간 비율이 비숫해지도록 조정하였습니다.|


### 증강 데이터 분포
**V1_Downsampling**
|<img src="https://github.com/user-attachments/assets/543fb669-dc70-4f78-9cd6-8bbcd5307ca7" width="500" height="400" />|<img src="https://github.com/user-attachments/assets/dec4e8d2-8570-4f31-a301-dfeb82b2f209" width="500" height="400" />|
|:--:|:--:|
|label 별 분포|0.5단위 구간 별 분포|
<br>

**V2_augmentation_biased**
|<img src="https://github.com/user-attachments/assets/747863a6-5e62-41d9-ae5c-885c01a9a929" />|<img src="https://github.com/user-attachments/assets/348ccd06-07f4-49ae-9822-e776581c0995" />|
|:--:|:--:|
|label 별 분포|0.5단위 구간 별 분포|
<br>

**V3_augmentation_uniform**
|<img src="https://github.com/user-attachments/assets/ac0d5f75-4a50-48d7-b8a1-a106e274eefe" width="500" height="400" />|<img src="https://github.com/user-attachments/assets/33d6c69f-c602-4129-ae9f-5f5b98a87362" width="500" height="400" />|
|:--:|:--:|
|label 별 분포|0.5단위 구간 별 분포|
<br>

**V4_augmentation_spellcheck**
|<img src="https://github.com/user-attachments/assets/5be1d51a-96e6-4210-87c9-826a3dfd285c" width="500" height="400" />|<img src="https://github.com/user-attachments/assets/eb403cf1-9ccd-4a7f-80f1-50ba363f0861" width="500" height="400" />|
|:--:|:--:|
|label 별 분포|0.5단위 구간 별 분포|
<br>


## 4. 모델
|**Model**|**Learing Rate**|**Batch Size**|**loss**|**epoch**|**beta**|**Data Augmentation**|**Public Pearson**|**Ensemble Weight**|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|**candidate_v1_deberta_CL**|1e-5|16|L1|5|Spell Check|AugmentationV2|0.9125|0.9125|
|**candidate_v1_clean_deberta**|1e-5|16|MSE|2|Spell Check|AugmentationV3|0.9166|0.9166|
|**candidate_v2_roberta_after**|2e-5|32|L1|23|Spell Check|AugmentationV2|0.9216|0.9216|
|**candidate_v2_electra**|1e-5|32|L1|15||AugmentationV1|0.9179|0.9179|
|**candidate_v3_electra**|2e-5|32|L1|15|Spell Check|AugmentationV2|0.9217|0.9217|


## 5. 프로젝트 구조
```sh
.
├── model
│     ├── fine_tune_sts.py
│     └── SimCSE.py
├── preprocessing
│         ├── BERT_augmentation.py
│         ├── EDA.py
│         ├── adverb_augmentation.py
│         ├── v1_downsampling.ipynb
│         ├── v2_augmentation_biassed.ipynb
│         └── v3_augmentation_uniform.ipynb
├── resources
│       ├── log
│       └── sample 
├── utils
│     ├── data_modeul.py
│     └── helpers.py
└── environment.yml
├── train.py
├── train_unsup_CL.py
└── inference.py
```

## 6. Installation and Quick Start

**Step 1.** 해당 repository를 clone해서 사용해주세요.
**Step 2.** 프로젝트에 필요한 모든 dependencies는 `requirements.txt`에 있고, 이에 대한 가상환경을 생성해서 프로젝트를 실행합니다.
```sh
$ python -m venv .venv
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

**Step 3.** 본인의 가상환경에 원하는 Task 수행하시면 됩니다.
```sh
$ . .venv/bin/activate
$ wandb login # init set up : 본인의 wandb 계정을 한번 로그인 해두면 그 다음부터는 실행 안해도 됩니다.
$ python train.py
```

**Optional.** 원격 연결 끊어졌을 때도 돌아갈 수 있도록 Tmux 사용을 권장합니다. 더 자세한 명령어는 구글링 해주세요!
```sh
# 새로운 세션 생성
$ tmux new -s (session_name)

# 세션 목록
$ tmux ls

# 세션 시작하기 (다시 불러오기)
tmux attach -t (session_name)

# 세션에서 나가기
(ctrl + b) d

# 특정 세션 강제 종료
$ tmux kill-session -t (session_name)
```