# 🔥 네이버 AI Tech NLP 8조 The AIluminator 🌟
## Level 1 Project - Semantic Text Similarity (STS)

## 목차
1. [프로젝트 소개](#1-프로젝트-소개)
2. [팀원 소개](#2-팀원-소개)
3. [사용한 데이터셋](#3-사용한-데이터셋)
4. [모델](#4-모델)
5. [프로젝트 구조](#5-프로젝트-구조)

## 1. 프로젝트 소개
1. 주제 및 목표 <br>
- 부스트캠프 AI Tech NLP 트랙 level 1 기초 대회
- 주제 : 문장 간 유사도 측정 (Semantic Text Similarity, STS)    
      : STS 데이터셋을 활용해 두 문장의 유사도를 0 ~ 5 사이의 점수로 예측한다.  
2. 평가지표 <br>
- 피어슨 상관 계수(Pearson Correlation Coefficient ,PCC)
3. 개발 환경 <br>
GPU : Tesla V100 * 4
4. 협업 환경 <br>
- 노션 - 팀 노션 페이지에 해야할 일, 상황 공유    
- 슬랙 - 허들, DM을 활용해 팀원 간 실시간 소통   
- 깃허브 - 코드 공유

## 2. 팀원 소개
|김동한|김성훈|김수아|김현욱|송수빈|신수환|
|:--:|:--:|:--:|:--:|:--:|:--:|
|![Alt text](./markdownimg/image-3.png)|![Alt text]()|![Alt text]()|<img src="https://github.com/user-attachments/assets/c90f4226-3bea-41d9-8b28-4d6227c1d254" width="100" height="100" />|![Alt text]()|![Alt text]()|
|[Github]()|[Github]()|[Github](https://github.com/tndkkim)|[Github](https://github.com/hwk9764)|[Github]()|[Github]()|

### 맡은 역할
<br>

|**팀원**|**역할**|
|:--|:--|
|**김동한**&nbsp;|**EDA**(`데이터 셋 특성 분석`), **데이터 증강**(`back translation`), **모델링 및 튜닝**(`Bert, Roberta, Albert, SBERT, WandB`)|
|**김성훈**&nbsp;|**EDA**(`label-pred 분포 분석`), **데이터 증강**(`back translation/nnp_sl_masking/어순도치/단순복제`), **모델 튜닝**(`roberta-large, kr-electra-discriminator`)|
|**김수아**&nbsp;|**EDA**(`label 분포 및 문장 길이 분석`)
|**김현욱**&nbsp;|**EDA**(`label 분포 분석`), **데이터 증강**(`/sentence swap/Adverb Augmentation/BERT-Mask Insertion`)|
|**송수빈**&nbsp;|**데이터 전처리**(`띄어쓰기 통일`), **데이터 증강**(`부사/고유명사 제거 Augmentation`), **모델링**(`KoSimCSE-roberta`), **앙상블**(`variance-based ensemble`)|
|**신수환**&nbsp;|**모델링 및 튜닝**(`RoBERTa, T5, SBERT`), **모델 경량화**(`Roberta-large with deepspeed`)|

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
![Alt text](./markdownimg/train_dev_state.png)  
train data의 불균형을 해소하기 위해 label 0.0에 해당하는 데이터 수를 줄이고 여러 증강 기법들을을 활용하였다. <br>
<br>

### 데이터 증강
|**Version**|**Abstract**|**개수**|
|:--:|--|:--:|
|**V1_Downsampling**|label 0인 값 천개 downsampling????|개수|
|**V2_augmentation_biased**|`원본 데이터` + `맞춤법 검사 데이터` + `SR` + `Swap Sentence` + `Copied Sentence`|개수|
|**V3_augmentation_uniform**|`AugmentationV2` + `Adverb Augmentation + Sentence Swap + BERT-Token Insertion`|개수|

### 증강 데이터 버전 설명
|**Version**|**Description**|
|:--:|--|
|**V1_Downsampling** | downsampling |
|**V2_augmentation_biased**| augmentation|
|**V3_augmentation_uniform**|1. `0.5, 1.5, 1.6, 2.2, 2.4, 2.5, 3.5 데이터에 대해 Adverb Augmentation 수행` <br> 2. `0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.8, 2.6, 2.8, 3, 3.2, 3.4, 3.5 데이터에 대해 Sentence Swap 수행` <br> 3. `1.5, 2.5, 3.5 데이터에 대해 BERT-Masking Insertion 수행` <br> * 데이터 증강 과정에서 라벨 분포를 균형있게 맞추고자 **라벨별 증강 비율을 조정**하였습니다.|

### 증강 데이터 분포
**V1_Downsampling**
<br>
![Alt text](./markdownimg/image-9.png)
<br>
**V2_augmentation_biased**
<br>
![Alt text](./markdownimg/image-9.png)
<br>
**V3_augmentation_uniform**
<br>
label 별 분포
<br>
![image](https://github.com/user-attachments/assets/4bac99f6-5b77-465a-8d34-6fb30441bc6e)
<br>
0.5 구간 별 분포
<br>
![image](https://github.com/user-attachments/assets/2518d00a-0b11-4ccb-9eba-709bac30ff76)
<br>

## 4. 모델
|**Model**|**Learing Rate**|**Batch Size**|**loss**|**epoch**|**beta**|**Data Augmentation**|**Public Pearson**|**Ensemble Weight**|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|**klue/RoBERTa-large**|1e-5|16|L1|5|Spell Check|AugmentationV2|0.9125|0.9125|
|**klue/RoBERTa-large**|1e-5|16|MSE|2|Spell Check|AugmentationV3|0.9166|0.9166|
|**kykim/electra-kor-base**|2e-5|32|L1|23|Spell Check|AugmentationV2|0.9216|0.9216|
|**snunlp/KR-ELECTRA-discriminator**|1e-5|32|L1|15||AugmentationV1|0.9179|0.9179|
|**snunlp/KR-ELECTRA-discriminator**|2e-5|32|L1|15|Spell Check|AugmentationV2|0.9217|0.9217|


## 5. 프로젝트 구조
```sh
.
├── model
│   ├── __init__.py
│   ├── fine_tune_cls.py
│   └── ... # 필요한 모델 생성
├── resources
│   ├── log
│   ├── sample
│   └── ... # 필요한 폴더 생성
├── utils
│   ├── config
│   ├── ├── cls_config.py
│   ├── └── ... # 필요한 config 생성
│   ├── loader
│   ├── ├── cls_datamodule.py
│   ├── └── ... # 필요한 datamodule 생성
│   ├── log
│   ├── helpers.py
│   ├── scheduler.py
│   └── ... # 필요한 util 함수 생성
└── train.py 
```
