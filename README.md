# 🔥 네이버 AI Tech NLP 8조 The AIluminator 🌟


## Table of Contents
1. [Installation and Quick Start](#1-installation-and-quick-start)
2. [Structure](#2-project-structure)
3. [Rule](#3-project-rule)
4. [Task](#4-project-task)

## 1. Installation and Quick Start

**Step 1.** 해당 repository를 clone해서 사용해주세요.

```sh
$ git clone https://github.com/gunny97/nlp-8-toy.git
$ cd nlp-8-toy
```
**Step 2.** 프로젝트에 필요한 모든 dependencies는 `environment.yml`에 있고, 이에 대한 가상환경을 생성해서 프로젝트를 실행합니다. 아래는 `conda`(miniconda 추천)를 이용해 가상환경을 만들기 위한 command 입니다.
- 프로젝트를 진행하면서 필요한 dependencies는 `environment.yml`에 추가해주시고, 변경사항에 대해서는 `issue`에 올려주세요!
```sh
# create -n 다음 나오는 'toy'는 가상환경의 이름으로 원하시는 명칭으로 변경해서 사용해주세요. 설치하는데 시간이 좀 걸려요!
$ conda env create -n toy -f environment.yml
# 콘다 가상환경 확인
$ conda info --envs
```
**Step 3.** 본인의 가상환경에 원하는 Task 수행하시면 됩니다.
- train.py를 돌릴 때 버그가 발생할 수 있습니다. EarlyStopping으로 Test 단계에 들어갈 때 `TypeError: cannot unpack non-iterable NoneType object`가 뜨는데, 서버 환경 (4개의 GPU를 사용하는 서버)에 따라 에러가 발생하는 것 같습니다. Training은 완료된 것이니 Test만 따로 돌려주시면 됩니다.
```sh
$ conda activate toy
# init set up : 본인의 wandb 계정을 처음에만 입력하면 됩니다.
$ wandb login
$ python train.py
```

**Optional.** 원격 연결 끊어졌을 때도 돌아살 수 있도록 Tmux 사용을 권장합니다! 더 자세한 명령어는 구글링 해주세요!
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


## 2. Project Structure
Project Structure
### 📝 알아두면 좋은 몇가지 관례
* 함수는 `헬퍼 함수` 또는 `API 함수`로 분류할 수 있습니다. `헬퍼 함수`임을 나타내기 위해 `_` 를 함수의 이름 앞에 표기합니다. `_`는 주로 비공개 또는 내부용이라는 의미입니다. 본인의 전처리 코드는 `_`를 사용해서 넣어주시면 좋을 것 같아요!
```python
# 모델에서 사용된 헬퍼 함수 예시입니다.
class TransformerModule(LightningModule):
    def __init__(self,):

    ...
    # API 함수입니다. 학습할 때 Trainer가 자동으로 해당 함수를 사용합니다.
    def forward(self): 
    
    # 헬퍼 함수입니다. 학습 및 테스트 과정 중 제가 원하는 작업을 할 때 사용합니다.   
    def _compute_metrics(self, pred_class, label, prefix) -> tuple:
      metrics = {
          f"{prefix}_Acc": multiclass_accuracy(
              preds=pred_class,
              target=label,
              num_classes=self.num_classes
          )
      return metrics
```
* `__init__.py` 파일은 모듈 내 함수의 공개 범위를 설정할 수 있습니다. 특정 모듈이나 함수만을 `__init__.py`에서 임포트하면, 패키지를 사용하는 사람은 내부 구조를 알 필요 없이 간단하게 사용할 수 있습니다. 
```python
# __init__.py에서 임포트
from .fine_tune_cls import TransformerModule
```
```python
# train.py에서는 다음과 같이 줄여서 사용할 수 있습니다.
from model import TransformerModule

# 만약 __init__.py를 설정하지 않으면 다음과 같이 상대경로를 작성합니다. 
from model.fine_tune_cls import TransformerModule
```

### 🚶다음은 주요 모듈에 대한 설명입니다.
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
다른 Task 수행을 위해 참고해주세요!
* `model`: Specifies한 Task를 수행하는 모델들을 작성해서 넣어주세요. 기본적인 구조는 `pytorch lighting`을 따르고 있어, 헬퍼 함수를 통해 본인만의 전처리 코드를 작성하시면 됩니다.  `fine_tune_cls.py`를 참고해주세요.
* `config`: 모델 학습에 필요한 파라미터들을 `.py`에 작성해주세요. `cls_config.py`를 한번 꼭 봐주세요! 
* `loader`: 모델 학습에 필요한 `datamodule`을 작성해주세요. 마찬가지로 `pytorch lighting`구조이고, 헬퍼 함수를 사용해서 전처리 코드 작성해주세요.
* `utils` : Task를 수행하면서 필요한 여러 함수들을 넣어주세요.
* `train.py`: 메인 실행파일로 생각하시면 됩니다. 위에서 정의한 모듈들을 불러와서 실행합니다. 현재는 cls task를 수행하도록 작성했습니다. 새로운 Task를 수행할 때는 `train_generation.py`과 같이 새로운 실행파일을 만들어서 사용해주세요. 추후 모든 `train_#.py`를 병합할 예정입니다.
* `test.py`: 현재는 없지는 테스트를 위한 파일을 하나 만들어주세요!

## 3. Project Rule
**Rule 1.** 모든 작업자는 새로운 Task를 수행할 때, 새로운 브랜치를 만들어 작업해주세요. 모든 작업이 끝난 후 `main` 브랜치에 `merge` 합니다.


**Rule 2.** 프로젝트 내 이슈가 발생했을 때 `Issues`에 올려주세요. 이슈는 프로젝트 구조 변경, 모듈 질의응답, 버그 발생 등 이슈라고 생각하는 건 다 올려주세요! 슬랙이나 피어세션 시간에 한 이야기라도 올려주세요! 기록으로 남기려고 합니다.


**Rule 3.** `main` 브랜치에 `merge`할때는 모두 다 같이 `review`하면 좋을 것 같습니다.

**Rule #.** 언제든 필요로 하는 rule을 생성해주세요!


## 4. Project Task
저희가 앞으로 진행한 모든 Task의 내용을 간단하게 남길 예정입니다. 포트폴리오로를 만든다고 생각하면 좋을 것 같아요!

### Text Classification with LLM
추후 기록.
### Text Summarization with LLM
추후 기록.
### Text Semantic Text Similarity
추후 기록.
