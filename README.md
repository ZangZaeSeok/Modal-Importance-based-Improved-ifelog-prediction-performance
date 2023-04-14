# Modal-Importance-based-Improved-lifelog-prediction-performance
- 본 연구는 '라이프로그 예측 성능 향상을 위한 다중 인스턴스 기반 개인별 모달 주요도 추출 및 그룹화' 논문에서 구현한 실험을 재현할 수 있게 하는 코드입니다.
- 멀티 모달 센서 데이터를 활용해서 사용자들의 라이프로그를 예측하는 문제는 개인의 다양성으로 인해 발생되는 편차로 인해 어려움이 있습니다.
- 본 연구에서는 각 개인에게서 사용자들의 라이프로그 예측에 영향을 많이 주는 모달 주요도를 설명가능한 모델인 Mulitple Instance learning을 최초로 사용해서 추출하고, 추출한 모달 주요도를 사용해서 그룹별 예측 모델이 전체 데이터를 사용한 방법론보다 성능이 좋다는 것을 보였습니다.

## 코드 구성
> human_lifelog_mil_pytorch
>> 본 연구에서 사용한 모달 주요도를 추출하는 Multiple Instance Learning model의 structure과 그룹별 라이프로그 예측 모델 structure, 그 외에 실험에 필요한 기본적인 코드들이 있는 디렉토리입니다.
>>> FocalLoss.py
>>> - 본 연구에서 예측하는 라벨인 불균형이 있기 때문에, 불균형한 분류 문제에 도움이 되는 FocalLoss를 사용했습니다.  
>>> - https://github.com/AdeelH/pytorch-multi-class-focal-loss 에서 구현된 코드를 사용했습니다.  
>>> human_lifelog_mil_pytorch.py  
>>> - 멀티 모달 문제를 Multiple Instnace Learning 구조로 해결하게 해주는 코드입니다.  
>>> human_lifelog_predictor.py  
>>> - 각 그룹 맞춤 라이프로그 예측 모델 class가 있는 코드입니다.  
>>> modules.py  
>>> - 본 연구에서는 길이가 매우 긴 시계열 데이터의 정보를 빠르고 잘 반영하는 Cuasal dilated Convolution을 인코더로 사용했습니다.  
>>> - https://github.com/flaviagiammarino/usrl-mts-pytorch에서 구현된 코드를 사용했습니다.  
>>> utils.py  
>>> - multi classification 문제에서 F1 score을 구하기 위해 따로 구현한 코드입니다.  

> Dataset Preprocessor.ipynb
