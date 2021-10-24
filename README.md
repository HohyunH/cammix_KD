# cammix_KD

#### 2021-1 Deep-learning class team project

### How to Use

<pre>
<code>

python kd_main.py --alpha 0.5 --T 0.1 --teacher resnet18 --teacher resnet18 --model resnet18 --cam grad --cammix_prob 0.7 --epoch 5 --device [cuda, cpu] --method baseline

</code>
</pre>

## Introduction

#### 연구배경
- 딥러닝 모델의 성능과 학습 속도는 layer의 개수에 의존하면 서로 trade-off 관계를 이루고 있다.
- 모바일 기기나 차량, 테블릿과 같은 임베디드 기기의 성능이 증가하여 이에 장착할 수 있는 모델을 만들기 위해 복잡도가 낮으면서 성능을 보존하는 모델을 찾고자 하는 연구들이 진행됨

#### 연구동기
- Knowledge distillation은 teacher 모델의 지식을 student 모델에 전달하여 빠른 학습 속도와 안정적인 성능을 보장하고자 함
- Data augmentation은 머신러닝 모델에 사용되는 전통적인 정규화 기법 중 하나로 모델의 성능을 높이는데 주요한 역할을 함

#### 연구목적
- Knowledge distillation 알고리즘에 data augmentation 기법을 적용하여 모델의 성능을 향상시키고자 함
- Cutmix를 Grad-CAM과 결합한 augmentation을 활용하여 distillation 작업에 성능 향상을 기대함

### Model Framework

- teacher model에서 해당 이미지에 gradcam을 측정해서, student model을 학습할 때 CAM 값이 높은 부분을 crop하여 cutmix방식 처럼 활용한다.

![image](https://user-images.githubusercontent.com/46701548/121029033-3b389680-c7e3-11eb-9bb2-bf8ac78b3f34.png)

![image](https://user-images.githubusercontent.com/46701548/121029091-45f32b80-c7e3-11eb-8953-89b08c4c33bb.png)

### KD Loss
![image](https://user-images.githubusercontent.com/46701548/138590895-614b9aa5-de1b-4060-a410-f5ca3af56dd5.png)
여기서 L은 손실함수, S는 Student model, T는 Teacher model을 의미합니다. 또한 (x,y)는 하나의 이미지와 그 레이블,θ는 모델의 학습 파라미터, τ는 temperature를 의미.


### Experiment

- Dataset : Intel scene image dataset
- https://www.kaggle.com/spsayakpaul/intel-scene-classification-challenge/data
- 총 데이터 수( train : 15k, test : 3k ) : 6 categories (buildings, forest, glacier, mountain, sea, street)
- Image Shape : 3x224x224

#### Hyper-parameter 설정
|Hyper-parameter|Options|
|---|---|
|Epochs|10|
|Batch_size|32|
|Learning rate|0.001|
|Alpha|0.2|
|Temperature|0.1|
|Cammix probability|0.3|

#### Experiment Results
![image](https://user-images.githubusercontent.com/46701548/138590874-a2779e54-6319-40af-b75e-cb6c94f5ad0e.png)
