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

![image](https://user-images.githubusercontent.com/46701548/121029033-3b389680-c7e3-11eb-9bb2-bf8ac78b3f34.png)

![image](https://user-images.githubusercontent.com/46701548/121029091-45f32b80-c7e3-11eb-8953-89b08c4c33bb.png)
