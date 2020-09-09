- 9/8 공부 내용
  - DataLoader 공부하던 도중 관련된 사이트를 찾아서 공부
    - https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html
    - 실습 : https://wingnim.tistory.com/33
  - 해결하지 못 한 내용
    - data_transform 부분에서, normalize 행렬을 어떤 기준으로 설정한건지
    - 매 step마다 optimizer.zero_grad() 처리를 하는 이유가 뭔지
    - 매 step마다 loss, critereon 값을 mixup 하는 이유가 뭔지
    - 맨 처음에, torch의 모든 함수들에 대해 seed값을 고정시키는 이유가 뭔지
  
- 9/9 공부내용
  - data augmentation 실습
    - https://www.kaggle.com/yangsaewon/want-to-see-my-augmentations-pytorch
  - k fold 나누기 실습
    - https://www.kaggle.com/yangsaewon/how-to-split-folds-and-train
    - k fold 를 만들어내는 과정이 잘 이해가 안 됨. 마저 공부 필요
  - 정확도 0.9 baseline 실습
    - 계속 training을 하는데 GPU 에러가 남 (`device-side assert triggered`에러)
      - training 데이터의 클래스에서 문제가 생긴 것 같음
      - 1 epoch 조차도 못 돌리는 중
  - 모르는 내용 참조
    - https://www.kaggle.com/tmheo74/3rd-ml-month-12th-solution
    - 현재 내가 공부하고 있는 baseline 을 기반으로 추가한 코드. 내일 공부 필요