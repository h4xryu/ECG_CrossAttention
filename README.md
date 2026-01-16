# ECG_CrossAttention

ECG 부정맥 분류를 위한 Cross-Attention 기반 딥러닝 모델

## 프로젝트 구조

```
ECG_CrossAttention/
├── main.py              # 학습 스크립트
├── model.py             # Cross-Attention 모델 정의
├── train.py             # 학습 루프
├── test.py              # 평가 및 메트릭 계산
├── dataloader.py        # 데이터 로더
├── utils.py             # 유틸리티 함수
├── config.py            # 설정 파일
└── logger.py            # 학습 로깅
```

## 사용법

### 1. 데이터 준비
MIT-BIH Arrhythmia Database를 `data/` 폴더에 배치

### 2. 설정
`config.py`에서 실험 설정 수정

### 3. 학습 실행
```bash
python main.py
```

## 모델 출력

학습 완료 후 결과:
- `model_weights/`: 매 에폭 가중치
- `best_weights/`: Best 모델 가중치 (AUPRC, AUROC, Recall, Last)
- `results_*.xlsx`: 평가 결과 (Macro, Weighted, Per-Class, Confusion Matrix)

## 분류 클래스
- N: Normal
- S: SVEB (Supraventricular Ectopic Beat)
- V: VEB (Ventricular Ectopic Beat)
- F: Fusion

## 요구사항
- Python 3.8+
- PyTorch
- NumPy, Pandas
- scikit-learn
- openpyxl
- wfdb

