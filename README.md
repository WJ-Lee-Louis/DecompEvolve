# DecompEvolve

Failure-Guided Subproblem Synthesis for Self-Evolving Prompt Optimization Agents

DecompEvolve 방법은 구현 전입니다. 현재 실행 가능한 API baseline은 독립 비교 실험입니다.

## 폴더 구성

```text
DecompEvolve/
├── experiments/
│   ├── decompevolve/   # 제안 방법의 구현 위치
│   ├── gepa/           # GEPA 비교 실험의 구현 위치
│   └── api_baseline/   # API baseline 실행·채점 코드, 설정, 결과 문서
├── src/benchmark_support/  # 공통 데이터 로딩과 환경 설정
├── data/               # 원본 372문항과 고정 train/val/test 분할
├── reports/
├── scripts/            # 공통 데이터 준비 스크립트
├── runs/               # 로컬 실행 결과, Git 제외
└── tests/              # baseline 실행·채점, 데이터 분할, 환경 설정 검증
```

방법 간 성능을 비교할 때는 같은 데이터셋의 평가 문항과 채점 규칙을 사용합니다. 프롬프트 개선에는 Train, 후보 선택에는 Validation, 최종 비교에는 Test를 사용합니다.

## 설치

Python 3.9 이상에서 실행합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

이미 생성된 split을 사용합니다. 분할 재생성은 `python scripts/prepare_data.py`이며,
기존 파일이 있으면 중단합니다. 의도적으로 변경할 때만 `--force`를 사용하세요.
문제 앞부분·어휘·수치 유사도와 직접 검토한 연결을 이용해 313개 그룹으로 묶었습니다.
이는 의미적 중복 제거의 완전성을 보장하지 않으며 LP/IP별 층화 분할도 아닙니다.
현재 탐지·검토된 같은 그룹은 split을 넘지 않습니다. 데이터셋 변경이나 split 중복은 평가 시작 전에 검사합니다.

## Baseline 실행과 결과

설치 후 레포지토리 최상위에서 아래 명령으로 API 호출과 비용 없이 Test 구성을 확인할 수 있습니다.

```bash
python -m experiments.api_baseline.evaluate --config experiments/api_baseline/configs/baseline_luna.json --dry-run
```

API 키 설정, 4개 모델의 설정 파일과 실행·재개 방법은 [Baseline 실행 안내](experiments/api_baseline/README.md)에 있습니다. 기존 모델 4종의 Test 결과는 [Baseline 결과 보고서](experiments/api_baseline/docs/261109_baseline_results.md)에서 볼 수 있습니다. 원본 실행 기록은 각 실행 환경의 `runs/baseline/`에 있습니다.

## 로컬 검증

```bash
python -m unittest discover -s tests -v
```
