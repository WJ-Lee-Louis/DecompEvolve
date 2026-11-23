# DecompEvolve

Failure-Guided Subproblem Synthesis for Self-Evolving Prompt Optimization Agents

OptiMind IndustryOR의 API Baseline과 GEPA 비교 실험을 실행할 수 있습니다. DecompEvolve 방법은 설계 중입니다.

## 폴더 구성

```text
DecompEvolve/
├── experiments/
│   ├── decompevolve/   # 제안 방법의 구현 위치
│   ├── gepa/           # GEPA 프롬프트 최적화와 Test 평가
│   └── api_baseline/   # API baseline 실행·채점 코드, 설정, 결과 문서
├── src/benchmark_support/  # 공통 데이터 로딩과 환경 설정
├── data/
│   ├── icml/           # ICML 372문항, 그룹 설정, 고정 train/val/test 분할
│   ├── industryor/     # 기존 IndustryOR CSV 85문항
│   └── industryor_optimind/  # OptiMind 교정판 99문항, 정수 정답 CSV 85문항
├── reports/
├── scripts/            # 데이터 분할 및 Modal 실행
├── runs/               # 로컬 원본과 재개용 체크포인트
├── results/            # Git으로 공유하는 완료 결과
└── tests/              # baseline 실행·채점, 데이터 분할, 환경 설정 검증
```

방법 간 성능을 비교할 때는 같은 데이터셋의 평가 문항과 채점 규칙을 사용합니다. 프롬프트 개선에는 Train, 후보 선택에는 Validation, 최종 비교에는 Test를 사용합니다.

OptiMind IndustryOR는 저자가 평가용으로 공개한 벤치마크입니다. 정수 정답 CSV의 `index`는 헤더를 제외한 OptiMind 원본 CSV의 1부터 시작하는 레코드 순번입니다.

## 설치

Baseline만 실행하려면 Python 3.9 이상을 사용할 수 있습니다. 팀 실험 환경은 Python 3.12를 권장합니다. GEPA와 Modal은 Python 3.10 이상이 필요합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[gepa,modal]'
```

ICML 평가는 `data/icml/splits/`에 저장된 고정 분할을 사용합니다. 분할 재생성은 `python scripts/prepare_data.py`이며,
기존 파일이 있으면 중단합니다. 의도적으로 변경할 때만 `--force`를 사용하세요.
문제 앞부분·어휘·수치 유사도와 직접 검토한 연결을 이용해 313개 그룹으로 묶었습니다.
이는 의미적 중복 제거의 완전성을 보장하지 않으며 LP/IP별 층화 분할도 아닙니다.
현재 탐지·검토된 같은 그룹은 split을 넘지 않습니다. 데이터셋 변경이나 split 중복은 평가 시작 전에 검사합니다.

## 실험 실행

팀원용 [실험 환경 세팅 가이드](reports/WJ/261121_실험환경세팅가이드.md)에서 API 키와 Modal 계정을 연결하고 실행을 요청할 수 있습니다. 새 비교 실험은 OptiMind IndustryOR의 고정 Train/Validation/Test 35/20/30문항을 사용합니다. 실행 옵션과 결과 형식은 [Baseline](experiments/api_baseline/README.md)과 [GEPA](experiments/gepa/README.md)에 있습니다.

저장소 최상위에서 OptiMind Test 구성을 API 호출 없이 확인할 수 있습니다.

```bash
python -m experiments.api_baseline.evaluate --config experiments/api_baseline/configs/optimind_openai.json --dry-run
```

이전 ICML 실험은 기존 모델별 설정으로 재현할 수 있습니다. [ICML Baseline 결과 보고서](experiments/api_baseline/docs/261109_baseline_results.md)

OptiMind 완료 결과와 내보내기 방법은 [공유용 실험 결과](results/README.md)에서 확인할 수 있습니다.

## 로컬 검증

```bash
python -m unittest discover -s tests -v
```
