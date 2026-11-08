# DecompEvolve

Failure-Guided Subproblem Synthesis for Self-Evolving Prompt Optimization Agents

현재 단계는 외부 도구 없이 단일 LLM 호출로 최적화 문제를 푸는 baseline입니다.
GEPA와 Ours는 추후 같은 에이전트·평가기 위에 연결합니다.

## 구성

- `data/datasets_ICML_int.csv`: 372문항, `index`, `question`, `answer`. 기존 정답과 질문을 그대로 유지했습니다.
- `data/splits/`: Train 222 / Validation 50 / Test 100의 고정 ID 목록, seed 42, 데이터 SHA-256, 그룹 목록.
- `data/group_overrides.json`: 직접 대조한 표·서술형 등 문제 변형의 그룹 연결.
- `configs/`: Luna와 Mini의 모델·추론·출력 예산 설정.
- `prompts/baseline.txt`: 초기 풀이 프롬프트. 출력·채점 규약은 공통 평가기에 고정되어 있습니다.
- `src/decompevolve/`: 공통 데이터 로더·평가기.
- `runs/baseline/`: 실행별 기록. Git에서 제외됩니다.

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

## 비용 없는 사전 확인

API 키 없이도 실행되며 API 호출과 결과 파일 생성이 없습니다.

```bash
python scripts/evaluate.py --config configs/baseline_luna.json --dry-run
python scripts/evaluate.py --config configs/baseline_mini.json --dry-run
```

## Baseline 실행

프로젝트 최상위 `.env` 파일에 `OPENAI_API_KEY="실제 API 키"`를 저장하세요.
평가 스크립트가 실행 위치와 관계없이 이 파일을 자동으로 불러오므로 `source .env`는 필요 없습니다.
이미 설정된 셸 환경변수가 있으면 해당 값이 우선합니다. 새 터미널에서도 `.env`를 자동으로 읽습니다.
키는 코드·설정·Git에 넣지 마세요. 실제 평가는 아래 명령에서 비용이 발생합니다.
모델 접근 권한은 계정에서 확인해야 하며, 사용 불가능한 모델을 자동 대체하지 않습니다.

```bash
# 먼저 Train 3문항으로 API·출력 형식 확인
python scripts/evaluate.py --config configs/baseline_luna.json --split train --limit 3

# 같은 Test 100문항에서 두 모델 평가
python scripts/evaluate.py --config configs/baseline_luna.json --split test
python scripts/evaluate.py --config configs/baseline_mini.json --split test
```

Luna는 `gpt-6-luna`, Mini는 `gpt-5.4-mini-2026-03-17`을 사용합니다.
두 설정 모두 `reasoning.effort=medium`, `max_output_tokens=8192`입니다.
추가 비교 설정은 `configs/baseline_luna_5_6.json` (`gpt-5.6-luna`)과
`configs/baseline_mini_5.json` (`gpt-5-mini-2025-08-07`)입니다. 추론 강도·출력 예산·프롬프트·Test 100문항은 동일합니다.
실행하려면 위 평가 명령의 `--config` 경로를 해당 파일로 변경하세요.
출력 예산에는 보이지 않는 추론 토큰도 포함되므로, `incomplete`가 많으면 Train에서 예산을 점검하세요.

문항당 Responses API 요청은 1회이며 `tools=[]`, `tool_choice=none`, `store=false`입니다.
SDK 자동 재시도와 정답 불일치 재시도는 없습니다. 문제별 대화는 독립적이며 정답 라벨은 모델에 전달하지 않습니다.
API 오류가 발생하면 해당 결과를 저장하고 실행을 중단합니다.

출력 마지막 비어 있지 않은 줄은 `Answer: -200`처럼 부호 있는 정수여야 합니다.
모델은 짧은 모델링 요약과 **최적 목적함수 값**을 출력합니다.
소수를 잘라내거나 반올림하지 않습니다. 정수 답 라벨이 있다고 변수까지 정수로 강제하지 않습니다.
기존 라벨과의 일치 여부를 채점하며, 수학적 정답을 별도로 보증하지 않습니다.

실행 폴더에는 다음 파일을 저장합니다.

- `manifest.json`: 모델 설정, 실제 지침, 선택 ID, 데이터·평가기 해시. 재개 시 동일 조건을 검사합니다.
- `predictions.jsonl`: 문제별 응답, 파싱 답, 정답 라벨, 상태, 응답 ID, 토큰 사용량, 지연시간, 원본 API 응답.
- `summary.json`: 정확도, 상태별 개수, 처리 진행률, 토큰 사용량. 추론 토큰은 출력 토큰에 이미 포함됩니다.

형식 오류·미완료·거절·API 오류는 0점이며 별도 집계됩니다.
처리가 중단되면 전체 정확도는 `null`, 기록된 문항 기준 정확도는 별도로 표시됩니다.
계정의 실제 청구 금액은 토큰 기록과 API 대시보드로 확인하세요.

중단된 실행은 출력된 실행 폴더를 지정해 재개할 수 있습니다.
이미 기록된 문항은 API 오류를 포함해 모두 건너뜁니다. API 오류 문항도 자동 재시도하지 않습니다.
부분적으로 잘린 JSONL 레코드가 있으면 자동 복구 대신 중단하므로 먼저 기록을 확인하세요.

```bash
python scripts/evaluate.py --config configs/baseline_luna.json --split test \
  --run-dir runs/baseline/gpt-6-luna/실행폴더 --resume
```

## 로컬 검증

```bash
python -m unittest discover -s tests -v
```

공식 문서: [Responses/추론](https://developers.openai.com/api/docs/guides/reasoning),
[Luna](https://developers.openai.com/api/docs/models/gpt-6-luna),
[Mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini).
