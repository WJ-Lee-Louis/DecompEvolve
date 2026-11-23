# API Baseline

고정 프롬프트를 사용해 문항당 도구 없이 LLM API를 한 번 호출하는 독립 비교 실험입니다. 설치 방법은 [프로젝트 README](../../README.md#설치)를 참고하세요. 아래 명령은 레포지토리 최상위에서 실행합니다.

- [평가 과정과 설정](docs/261109_baseline_setting.md)
- [기존 모델 4종의 Test 결과](docs/261109_baseline_results.md)

## OptiMind IndustryOR 실행

새 비교 실험의 기본 평가는 Test 30문항입니다. `optimind_openai.json`은 OpenAI Responses API와 `OPENAI_API_KEY`를, `optimind_exaone.json`은 LG EXAONE Chat Completions API와 `EXAONE_API_KEY`를 사용합니다. 두 설정 모두 문항당 한 번 호출하고 동일한 정수 채점 규칙을 적용합니다. [EXAONE API 사양](https://docs.exaone.ai/en/docs/k-exaone-2.5)

```bash
python -m experiments.api_baseline.evaluate --config experiments/api_baseline/configs/optimind_openai.json --dry-run
python -m experiments.api_baseline.evaluate --config experiments/api_baseline/configs/optimind_exaone.json --dry-run
modal run --detach scripts/modal_experiments.py::run --method baseline --provider exaone --model k-exaone_v2.5 --run-id baseline-exaone-01
```

마지막 명령은 모델을 실제 호출합니다. 먼저 `--dry-run`으로 설정을 확인하거나 `--split train --limit 3`을 지정해 소규모 실행을 할 수 있습니다. 제공자에 맞는 모델 ID를 `--model`로 지정하세요. OptiMind 풀이 호출의 상한은 문항당 15분(900초)과 생성 토큰 65,536개입니다. 생성 토큰에는 추론 토큰도 포함됩니다. 시간이나 토큰 한도에 도달하면 0점을 기록하고 다음 문항으로 넘어갑니다. 원격 결과는 `decompevolve-results` 볼륨의 `/baseline/<run-id>`에 저장됩니다. `summary.json`에는 전체 및 난이도별 정확도가 들어 있습니다.

OptiMind의 [기본 프롬프트](prompts/optimind_baseline.txt)는 GEPA 초기 후보를 평가할 때도 동일하게 사용합니다. `--parallelism 5`를 지정하면 로컬 또는 Modal Baseline에서 독립적인 문항 요청을 최대 5개씩 동시에 보냅니다. 중단된 병렬 실행을 재개할 때는 같은 값을 사용하세요.

아래 절은 기존 ICML Baseline 설정과 실행 기록을 설명합니다.

## 비용 없는 사전 확인

API 키 없이도 실행되며 API 호출과 결과 파일 생성이 없습니다.

```bash
python -m experiments.api_baseline.evaluate --config experiments/api_baseline/configs/baseline_luna.json --dry-run
python -m experiments.api_baseline.evaluate --config experiments/api_baseline/configs/baseline_mini.json --dry-run
```

## Baseline 실행

프로젝트 최상위 `.env` 파일에 `OPENAI_API_KEY="실제 API 키"`를 저장하세요.
평가 스크립트가 실행 위치와 관계없이 이 파일을 자동으로 불러오므로 `source .env`는 필요 없습니다.
이미 설정된 셸 환경변수가 있으면 해당 값이 우선합니다. 새 터미널에서도 `.env`를 자동으로 읽습니다.
키는 코드·설정·Git에 넣지 마세요. 실제 평가는 아래 명령에서 비용이 발생합니다.
모델 접근 권한은 계정에서 확인해야 하며, 사용 불가능한 모델을 자동 대체하지 않습니다.

```bash
# 먼저 Train 3문항으로 API·출력 형식 확인
python -m experiments.api_baseline.evaluate --config experiments/api_baseline/configs/baseline_luna.json --split train --limit 3

# 같은 Test 100문항에서 두 모델 평가
python -m experiments.api_baseline.evaluate --config experiments/api_baseline/configs/baseline_luna.json --split test
python -m experiments.api_baseline.evaluate --config experiments/api_baseline/configs/baseline_mini.json --split test
```

Luna는 `gpt-6-luna`, Mini는 `gpt-5.4-mini-2026-03-17`을 사용합니다.
두 설정 모두 `reasoning.effort=medium`, `max_output_tokens=8192`입니다.
추가 비교 설정은 `experiments/api_baseline/configs/baseline_luna_5_6.json` (`gpt-5.6-luna`)과
`experiments/api_baseline/configs/baseline_mini_5.json` (`gpt-5-mini-2025-08-07`)입니다. 추론 강도·출력 예산·프롬프트·Test 100문항은 동일합니다.
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

중단된 실행은 출력된 실행 폴더를 지정해 재개할 수 있습니다. 폴더 정리 전에 만든 실행도 같은 설정과 문항으로 재개할 수 있습니다.
이미 기록된 문항은 API 오류를 포함해 모두 건너뜁니다. API 오류 문항도 자동 재시도하지 않습니다.
부분적으로 잘린 JSONL 레코드가 있으면 자동 복구 대신 중단하므로 먼저 기록을 확인하세요.

```bash
python -m experiments.api_baseline.evaluate --config experiments/api_baseline/configs/baseline_luna.json --split test \
  --run-dir runs/baseline/gpt-6-luna/실행폴더 --resume
```

공식 문서: [Responses/추론](https://developers.openai.com/api/docs/guides/reasoning),
[Luna](https://developers.openai.com/api/docs/models/gpt-6-luna),
[Mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini).
