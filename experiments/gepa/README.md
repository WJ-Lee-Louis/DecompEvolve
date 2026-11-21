# GEPA 비교 실험

GEPA 0.1.4로 OptiMind IndustryOR의 풀이 프롬프트를 최적화합니다. 실행 전에 프로젝트 최상위에서 `python -m pip install -e '.[gepa,modal]'`로 의존성을 설치하세요.

- Train 35문항에서 매 반복마다 Easy·Medium·Hard 각 1문항을 뽑습니다. 난이도별 문항을 한 바퀴 돈 뒤 다시 섞습니다.
- 부모와 수정 후보를 같은 3문항으로 채점합니다. 부모가 3문항 모두 맞히면 반성 호출을 건너뜁니다. 점수가 오른 후보는 Validation 20문항에서 평가하고, 문항별 Pareto 선택과 Validation 평균 정확도를 사용합니다.
- 반성 모델은 문제, 공개 풀이 출력 전체, 정오답, 파싱된 정답, 참조 정답을 받습니다. 시간·토큰 한도에 도달한 문항도 0점이고, 반성 모델에는 일반 오답으로 전달합니다. 실행 기록에는 실제 상태를 남깁니다. 이 피드백은 Train에서만 생성합니다. 최종 평가에는 확정된 프롬프트와 Test 30문항을 사용합니다.

```bash
python -m experiments.gepa.optimize --config experiments/api_baseline/configs/optimind_openai.json --dry-run
modal run --detach scripts/modal_experiments.py::run --method gepa --provider openai --model gpt-5.4-mini-2026-03-17 --max-metric-calls 600 --parallelism 4 --seed 42 --run-id gepa-openai-01
```

마지막 명령은 모델을 실제 호출합니다. Train과 Validation 풀이의 기본 상한은 `--max-metric-calls 600`입니다. 최종 Test 30회를 포함하면 풀이 호출은 최대 630회입니다. 최소 예산은 초기 Validation 20회와 전체 반복 26회를 합한 46회이며, 남은 예산으로 전체 반복을 마칠 수 없으면 최적화를 종료합니다. 풀이 호출은 120초·생성 토큰 16,384개가 상한입니다. 별도 집계하는 반성 호출은 60초·8,192개가 상한이며, 이 한도에서 중단되면 해당 후보 생성을 건너뜁니다. 반성 모델은 기본적으로 풀이 모델과 같으며, `--reflection-provider`와 `--reflection-model`로 바꿀 수 있습니다.

Modal CPU 한 개에서 같은 배치의 API 요청을 기본 최대 4개씩 동시에 보냅니다. Train의 3문항은 함께 보내고, Validation은 4문항씩 나눠 보냅니다. GEPA의 반복과 후보 선택은 순서대로 진행됩니다. 팀원들은 서로 다른 모델이나 시드를 맡고, 각 실행의 결과를 모아 비교할 수 있습니다.

Modal 실행은 최대 24시간입니다. 남은 시간으로 안전하게 다음 반복을 마칠 수 없으면 상태를 저장하고 종료합니다. 원래 명령에 `--resume`을 붙이면 같은 실행 ID의 체크포인트에서 이어집니다. 최종 Test가 이미 시작됐다면 기록된 문항은 건너뜁니다. 강제 종료 시 체크포인트 이후의 미완료 반복은 다시 실행될 수 있으며, 이미 기록된 호출도 전체 예산에 포함됩니다. [Modal 실행 시간 제한](https://modal.com/docs/guide/timeouts)

원격 결과는 `decompevolve-results` 볼륨의 `/gepa/<run-id>`에 저장됩니다. `best_prompt.txt`는 확정 프롬프트, `optimization.json`은 후보와 Validation 결과, `rollouts.jsonl`과 `reflections.jsonl`은 실행 기록, `test/summary.json`은 전체 및 난이도별 Test 성능입니다. 해석이 필요한 문항도 고정 분할에 포함되어 있으며, 정확도는 데이터셋 정답과의 일치율입니다.
