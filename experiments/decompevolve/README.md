# DecompEvolve 설계와 실행

목표는 LP/IP 풀이에서 실제로 관찰한 오류를 다음 문제에서도 쓸 수 있는 점검 지침으로 바꾸는 것이다. 예를 들어 수요처별 최소 공급량을 전체 합계 하나로 처리했다면 “각 수요처의 하한을 별도 제약으로 쓰고, 합산하면서 사라진 조건이 있는지 확인한다”가 지침 후보다. 특정 수요량이나 정답은 지침에 넣지 않는다. GEPA의 문항별 Pareto 부모 선택과 후보 계보, LLMOPT의 문제 정식화 관점에서 출발했다. 아래 진단 분류와 합성 정책은 이 프로젝트의 설계다. [GEPA](https://arxiv.org/abs/2507.19457) · [LLMOPT](https://arxiv.org/abs/2410.13213)

주 비교의 ④는 [진단 기반 nano 출제·solver 검증 설계안](../../reports/WJ/261127_DecompEvolve_진단기반_출제_재설계.md)에 따른다.

## 한 번의 갱신

1. 같은 기본 프롬프트에서 시작해 기존 GEPA가 Validation 문항별 성과를 기준으로 부모를 선택한다. 원본 Train은 Easy, Medium, Hard에서 하나씩 순환 추출한다. 적합한 연습문제가 있으면 매번 교대로 한 자리를 교체한다.
2. 부모가 3개를 모두 맞히면 기존처럼 갱신을 건너뛴다. 오답이 있으면 공개 응답, 문제 본문, 파싱값과 참조값으로 진단한다. 정답을 맞힌 문항도 기존 GEPA 피드백에 남아 수정 시 보존할 전략을 보여 준다.
3. 반성 모델은 진단과 기존 응답을 읽고 재사용할 풀이 지침을 제안한다. 부모와 자식은 합성 문항까지 동일한 3문항으로 비교한다. 점수가 엄격히 오른 자식만 원본 Validation 20문항 전체에서 평가한다. GEPA가 후보 계보와 문항별 Pareto 성과를 관리하며 교차 병합은 기존 설정대로 꺼 둔다.
4. 원본 Train 34문항에서 근거가 모인 취약점은 다음 배치의 문제 합성에 사용한다. 합성 문항 오답도 프롬프트 개선에 쓰되 재귀적 합성의 종자로 쓰지 않는다. 최종 후보를 확정한 뒤 원본 Test 30문항으로 평가한다.

## 진단 근거와 판단 보류

진단 순서는 수량·단위 → 의사결정변수와 인덱스 → 변수 범위·정수성 → 목적식 → 제약식 → 실행 가능성과 최적성 → 산술이다. 모델은 공개 풀이에서 드러난 가장 이른 오류 하나를 고르고, **풀이 인용·대응하는 문제 조건 인용·불일치 설명·다른 문제에도 적용할 점검 지침**을 반환한다. 두 인용이 실제 입력에 포함되는지 코드로 검사한다. 인용 검사는 의미적 판정의 정확성을 보장하지 않으므로 후속 실험에서 진단 표본을 사람이 재검토해야 한다.

틀린 최종 숫자만 있거나 근거가 부족하면 `unknown`, 형식 오류는 `format`, 타임아웃 등은 `execution`으로 기록한다. 이 경우 수학 오류를 추정해 합성하지 않는다. 풀이·진단·프롬프트 수정은 모두 도구 없이 `gpt-5.4-nano`를 사용한다.

## 합성의 정확성과 난이도

출제 nano는 진단 근거와 원본 Train 맥락을 읽고 유계 2변수 LP/IP를 제안한다. 자유 서술 문제와 정답을 모델에 따로 쓰게 하면 수학 모형과 문장이 어긋날 수 있어, 계수·단위·도메인에서 문장과 수식을 함께 만든다. 실제 문제와 연결되는 단위는 아래 다섯 가지 구체적 실수다.

| 관찰된 실수 | 합성 문항의 채택 검사 |
|---|---|
| 시간과 분의 환산 누락 | 시간을 분으로 바꾸지 않으면 최적값이 달라짐 |
| 정수 변수의 연속 완화 | 완화 최적값과 정수 최적값이 다름 |
| 순이익 대신 매출 최대화 | 매출로 채점한 값과 순이익 정답이 다름 |
| 두 변수의 비례 제약 누락 | 해당 제약을 제거하면 최적값이 달라짐 |
| 한 제품만 생산하는 해를 최적이라고 판단 | 혼합 생산이 두 단일 제품 계획보다 엄격히 우수함 |

합성기 안에서만 연속 문제의 꼭짓점을 유리수로 열거하거나, 유계 정수 문제의 격자점을 완전 탐색한다. 실행 가능한 최적해와 정확한 정수 목적값을 확보하고 표의 검사까지 통과한 문항만 사용한다. 목표값이 소수이면 반올림하지 않고 버린다. 최초 제안과 수정 1회까지 검증한 뒤 채택할 문항이 없으면 원본 배치를 사용한다. 출제 호출과 토큰은 별도 집계한다.

원본 문항·실수 조합마다 중간 단계에서 시작해 부모가 맞히면 제약을 더한 상위 단계로, 틀리면 단순한 하위 단계로 이동한다. 단계는 모형의 구조에 붙인 난이도 가설이다. 최상단 정답·최하단 오답에서 중단하며 한 조합당 최대 3문항까지 사용한다. 아직 덜 연습한 조합, 반복 진단된 조합 순으로 우선한다. 합성 자리는 최대 1/3이고, 목표 실수가 확인되기 전에는 세 자리 모두 원본이다. 다른 종류의 제약·다기간 인덱싱 등은 진단과 지침 개선까지만 지원한다.

## 실행과 후속 비교

저장소 루트에서 Python 3.12 환경에 `python -m pip install -e '.[gepa]'`를 실행한다. 아래 명령은 모두 API 호출 없이 설정을 확인한다.

```bash
python -m experiments.api_baseline.evaluate --dry-run
python -m experiments.gepa.optimize --dry-run
python -m experiments.decompevolve.optimize --dry-run
python -m experiments.decompevolve.optimize --diagnosis-only --dry-run
```

실제 최적화에는 `--dry-run` 대신 `--run-dir runs/<새 실행 이름>`을 지정한다. 재개 시 같은 인자에 `--resume`을 추가한다. Modal에서도 `--method decompevolve --provider openai`를 사용할 수 있다. 세 방법의 기본 구성은 동일한 nano 모델, 기본 프롬프트, 답안 형식, 데이터 분할을 사용한다.

## 1200회 비교 설계

주 질문은 **원본 오답의 진단 근거를 읽은 nano가 새 LP를 설계하고 solver가 검증할 때, 진단만 사용한 경우보다 원본 Test 성능이 개선되는가**이다. 아래 네 방법을 같은 `gpt-5.4-nano`, 기본 지침, 도구 없는 풀이, 고정 답안 형식 및 원본 Test 30문항으로 비교한다.

| 주 비교 | Train / Validation | 무엇을 측정하는가 |
|---|---|---|
| ① Base prompt | 최적화 없이 Test만 평가 | 모델과 초기 지침의 기준 성능 |
| ② GEPA | 원본 Train, 일반 반성 | ②−① 지침 최적화 효과 |
| ③ GEPA + 진단 | ②에 근거 기반 오류 진단 추가 | ③−② 진단 효과 |
| ④ DecompEvolve (nano 출제) | ③의 동일 진단 + nano 출제 + solver 검증, 이후 배치에 최대 한 자리 교체 | ④−③ 진단 기반 합성·노출의 추가 효과 |

**출제 절차.** 원본 Train의 인용·조건·불일치 이유·점검 지침·skill을 출제 의뢰로 전달한다. nano가 유계 2변수 LP/IP 모형을 제안하면 프로그램이 단위·도메인·제약을 검사하고 등록된 실수 연산 하나를 적용한다. 정상 모형과 오류 모형의 정확한 최적값이 다르고 정상 답이 정수일 때 같은 모형에서 문장을 생성한다. 슬롯당 제안 1회와 수정 최대 1회 후에도 기각되면 원본을 사용한다. 풀이 모델에는 문제와 지침만 전달한다. [출제 규격과 검증 사례](../../reports/WJ/261127_DecompEvolve_진단기반_출제_재설계.md)

**예산과 통제.** ②~④ 각각 Train 34·Validation 20, 시드 42, 병렬도 4, 동일한 후보 선택·수락 규칙을 쓴다. 각 방법의 시드 검증·부모/자식 풀이(합성 포함)·수락 후보 검증을 합쳐 최대 1,200회다. 다음 반복에 26회를 남길 수 있을 때만 시작한다. 시드 하나 기준 주 비교의 상한은 최적화 풀이 3,600회와 최종 Test 120회다. 모든 방법의 지침은 Test 전에 확정한다.

**실행 순서.** 새 ④의 첫 단계는 Train만 이용한 출제 파일럿이다. 정상/오류 최적값, 단위와 문장 대응, 기각 사유와 실제 출제 토큰을 확인한다. Train 13·88번의 모의 오류를 사용한 최종 파일럿에서 3회 호출로 7,467토큰을 사용했다. 두 의뢰 중 하나는 정상값 13과 오류값 25를 확인해 채택했고, 다른 하나는 수정 후에도 값이 같아 기각했다. 이 규모에서 약 120회의 평균 출제 요청에 해당하는 누적 상한 300,000토큰을 본 실험 전에 고정하고, 소진 시 남은 반복에는 원본 배치를 쓴다. 본 실험에서는 풀이·진단·변이·출제(수정 포함)별 호출과 입출력 토큰을 분리한다. Solver 계산은 LLM 토큰 0이며 연산 시간은 별도다.

**결과표.** 주 지표는 원본 Test 정답 수/30과 정확도다. 같은 문항에서 두 방법의 정오답이 엇갈린 수도 제시한다. 과정 지표는 원본/합성 혼합 배치별 부모 3/3 비율, 출제 시도·채택·기각·실제 노출 수, 합성에서만 점수가 올라 수락된 자식의 부모 대비 원본 Validation 점수 변화다. 동일 풀이 예산과 실제 총비용을 구분해 보고하며, 최종 원본 Test에서 조건별 성능을 비교한다.

### Modal 실행

Python 3.12 환경에서 저장소를 기준으로 `python -m pip install -e '.[gepa,modal]'`을 실행한다. `modal profile list`가 비어 있으면 `modal token new`로 브라우저에서 로그인하고 대상 작업공간을 선택한다. `modal token info`로 서버가 반환한 사용자와 작업공간을 확인한다. 여러 프로필이 있으면 아래 명령의 `--profile`에 해당 로컬 프로필명을 지정한다. 대상 환경의 `decompevolve-api-keys` Secret에 `OPENAI_API_KEY`가 필요하다.

```bash
modal run --profile woojin716 scripts/modal_experiments.py::smoke
```

smoke 출력에서 원격 실행 여부, 34/20/30 분할, OpenAI 키 존재 여부를 확인한다. 키 유효성이나 모델 사용 권한은 이 검사만으로 확인할 수 없다. 아래 명령은 실제 모델 호출을 발생시킨다. 이미 제출된 실행과 별도 실험을 구분한다. 설정만 확인하려면 각 명령 끝에 `--dry-run`을 추가한다. 원격 dry-run도 Modal 컨테이너는 사용한다.

```bash
modal run --profile woojin716 --detach scripts/modal_experiments.py::run --method baseline --split test --parallelism 4 --run-id nano-base-test30-s42
modal run --profile woojin716 --detach scripts/modal_experiments.py::run --method gepa --seed 42 --run-id nano-gepa-1200-s42
modal run --profile woojin716 --detach scripts/modal_experiments.py::run --method decompevolve --diagnosis-only --seed 42 --run-id nano-diag-1200-s42
modal run --profile woojin716 --detach scripts/modal_experiments.py::run --method decompevolve --seed 42 --run-id nano-authored-1200-s42
```

원격 함수는 1 CPU·512 MiB로 설정돼 있다. 각 실행은 별도 ID를 사용하고, 일시정지한 실행은 같은 옵션과 ID에 `--resume`을 붙여 이어간다. GEPA의 결과는 볼륨의 `/gepa/<run-id>`, 두 DecompEvolve 조건은 `/decompevolve/<run-id>`에 저장된다. 두 조건의 manifest에서 `decompevolve.synthesis`가 각각 false/true인지 확인한다.

### 실행 기록

다운로드한 각 실행에 `python scripts/summarize_budget.py <run-dir>`를 적용해 풀이·진단·지침 수정·출제·시험별 호출 수와 보고된 토큰을 집계한다. `missing_usage`가 양수이면 토큰 합계는 확인된 부분만 나타낸다.

- `train_batches.jsonl`에서 부모의 3/3 비율과 원본 전용·합성 혼합 배치별 비율을 계산한다. 공동작업자의 약 2/3 관찰은 nano 결과로 재확인한다.
- `diagnoses.jsonl`에서 판단 보류 비율, 오류 단계와 점검 지침을 확인한다. 표본에서 인용이 실제 오류를 설명하는지, 최종 후보 지침에 반영됐는지 검토한다.
- `synthetic.jsonl`은 원본 문항 ID, 목표 실수, 구조 단계, 문장, 수식, 최적해와 오개념 적용값을 보관한다. 음수 ID로 부모 배치와 연결해 단계별 정답률을 측정한다.
- `reflections.jsonl`의 purpose와 usage로 진단·수정 비용을 분리한다. `optimization.json`과 `best_prompt.txt`에서 후보 계보·Validation 성과·최종 지침을 확인하고 `test/summary.json`에서 원본 Test 정확도를 비교한다.

### 배포된 실험 확인

`woojin716`의 `main` 환경에 배포한 `decompevolve-nano-1200` 앱은 기본 지침의 Test를 수행한다. 전체 방법은 `decompevolve-nano-author-1200` 앱에서 `nano-authored-1200-s42` 실행 ID를 사용한다. 배포된 함수에 `spawn`으로 제출했으므로 로컬 터미널 연결이 필요하지 않다. 이번 제출은 `auto_continue=True`이며, 세션의 안전 종료 시점에 도달하면 체크포인트를 커밋하고 같은 ID로 다음 호출을 제출한다. API 오류로 종료된 실행은 로그에서 원인을 확인해야 한다.

[Base Test 로그](https://modal.com/apps/woojin716/main/deployed/decompevolve-nano-1200)와 [전체 방법 로그](https://modal.com/apps/woojin716/main/deployed/decompevolve-nano-author-1200)에서 확인하거나 아래 명령으로 로그와 결과 목록을 본다. 로컬 설정은 `source ~/.local/share/decompevolve-modal/venv/bin/activate`로 불러온다.

```bash
modal app logs --profile woojin716 decompevolve-nano-author-1200 --tail 50
modal volume ls --profile woojin716 decompevolve-results /decompevolve/nano-authored-1200-s42
```

기본 지침의 `summary.json`, 최적화 실행의 `test/summary.json`에서 `completed`가 true인지 확인한다. CPU와 메모리는 Modal에, 모델 사용량은 OpenAI API 계정에 집계된다. 이번 배포는 기존 설정인 `nonpreemptible=True`를 사용하며 배포 시 Modal이 표시한 해당 CPU·메모리 요율은 일반 요율의 3배다.
