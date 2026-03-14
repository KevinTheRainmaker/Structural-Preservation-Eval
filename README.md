# Structural-Preservation-Eval

## 1. 구조 보존의 개념

번안에서의 구조 보존은 원문과 번안문 사이의 **표면적 문장 형태의 일치**를 의미하지 않는다.

번안은 쉬운 표현으로 바꾸고, 문장을 분할하거나 병합하고, 필요한 설명을 추가하는 작업을 포함하므로, 문장 수준의 형식 동일성을 기준으로 구조 보존을 평가하는 것은 적절하지 않다.

따라서 구조 보존은 다음과 같이 정의할 수 있다.

> **구조 보존이란 원문에서 정보가 조직되고 강조되며 전개되는 방식이 번안문에서도 얼마나 유지되는지를 평가하는 것이다.**
> 

즉 구조 보존은 **무엇을 말했는가**가 아니라, **그 내용을 어떤 순서와 방식으로 조직해 전달했는가**를 다룬다.

---

## 2. 구조 보존의 이론적 근거

구조 보존은 여러 이론적 전통과 연결된다.

### 2.1 담화 분석(Discourse Analysis)

텍스트는 문장들의 단순한 나열이 아니라, 화제 전개와 정보 배열, 응집성과 일관성을 가진 담화 단위이다. 따라서 구조 보존은 문장 간 관계와 정보 흐름의 유지 여부를 평가해야 한다.

### 2.2 Halliday의 체계기능언어학(Systemic Functional Linguistics)

Halliday의 체계기능언어학에서 구조 보존은 특히 **textual metafunction**과 관련된다. 이는 텍스트가 정보를 어떻게 조직하는지, 즉 **theme-rheme**, **information flow**, **cohesion**의 측면에서 의미를 형성한다는 관점을 제공한다.

### 2.3 Rhetorical Structure Theory (RST)

RST는 텍스트를 절과 문장 사이의 담화 관계로 설명한다. Cause, Elaboration, Contrast, Background와 같은 관계, 그리고 **Nucleus-Satellite** 구조는 텍스트의 큰 담화 뼈대를 보여준다. 따라서 RST는 구조 보존의 담화 수준 평가에 직접적 근거를 제공한다.

### 2.4 Dependency Structure

Dependency parsing은 문장 내부에서 단어와 구가 어떻게 연결되는지를 보여준다. 이는 문장 내부 정보 배열과 술어 중심 구조를 평가하는 데 유용하며, 구조 보존의 문장 수준 평가에 해당한다.

---

## 3. 구조 보존의 평가 대상

구조 보존은 다음 네 가지 요소를 중심으로 평가한다.

### 3.1 화제 전개

텍스트가 어떤 화제를 중심으로 어떤 순서로 전개되는가

- 중심 화제가 유지되는가
- 화제 전환의 흐름이 비슷한가

### 3.2 강조 구조

어떤 정보가 핵심으로 강조되고 어떤 정보가 보조로 배치되는가

- 핵심 메시지가 동일하게 부각되는가
- 중심 정보와 부연 정보의 위계가 유지되는가

### 3.3 정보 배열

정보가 어떤 논리적 순서로 제시되는가

- 원인–결과
- 배경–사건
- 설명–예시
- 문제–해결

같은 배열이 유지되는가

### 3.4 장면/세그먼트 전환

텍스트가 어떤 단위로 분절되고 어디에서 전환되는가

- 세그먼트 수가 크게 달라졌는가
- 장면 전환 위치가 유사한가

---

## 4. 구조 보존 평가의 기본 원칙

번안은 설명 추가와 단순화를 허용하므로, 구조 평가에서는 다음 원칙을 따른다.

### 허용되는 변화

- 문장 수 변화
- 설명 추가
- 쉬운 표현으로의 치환
- 문장 분할 및 병합

### 평가해야 하는 변화

- 중심 화제의 이동
- 핵심 강조점의 변경
- 정보 순서의 역전
- 장면/세그먼트 구조의 붕괴
- 주요 담화 관계의 소실

즉 구조 보존 평가는 **surface form preservation**이 아니라 **organizational meaning preservation**을 목표로 한다.

---

# 5. 자동 평가 프레임워크 설계

구조 보존 자동평가는 서로 다른 수준의 구조를 분리하여 평가하는 것이 가장 적절하다.

본 프레임워크는 다음 6개의 독립 지표로 구성된다.

- $S_{seg}$: 세그먼트 구조 보존
- $S_{topic}$: 화제 전개 보존
- $S_{order}$: 정보 배열 보존
- $S_{focus}$: 강조 구조 보존
- $S_{rst}$: 담화 관계 구조 보존
- $S_{dep}$: 문장 내부 구조 보존

각 점수는 **0~1 범위**에서 계산되며, 1에 가까울수록 구조 보존이 높음을 의미한다.

이 접근의 장점은 구조 보존을 단일 점수로 뭉뚱그리지 않고, **어떤 수준의 구조가 유지되었고 어떤 수준이 무너졌는지 진단 가능하다**는 점이다.

---

## 5.1 세그먼트 구조 보존 ($S_{seg}$): 분절 구조가 얼마나 비슷한가

### 목적

원문과 번안문이 비슷한 위치에서 비슷한 단위로 분절되는지를 평가

### 방법

1. 문장 임베딩 생성
2. 인접 문장 간 cosine similarity 계산
3. 유사도 급락 지점을 세그먼트 경계로 탐지
4. 원문과 번안문의 경계 위치와 세그먼트 수 비교

### 계산식

문장 임베딩 $e_i$ 에 대해 $sim_i = \cos(e_i, e_{i+1})$

경계 후보는 평균 대비 급락 지점으로 탐지

원문 경계 집합 $B_s$, 번안 경계 집합 $B_t$ 라 할 때, 허용 오차 $\delta$ 이내에서 일치하는 경계 수를 사용해 

$$
S_{boundary} = \frac{|match(B_s, B_t)|}{|B_s|}
$$

세그먼트 수 유사도는

$$
S_{count} = 1 - \frac{|N_s - N_t|}{\max(N_s, N_t)}
$$

최종 점수는 이 둘을 가중합 (예: $S_{seg} = 0.6S_{boundary} + 0.4S_{count}$)

---

## 5.2 화제 전개 보존 ($S_{topic}$): 화제 흐름이 얼마나 유지되는가

### 목적

세그먼트별 주제 흐름이 유지되는지를 평가

### 방법

1. 각 세그먼트의 대표 임베딩 생성
2. 원문과 번안문의 세그먼트 정렬
3. 대응 세그먼트 간 topic similarity 계산
4. topic transition 흐름 비교

### 계산식

세그먼트 $i$ 의 임베딩을 $E_i$ 라 할 때, 정렬 집합 $A = {(i,j)}$ 에 대해

$$
S_{topic}^{local} =
\frac{1}{|A|}
\sum_{(i,j)\in A}
\cos(E_i, E_j)
$$

또한 전이 벡터를 $T_i = E_{i+1} - E_i$ 라고 할 때, 흐름 유사도는

$$
S_{flow} =
\frac{1}{m}
\sum_{i=1}^{m}
\cos(T_i^s, T_i^t)
$$

최종 점수는 이 둘을 가중합 (예: $S_{topic} = 0.7S_{topic}^{local} + 0.3S_{flow}$)

---

## 5.3 정보 배열 보존 ($S_{order}$): 정보 순서가 얼마나 유지되는가

### 목적

정보의 논리적 순서가 유지되는지를 평가

### 방법

1. 문장 또는 세그먼트 정렬
2. 정렬된 대응쌍의 순서 비교
3. inversion이 많을수록 점수 하락

### 계산식

정렬된 대응 순서로 Kendall’s $\tau$ 를 계산한다.

$$
\tau = \frac{C - D}{\frac{1}{2}n(n-1)}
$$

- C: concordant pair 수
- D: discordant pair 수

이를 0~1로 정규화하면

$$
S_{order} = \frac{\tau + 1}{2}
$$

---

## 5.4 강조 구조 보존 ($S_{focus}$): 핵심 강조점이 얼마나 유지되는가

### 목적

원문이 핵심적으로 강조하는 정보가 번안문에서도 유지되는지를 평가

### 방법

1. 문장/세그먼트별 salience score 계산
2. 상위 k개 salient unit 추출
3. 의미 정렬 후 유사도 계산

### salience score

문장 $i$의 강조 점수는 다음과 같이 계산할 수 있다.

$$
sal_i = \alpha P_i + \beta R_i + \gamma C_i + \delta E_i
$$

- $P_i$: 위치 가중치
- $R_i$: 반복 키워드 비율
- $C_i$: 강조 cue phrase 존재 여부
- $E_i$: entity centrality

### 계산식

원문의 상위 salient unit 집합 $F_s$, 번안문의 집합 $F_t$에 대해

$$
S_{focus} =
\frac{1}{k}
\sum_{i=1}^{k}
\max_j \cos(f_i, g_j)
$$

여기서 $f_i \in F_s, g_j \in F_t$ 이다.

---

## 5.5 담화 관계 구조 보존 ($S_{rst}$): 담화 관계 뼈대가 얼마나 유지되는가

### 목적

문장·절 사이의 담화 관계가 유지되는지를 평가

### 방법

1. 원문과 번안문에 대해 RST parsing 수행
2. relation label, span, nucleus/satellite 구조 추출
3. relation matching 수행
4. relation 보존도와 트리 형상 유사도 계산

### (1) Relation Matching Score

원문 관계 집합 $R_s$, 번안 관계 집합 $R_t$, 매칭 집합 $M$에 대해

$$
P = \frac{|M|}{|R_t|}
$$

$$
R = \frac{|M|}{|R_s|}
$$

$$
S_{rel} = \frac{2PR}{P + R}
$$

### (2) Tree Shape Similarity

RST 트리의 레벨 분포를 비교하여 Wasserstein distance $W$를 구한 후 계산

$$
S_{shape} = 1 - \frac{W(D_s, D_t)}{\max(depth)}
$$

최종 점수는 이 둘의 가중합 (예: $S_{rst} = 0.7S_{rel} + 0.3S_{shape}$)

---

## 5.6 문장 내부 구조 보존 ($S_{dep}$): 문장 내부 구조가 얼마나 유지되는가

### 목적

문장 내부의 dependency 구조가 유지되는지를 평가

### 방법

1. 원문/번안문 문장 dependency parse
2. 대응 문장 쌍 정렬
3. 핵심 dependency edge, root predicate, argument structure 비교

### 계산식

### (1) Edge overlap

원문 edge 집합 $E_s$, 번안 edge 집합 $E_t$ 에 대해

$$
S_{edge} = \frac{|E_s \cap E_t|}{|E_s \cup E_t|}
$$

### (2) Root predicate preservation

$$
S_{root} = \cos(r_s, r_t)
$$

### (3) Argument preservation

$$
S_{arg} = \frac{|Arg_s \cap Arg_t|}{|Arg_s|}
$$

### (4) Depth similarity

$$
S_{depth} = 1 - \frac{|depth_s - depth_t|}{\max(depth_s, depth_t)}
$$

최종 점수는 가중합 ($S_{dep} = \alpha S_{root} + \beta S_{arg} + \gamma S_{edge} + \delta S_{depth}$)
