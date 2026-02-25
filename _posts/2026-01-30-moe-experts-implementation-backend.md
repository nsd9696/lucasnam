---
layout: post
title: "MoE Expert FFN Backend: experts_implementation"
date: 2026-01-30 19:00:00 +0900
description: "HuggingFace Transformers에서 MoE 모델의 Expert FFN 연산 백엔드(eager, batched_mm, grouped_mm)를 선택하고, Solar-Open 100B로 벤치마크한 결과"
tags: moe experts-implementation huggingface transformers torch-compile grouped-gemm
categories: ml-engineering
lang: ko
permalink: /ko/blog/2026/moe-experts-implementation-backend/
toc:
  beginning: true
---

## 1. experts_implementation

HuggingFace Transformers에 MoE 모델의 Expert FFN 연산 방식을 선택할 수 있도록 지원해주는 [PR (#42697)](https://github.com/huggingface/transformers/pull/42697)이 머지되었습니다. 기존에 `attn_implementation`으로 어텐션 연산 백엔드를 선택할 수 있었던 것처럼, 이제 expert 연산도 hooking 해서 원하는 방식으로 돌릴 수 있게 되었습니다.

---

## 2. eager, batched_mm, grouped_mm

기본적으로 expert FFN은 router가 top-k expert를 선택하고 나서 expert parameter(`gate_up_proj`, `down_proj`)로 hidden state projection을 한 다음, routing weight로 가중 합산하는 로직은 동일합니다. 다만 이때, **expert 별 행렬 곱셈을 어떻게 하는가**가 차이점입니다.

### eager: 루프 기반 레퍼런스 구현

가장 직관적인 방식이고, activated expert를 Python 루프로 하나씩 순회하면서 해당 expert에 routing 된 token들만 고르고, 해당 토큰들에 대해 per-expert projection을 수행합니다. 이때 `torch.where`를 사용해 expert에 할당된 토큰을 선택하는데, 이 과정 때문에 `torch.compile`을 `fullgraph=True` 옵션과 사용하기가 어려워집니다.

### batched_mm

`batched_mm`은 selected expert의 weight를 토큰마다 duplicate 해서 3D tensor로 쌓은 뒤, `torch.bmm`으로 한 번에 배치 행렬 곱셈을 수행하는 방식입니다.

`torch.bmm`은 Batched Matrix Multiplication의 약자로, 동일한 크기의 행렬 쌍들을 한 번에 곱합니다.

특히, `batched_mm`은 `torch.compile`에도 호환되기 때문에 `fullgraph` 지원이 가능해집니다. 다만, expert weight들을 copy 하기 때문에 메모리 사용량이 eager 대비 2배 이상 뛸 수 있다는 단점이 있어, 짧은 시퀀스나 작은 배치 사이즈에서 유리하다고 볼 수 있습니다.

### grouped_mm

`grouped_mm`은 `torch._grouped_mm`을 사용하는 방식으로 Grouped GEMM을 지원합니다. 해당 방식은 weight copy를 하지 않고, 대신 token들을 expert 별로 grouping 한 뒤 Grouped GEMM 커널로 해당 expert의 projection을 동시에 처리합니다.

weight를 복제하지 않기 때문에 메모리 효율이 가장 좋고, 특히 긴 시퀀스나 큰 배치에서 강점을 보입니다.

{% include figure.liquid loading="eager" path="assets/img/experts_impl_blog_1.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Batched Matrix Multiplication (bmm) vs Grouped GEMM Approach" %}

---

## 3. Solar-Open 100B Benchmark

PR 상에서 성능 차이가 꽤 난다고 해서, 이번에 Upstage의 Solar-Open 모델로 직접 벤치마킹을 돌려봤습니다. Mean Latency 기준으로만 봐도 유의미한 차이가 확인됐는데요. `batched_mm`은 짧고 작은 입력에서 성능이 괜찮은 편이었지만, 전반적으로는 `grouped_mm`이 가장 좋은 성능을 보여줬습니다. eager 대비로 따져보면, compile 없이도 평균 4배, compile을 적용했을 때는 최대 10배까지 latency 차이가 나는 걸 확인할 수 있었습니다.

{% include figure.liquid loading="eager" path="assets/img/solar_latency_summary.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Solar-Open 100B Latency Comparison (Experts Backend & Torch Compile)" %}

```python
model = AutoModelForCausalLM.from_pretrained(
    "upstage/solar-open-...",
    experts_implementation="grouped_mm",  # 또는 "batched_mm", "eager"
)
```

다만, `batched_mm`에서 `batch_size=4`, `seq_len=128` 조건일 때 compile default와 no-compile 케이스에서 연산 과정 중 memory spike가 관찰됐습니다. 이 부분은 별도로 수정하지 않고 결과를 그대로 남겨두었으니 참고 부탁드립니다.

---

## 4. 마무리

MoE model inference에서 해당 backend option을 고려하면 좋을 것 같습니다. 다만, vLLM에서는 자체적인 `fused_moe` kernel이 있기 때문에 바로 사용이 어렵습니다. 해당 `fused_moe` kernel은 이후 포스트에서 한번 다뤄보겠습니다.
