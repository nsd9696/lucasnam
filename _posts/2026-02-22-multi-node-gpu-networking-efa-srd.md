---
layout: post
title: "Multi-Node P/D Disagg vLLM Serving: How EFA Works Compared to InfiniBand?"
date: 2026-02-22
description: "AWS EFA 환경에서의 멀티노드 GPU 통신, InfiniBand vs EFA 비교, vLLM P/D Disagg 구성까지"
tags: gpu networking efa infiniband rdma vllm
categories: infrastructure
lang: ko
permalink: /ko/blog/2026/multi-node-gpu-networking-efa-srd/
toc:
  beginning: true
---

## 1. 멀티노드 GPU 요구사항

단일노드에서 GPU 끼리 통신할 때는 NVLink가 초당 수백 GB 대역폭을 제공합니다. NVLink 는 GPU 간의 직접 연결 (point-to-point)을 가능하게 해서 CPU 를 거치지 않고 GPU 끼리 직접 통신을 가능하게 합니다.

문제는 멀티노드로 가면서 노드간 네트워크로 병목이 옮겨집니다. 특히 LLM 서빙에서 Tensor Parallelism 의 AllReduce 나 Disaggregated Serving(Prefill / Decode 분리) 에서의 KV cache transfer 가 그것이죠. 이때 당연히 TCP/IP 는 요구를 충족하지 못합니다.

### TCP/IP의 한계

TCP 는 매 패킷마다 커널 네트워크 스택들을 통과해야 합니다. 이때 system call, context switch, protocol 처리, buffer copy 같은게 연쇄적으로 발생해서 아주아주 느려지죠. 이 과정에서 불필요한 memcopy 도 같이 발생합니다.

```
TCP/IP 경로 (기존):
GPU → cudaMemcpy → CPU RAM → send() → 커널 → NIC
  → 네트워크 →
NIC → 커널 → recv() → CPU RAM → cudaMemcpy → GPU

RDMA + GPUDirect 경로:
GPU HBM → NIC (GPU 메모리 직접 DMA)
  → 네트워크 →
NIC → GPU HBM (직접 DMA)
```

RDMA(Remote Direct Memory Access)는 CPU와 커널을 우회하여 NIC가 직접 메모리에 접근할 수 있게 합니다. OS Bypass를 통해 system call과 context switch 오버헤드를 제거하고, GPUDirect RDMA는 여기서 한 발 더 나아가 NIC가 GPU 메모리(HBM)에 직접 DMA를 수행할 수 있게 합니다.

### InfiniBand

이런 GPUDirect RDMA 까지 가능하게 한 것이 InfiniBand이고 HDR(A100 표준), NDR(H100 표준)과 같은 세대에 따라 포트당 대역폭이 약 2배 정도씩(200Gbps → 400Gbps) 더 증가한다고 보시면 될 것 같습니다.

---

## 2. EFA와 SRD

### EFA (Elastic Fabric Adapter)

EFA(Elastic Fabric Adapter)는 AWS 에서 설계한 고성능 네트워크 인터페이스인데 InfiniBand 같은거라고 보시면 될 것 같습니다. 특정 instance type들(p4d, p5, p6 등) 에서 사용할 수 있고 위에서 설명드린 OS bypass 나 RDMA의 기능들을 제공합니다.

### 이더넷 vs InfiniBand

EFA의 가장 큰 차이점은 표준 이더넷 패브릭 위에서 동작한다는 점입니다. InfiniBand 는 NVIDIA(Mellanox) 단일 벤더입니다. 그래서 스위치나 NIC, 케이블 모두 Mellanox 제품이어야 하고 대안이 거의 없습니다. 반대로 이더넷 장비는 Broadcom, Intel 등의 다수 벤더가 가능한 범용적인 장비입니다.

### RDMA 전송 모드

| 모드                                 | 전용 연결 필요          | 전달 보장 | 순서 보장     | 설명                         |
| ------------------------------------ | ----------------------- | --------- | ------------- | ---------------------------- |
| **RC** (Reliable Connection)         | 필요 (서버마다 전용 QP) | O         | O             | InfiniBand 기본 전송 방식    |
| **UD** (Unreliable Datagram)         | 불필요 (하나의 QP)      | X         | X             | 가장 가벼운 방식             |
| **RD** (Reliable Datagram)           | 불필요 (하나의 QP)      | O         | O             | 이론적으로 이상적이나 미구현 |
| **SRD** (Scalable Reliable Datagram) | 불필요                  | O         | X (SW 재정렬) | AWS가 설계한 방식            |

```
RC (Reliable Connection):
서버 A ──전용 회선──→ 서버 B
서버 A ──전용 회선──→ 서버 C
→ 서버마다 전용 QP(Queue Pair)가 필요

UD (Unreliable Datagram):
서버 A ──회선──→ 아무에게나 보낼 수 있음
→ 전용 회선 불필요, 하나의 QP로 모든 상대와 통신

RD (Reliable Datagram):
서버 A ──회선──→ 아무에게나 보낼 수 있음
→ 전용 회선 불필요 + 전달 보장 + 순서 보장
```

### RD가 실제로 사용되지 않는 이유

다만, InfiniBand 에서 실질적으로 RD 를 잘 사용하고 있지는 않습니다. 이유는 결국 하드웨어 구현 난이도 인데요.

- **RC** 는 QP 하나당 상태를 하나만 추적하면 되고, **UD**는 아무에게나 보내는 것이기 때문에 사실상 상태 추적을 할 필요가 없습니다.
- 하지만 **RD**의 NIC 는 모든 통신 상대별 상태를 추적해야 합니다:
  - "서버 B에게 패킷 3까지 보냈고 ACK 2까지 받음"
  - "서버 C에게 패킷 7까지 보냈고 ACK 5까지 받음"
  - "서버 D에게 패킷 1 보냈고 ACK 아직 없음"
  - "서버 E에게 패킷 12까지 보냈고 ACK 10까지 받음"
  - ... × 통신하는 모든 상대방
- 예를 들어, 하나의 QP가 1000개의 서버와 통신하면 1000개의 상태를 NIC 에서 모두 관리해야 합니다. 이때 NIC 칩 안의 SRAM 에 저장해야 하는데 이게 비싸고 작습니다.
- 따라서 NVIDIA 입장에서는 RC 나 UD 로도 커버가 가능한데 RD까지의 수요가 없었고 비용 대비 효과가 안 맞아서 deprecate 합니다.

### SRD (Scalable Reliable Datagram)

그러면 AWS 에서 만든 SRD는 뭐가 다른가? **RD 에서의 순서 보장을 포기했습니다.**

```
RD:   패킷 1 → 패킷 2 → 패킷 3  (반드시 이 순서로 도착)
      경로: A ━━━━━━━━━━━━━→ B  (하나의 경로)

SRD:  패킷 1 ──경로 A──→ ┐
      패킷 2 ──경로 B──→ ├→ 도착 후 소프트웨어가 재정렬
      패킷 3 ──경로 C──→ ┘
```

순서를 포기하면 여러 경로에 패킷을 분산시킬 수 있고, 한 경로가 막혀도 다른 경로로 보내면 되기 때문에 이더넷 환경에서 유리합니다. 수천 대의 서버가 네트워크를 공유하는 환경에서는 이런 SRD 가 안정적이기 때문에 채택했다고 보시면 될 것 같습니다.

---

## 3. EFA → GPUDirect RDMA

EFA 를 사용했을 때 GPUDirect RDMA 흐름은 다음과 같습니다.

{% include figure.liquid loading="eager" path="assets/img/efa_blog_1.png" class="img-fluid rounded z-depth-1" zoomable=true caption="EFA GPUDirect RDMA Flow" %}

물론 이때 각 노드별 GPU 끼리의 통신에는 NVLink를 사용합니다.

### InfiniBand vs EFA 성능 비교

다만, EFA 가 InfiniBand 와 비교했을 때 performance 가 나오는가? 이를 비교하는 벤치마크를 요약하면 다음과 같습니다.

| 항목                   | InfiniBand     | EFA/이더넷          | 결론           |
| ---------------------- | -------------- | ------------------- | -------------- |
| **소규모 메시지 지연** | ~1 µs          | ~10 µs              | IB 압도적 우위 |
| **대용량 전송 대역폭** | ~200 Gbps      | ~200 Gbps           | 비슷           |
| **AI 학습 (대규모)**   | 기준선         | 적절한 튜닝 시 유사 | 차이 미미      |
| **AI 추론 (Decode)**   | 유리           | 평균 1.0166% 느림   | IB 약간 유리   |
| **비용**               | 1.5~2.5배 비쌈 | 기준선              | 이더넷 유리    |

> 출처: [WWT - The Battle of AI Networking](https://www.wwt.com/blog/the-battle-of-ai-networking-ethernet-vs-infiniband), [Vitex Tech - InfiniBand vs Ethernet](https://www.vitextech.com/blogs/blog/infiniband-vs-ethernet-for-ai-clusters-effective-gpu-networks-in-2025)

전반적으로 성능 자체는 InfiniBand가 좋은 것으로 보입니다. 다만, AWS 환경에 셋업이 되어 있는 것을 고려한다면 EFA가 유리할 수 있지만 최근 neo-cloud GPU 가격이 AWS 에 비해 저렴하게 많이 나오는 편인 것을 고려하면 InfiniBand가 더 나은 선택일 수 있을 것 같습니다.

---

## 4. EFA 위에서 P/D Disagg vLLM Serving

저의 경우 이번에 작업하던 내용이 A100 환경에서 Prefill/Decode Disagg 를 구성하는 것이였습니다. 해당 경우, EFA 가 직접적으로 관여하는 구간은 kv_transfer 입니다.

### KV Cache Transfer 소프트웨어 스택

{% include figure.liquid loading="eager" path="assets/img/efa_blog_2.png" class="img-fluid rounded z-depth-1" zoomable=true caption="KV Cache Transfer Software Stack on EFA" %}

이제 EFA 위의 각각의 구성요소들을 구체적으로 봐보겠습니다.

### NIXL (NVIDIA Inference Xfer Library)

NIXL은 GPU간의 메모리 전송을 위한 전용 라이브러리입니다. 기존의 NCCL 은 AllReduce 나 All-to-All 과 같은 집합 통신에는 특화되어 있는데, 특정 GPU → GPU 로 메모리 블록을 직접 옮기는 point-to-point 전송에는 적합하지 않아서 이를 위해 만들어진 것이 NIXL 입니다.

P/D Disagg 에서의 `Prefill GPU ──RDMA Write──▶ Decode GPU` 과정이나, Decode 인스턴스 간의 마이그레이션 `Decode GPU (서버 A, 과부하) ──RDMA──▶ Decode GPU (서버 B, 여유)` 이런 경우에서 사용하게 됩니다.

이때, NIXL 에는 NIXL Agent 가 있는데 메모리 등록이나, 메타데이터(GPU 메모리 주소, RDMA key, NIC 주소 등) 관리, 실제 전송을 수행하는 플러그인 백엔드(UCX, libfabric) 등을 관리합니다.

### UCX (Unified Communication X)

본래 InfiniBand 를 위한 범용 통신 프레임워크이며 내부적으로 RC, UD, TCP, cuda_ipc (같은 노드 GPU 간), cuda_copy(GPU ↔ CPU 복사) 등의 프로토콜을 지원합니다.

다만, EFA 에서는 `UCX_TLS=ib` 를 설정하면 사용이 가능한데, 이는 EFA가 ibverbs 호환 인터페이스를 제공하기 때문입니다.

- [EFA ibverbs 구현 (AWS driver)](https://github.com/amzn/amzn-drivers/blob/master/kernel/linux/efa/src/efa_verbs.c)
- [rdma-core EFA provider](https://github.com/linux-rdma/rdma-core/blob/master/providers/efa/efa.c)

### libfabric (Open Fabrics Interface)

네트워크 하드웨어 벤더들의 표준 추상화 계층입니다. 내부적으로 EFA provider, TCP provider, SHM provider 등으로 구성되어 있습니다.

공식적으로는 libfabric 을 사용하는 것이 맞는 것으로 보이는데 실제 작업 과정에서는 UCX 쪽이 dependency 가 잘 맞고 libfabric 쪽에서는 GPU memory bad address 이슈가 지속적으로 발생해서 우선 저의 경우는 deprecate 했습니다.

### aws-ofi-nccl (optional)

저의 경우 P/D Disagg 구성이었기 때문에 NCCL 사용 필요성이 따로 없었습니다. 다만, 모델 자체가 커서 multi-node 로 구성하거나 multi-node train의 경우 NCCL 이 필요하고 EFA 환경에서는 aws-ofi-nccl 이 필요합니다.

본래 NCCL 은 InfiniBand 를 위해 설계되어서 IB Verbs API 를 직접 호출하는 `net_ib` 전송이 내장되어 있는데 aws-ofi-nccl 은 NCCL의 네트워크 플러그인 API 를 구현해서 이를 libfabric 의 RDM 인터페이스로 변환합니다.

### efa-nv-peermem

NIC 가 GPU 메모리에 직접 접근할 수 있게 해주는 커널 모듈입니다. 본래는 nvidia-peermem 인데 EFA 에서는 efa-nv-peermem 을 사용합니다.

기본적으로 NIC 는 CPU 메모리만 읽을 수 있기 때문에 GPU 와 NIC, 이 둘을 연결하는게 efa-nv-peermem인겁니다. (GPUDirect RDMA 모듈입니다)
