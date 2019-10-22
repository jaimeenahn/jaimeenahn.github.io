---
layout: post
title:  "Zero-shot Word Sense Disambiguation using Sense"
categories: 
  - papers
tags:
  - update
---



# Zero-shot Word Sense Disambiguation using Sense Definition Embedding

---

## Contents
1. Introduction
2. Related Works
3. Background
4. EWISE

---

### 1. Introduction

For the task **set of possible senses for the word is assumed for te word is assumed to be known a prori** - 사전에 전제 조건으로 모든 단어의 Sense는 정의 되어있어야한다.  

goal of WSD(Word Sense Disambiguation) = predict the right sense - 주어진 문맥과 단어에 따라 정확한 의미를 찾아내는 것  

기존의 WSD는 discrete level로 Sense를 분류 -> 학습 데이터에서 자주 등장하지 않는 단어들에 대해 일반화 능력을 저하시킨다.  

학습 데이터에서 아예 등장하지 않는 단어들에 대해선 WordNet같은 외부 데이터를 이용하여 Most-Frequent-Sense를 적용합니다.  

Knowledge-Based(*unsupervised*)를 사용하여 어휘적 자료를 이용하여 풀고자 했다. 

KB를 사용하는 경우는 보통 2가지로 나뉜다.
* Context-definition overlap([Lesk, 1986](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.178.2744&rep=rep1&type=pdf); [Basile et al., 2014](https://www.aclweb.org/anthology/C14-1151/))
* structural properties of the lexical resource ([Moro
et al., 2014](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00179); [Weissenborn et al., 2015](https://aclweb.org/anthology/P15-1058/); [Chaplot
et al., 2015](http://stanford.edu/~ashwinpp/assets/pdf/AAAI15-unsupWSD.pdf); [Chaplot and Salakhutdinov, 2018](https://arxiv.org/abs/1801.01900);[Tripodi and Pelillo, 2017](https://www.mitpressjournals.org/doi/10.1162/COLI_a_00242)).

KB는 rare and unseen word의 rare sense로의 disambiguation이 방법을 제공하지만 supervised한 방법들(MFS)이 더 좋은 성능을 가진다.

*[KB]: Knowledge-Based

최근 WSD를 neural sequnce task로 보는 논문이 등장했지만 ([Raganato et al. 2017](https://aclweb.org/anthology/D17-1120/)), annotation process가 비싼 작업이기 때문에 ([Lopez de Lacalle and Agirre, 2015](https://aclweb.org/anthology/S15-1007/)) sense-annotated data는 supervised method를 사용하기 부족하다.

supervision bottleneck for WSD를 극복하기 위해 정의를 통합하는 관심이 있엇지만 senses를 discrete labels로 처리하기 때문에 한계가 있다. ([Luo et al. 2018](https://www.aclweb.org/anthology/P18-1230/))

**Hypothesis** = 지도학습을 이용한 방법은 어휘적 정보를 활용하여 WSD에서 관찰되거나 관찰되지 않은 words and senses 모두를 향상시킬 수 있을 것. -> **Extended WSD Incorporating Sense Embedding(EWISE)**

EWISE는 기존의 방법과는 다르게 continuous labels을 사용하는데 이는 일반화된 zero-shot learning을 가능하게 한다 
- 왜? : seen sense 만큼 unseen sense도 인식할 수 있는 능력이 있다
    - 왜? :

EWISE = sense definition + additional information from lexical resourse

- definition embedding을 위해 Knowledge-Graph embedding methods 사용 ([Bordes et al., 2013](), [Dettmers et al., 2018]()).

### 2. Related Work

**Lexical Resourse** gives knowledge about words and their meanings. 최근의 연구에 따르면 neural network가 사전적 정의로 부터 의미론적인(semantic) 정보를 추출할 수 있다는게 밝혀졌다([Bahdanau et al., 2017](); [Bosc and Vincent,
2018]()). -> **represention of word meaning을 위해 dictionary definition을 사용한다.**

사전적 정의는 지금까지 WSD를 위해 많이 사용됐지만, correct sense의 정의가 단어가 사용되는 문맥과 highly overlap되어야 한다는 전제가 붙었었다. -> heuristic insight에 기반이 된 방법.

더 최근엔, definition-context 유사도를 측정하기 위해 모듈을 합치는  gloss-augmented neural approach가 제안되었다. ([Luo et al. 2018](https://www.aclweb.org/anthology/P18-1230/))

EWISE = embedding of definition을 neural model의 target으로, heuristic한 정의 말고 WordNet에서 제공하는 single definition을 사용할 것이다. 

continuous representation for definition = **Universal Sentense Representations** + **learing deep contextualized word representation**을 이용하여 definition embedding을 평가할 것이다.

구조적 데이터는 Graph를 이용하여 단어를 가장 관계 있는 Sense와 맞추는 역할을 해왔다. **EWISE에서는 definition의 representation을 더 잘 학습시키기 위해 Graph-based structural Knowledge를 사용한다.**
- TransE : entity들의 translation을 통해 relation들을 모델링한다.
- ConvE : multi-layer convolutional network를 이용하여 더 표현적인 특징을 학습할 수 있게한다.

### 4.EWISE

![Figure1](./assets/img/Kumar et al fig 1.JPG)

**EWISE**
- Attentive Context Encoder(WSD Task) : 
    1. BiLSTM을 encoder를 이용하여 sequence of token을 context-aware embedding으로 변환하고, 
    2. self-attention을 이용하여 현재 단어의 문맥을 보강한다.
    3. Projection을 이용하여 sense embedding으로 사영하여 sense를 가져온다.
- Definition Encoder :
    1. WSD task와 독립적으로 진행하고
    2. pretrained sentence encoder로 definition을 encoding한다.

#### 4-1,2 Attentive Context Encoder

$$S_{w}$$

BiLSTM이 단어의 context dependent representation을 효과적으로 생성하기 때문에 BiLSTM을 이용하여 단어를 임베딩한다.

