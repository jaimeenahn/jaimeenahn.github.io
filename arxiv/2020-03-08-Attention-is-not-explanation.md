---
title:  "Attention is not explanation"
excerpt: "Reviewing the paper"
date: 2019-10-23
categories:
  - papers
sitemap :
  changefreq : daily
  priority : 1.0
tags:
  - update
  - LaTex
  - MathJax   
toc : true
toc_label: "Table of contents"
toc_icon: "list"  # corresponding Font Awesome icon name (without fa prefix)
toc_sticky: true
classes: wide  
use_math: true
---

# Attention is not explanation

Attention의 전제 : 높은 attention score를 가진 input(e.g., words)은 모델의 결과에 연관이 있다. --> 하지만 formally 확인된 적이 없다.

Attention이 예측에 신뢰할만한 영향을 줬다고 가정하면, 다음과 같은 성질은 만족해야한다.

[!image] (src=스크린샷 2020-03-18 오후 6.13.37.png)

1. Attention의 가중치는 feature importance measure와 상관관계가 있어야한다.
2. Alternative or counterfactual Attention 가중치 구성은 예측의 변화를 초래해야한다. -> 가중치가 바뀌면 상응하는 변화가 있어야한다.

하지만 BiLSTM을 사용했을 때 어떠한 task에서도 두 가지 성질을 꾸준히 관찰할 수 없었다.
