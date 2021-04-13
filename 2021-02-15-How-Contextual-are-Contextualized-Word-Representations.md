---
title:  "[BERTology] How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings"
excerpt: ""
date: 2021-02-15
categories:
  - papers
tags:
  - BERT
  - LaTex
  - MathJax   
toc : true
toc_label: "Table of contents"
toc_icon: "list"  # corresponding Font Awesome icon name (without fa prefix)
toc_sticky: true
classes: wide  
use_math: true
comments: true
---

# How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings
---

## Contents
1. Introduction
2. Related Works
3. Background
4. EWISE
5. Result

---

### 1. Introduction

기존의 probing task (syntaxtic task)에 대해서 연구가 되던 contextual word representations이 좋은 성능을 발휘한다고 증명이 됐지만 얼마나 contextual한지는 알지 못한다.

그리하여 이 논문에서는 최근 contextual language model들이 얼마나 word vector에 문맥의 정보를 담고 있는지 조사하였다.

실험 과정은

1. SemEval Semantic Textual Similarity (STS) 2012 -- 2016 데이터에 기반하여 단어가 가질 수 있는 vector representation을 모두 리스트에 집어 넣는다.

2. *SelfSim* $$
