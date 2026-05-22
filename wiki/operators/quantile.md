---
title: quantile 算子
type: operator
tags:
- cross_sectional
- operator
sources:
- worldquantbrain-api
created: '2026-05-22'
operator_name: quantile
category: Cross Sectional
operator_type: SCALAR
---

<!-- managed by wq-agent wiki import-wq; manual edits below -->

# `quantile`

**Category**：Cross Sectional　**Type**：SCALAR

## 签名

```
quantile(x, driver = gaussian, sigma = 1.0)
```

## 官方说明

Ranks and shifts a vector of Alpha values, then applies a chosen statistical distribution (gaussian, cauchy, or uniform) to reduce outliers. The sigma parameter controls the scale of the output.

## 使用提示（人工补充）

- TODO：典型适用场景、常配合的算子（用 [[wikilink]]）、参数范围、踩坑
