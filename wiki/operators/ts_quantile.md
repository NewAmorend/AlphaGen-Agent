---
title: ts_quantile 算子
type: operator
tags:
- operator
- time_series
sources:
- worldquantbrain-api
created: '2026-05-22'
operator_name: ts_quantile
category: Time Series
operator_type: SCALAR
---

<!-- managed by wq-agent wiki import-wq; manual edits below -->

# `ts_quantile`

**Category**：Time Series　**Type**：SCALAR

## 签名

```
ts_quantile(x,d, driver="gaussian" )
```

## 官方说明

Calculates the ts_rank of the input and transforms it using the inverse cumulative distribution function (quantile function) of a specified probability distribution (default: Gaussian/normal). This helps to normalize or reshape the distribution of your data over a rolling window.

## 使用提示（人工补充）

- TODO：典型适用场景、常配合的算子（用 [[wikilink]]）、参数范围、踩坑
