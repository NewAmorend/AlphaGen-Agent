---
title: ts_rank 算子
type: operator
tags:
- operator
- time_series
sources:
- worldquantbrain-api
created: '2026-05-22'
operator_name: ts_rank
category: Time Series
operator_type: SCALAR
---

<!-- managed by wq-agent wiki import-wq; manual edits below -->

# `ts_rank`

**Category**：Time Series　**Type**：SCALAR

## 签名

```
ts_rank(x, d, constant = 0)
```

## 官方说明

Ranks the value of a variable for each instrument over a specified number of past days, returning the rank of the current value (optionally adjusted by a constant). Useful for normalizing time-series data and highlighting relative performance over time.

## 使用提示（人工补充）

- TODO：典型适用场景、常配合的算子（用 [[wikilink]]）、参数范围、踩坑
