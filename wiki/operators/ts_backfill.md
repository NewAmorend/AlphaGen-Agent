---
title: ts_backfill 算子
type: operator
tags:
- operator
- time_series
sources:
- worldquantbrain-api
created: '2026-05-22'
operator_name: ts_backfill
category: Time Series
operator_type: SCALAR
---

<!-- managed by wq-agent wiki import-wq; manual edits below -->

# `ts_backfill`

**Category**：Time Series　**Type**：SCALAR

## 签名

```
ts_backfill(x,lookback = d, k=1)
```

## 官方说明

Replaces missing (NaN) values in a time series with the most recent valid value from a specified lookback window, improving data coverage and reducing risk from missing data.

## 使用提示（人工补充）

- TODO：典型适用场景、常配合的算子（用 [[wikilink]]）、参数范围、踩坑
