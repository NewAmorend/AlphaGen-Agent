---
title: kth_element 算子
type: operator
tags:
- operator
- time_series
sources:
- worldquantbrain-api
created: '2026-05-22'
operator_name: kth_element
category: Time Series
operator_type: SCALAR
---

<!-- managed by wq-agent wiki import-wq; manual edits below -->

# `kth_element`

**Category**：Time Series　**Type**：SCALAR

## 签名

```
kth_element(x, d, k, ignore=“NaN”)
```

## 官方说明

Returns the K-th value from a time series by looking back over a specified number of (‘d’) days, with the option to ignore certain values. Commonly used for backfilling missing data.

## 使用提示（人工补充）

- TODO：典型适用场景、常配合的算子（用 [[wikilink]]）、参数范围、踩坑
