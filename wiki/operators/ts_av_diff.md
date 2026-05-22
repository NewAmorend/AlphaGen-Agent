---
title: ts_av_diff 算子
type: operator
tags:
- operator
- time_series
sources:
- worldquantbrain-api
created: '2026-05-22'
operator_name: ts_av_diff
category: Time Series
operator_type: SCALAR
---

<!-- managed by wq-agent wiki import-wq; manual edits below -->

# `ts_av_diff`

**Category**：Time Series　**Type**：SCALAR

## 签名

```
ts_av_diff(x, d)
```

## 官方说明

Calculates the difference between a value and its mean over a specified period, ignoring NaN values in the mean calculation. In short, it returns x – ts_mean(x, d) with NaNs ignored.

## 使用提示（人工补充）

- TODO：典型适用场景、常配合的算子（用 [[wikilink]]）、参数范围、踩坑
