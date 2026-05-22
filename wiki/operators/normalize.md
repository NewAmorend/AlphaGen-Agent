---
title: normalize 算子
type: operator
tags:
- cross_sectional
- operator
sources:
- worldquantbrain-api
created: '2026-05-22'
operator_name: normalize
category: Cross Sectional
operator_type: SCALAR
---

<!-- managed by wq-agent wiki import-wq; manual edits below -->

# `normalize`

**Category**：Cross Sectional　**Type**：SCALAR

## 签名

```
normalize(x, useStd = false, limit = 0.0)
```

## 官方说明

Centers a daily cross section by subtracting the market mean; optionally divide by the cross sectional standard deviation and clamp the result to [?limit, +limit]. NaNs are ignored in mean/std.

## 使用提示（人工补充）

- TODO：典型适用场景、常配合的算子（用 [[wikilink]]）、参数范围、踩坑
