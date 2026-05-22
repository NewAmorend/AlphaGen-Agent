---
title: sign 算子
type: operator
tags:
- arithmetic
- operator
sources:
- worldquantbrain-api
created: '2026-05-22'
operator_name: sign
category: Arithmetic
operator_type: SCALAR
---

<!-- managed by wq-agent wiki import-wq; manual edits below -->

# `sign`

**Category**：Arithmetic　**Type**：SCALAR

## 签名

```
sign(x)
```

## 官方说明

Returns the sign of a number: +1 for positive, -1 for negative, and 0 for zero. If the input is NaN, returns NaN.

Input: Value of 7 instruments at day t: (2, -3, 5, 6, 3, NaN, -10)
Output: (1, -1, 1, 1, 1, NaN, -1)

## 使用提示（人工补充）

- TODO：典型适用场景、常配合的算子（用 [[wikilink]]）、参数范围、踩坑
