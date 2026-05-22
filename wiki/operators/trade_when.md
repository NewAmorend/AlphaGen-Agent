---
title: trade_when 算子
type: operator
tags:
- operator
- transformational
sources:
- worldquantbrain-api
created: '2026-05-22'
operator_name: trade_when
category: Transformational
operator_type: SCALAR
---

<!-- managed by wq-agent wiki import-wq; manual edits below -->

# `trade_when`

**Category**：Transformational　**Type**：SCALAR

## 签名

```
trade_when(x, y, z)
```

## 官方说明

The trade_when operator changes Alpha values only when a specific condition is met, keeps previous values otherwise, and can close positions by assigning NaN under an exit condition. It is useful for reducing turnover and controlling when trades are executed.

## 使用提示（人工补充）

- TODO：典型适用场景、常配合的算子（用 [[wikilink]]）、参数范围、踩坑
