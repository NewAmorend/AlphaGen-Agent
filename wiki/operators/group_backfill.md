---
title: group_backfill 算子
type: operator
tags:
- group
- operator
sources:
- worldquantbrain-api
created: '2026-05-22'
operator_name: group_backfill
category: Group
operator_type: SCALAR
---

<!-- managed by wq-agent wiki import-wq; manual edits below -->

# `group_backfill`

**Category**：Group　**Type**：SCALAR

## 签名

```
group_backfill(x, group, d, std = 4.0)
```

## 官方说明

Fills missing (NaN) values for instruments within the same group by calculating a winsorized mean of all non-NaN values over the past d days. The winsorized mean is computed by trimming extreme values based on a specified standard deviation multiplier (std, default 4.0).

## 使用提示（人工补充）

- TODO：典型适用场景、常配合的算子（用 [[wikilink]]）、参数范围、踩坑
