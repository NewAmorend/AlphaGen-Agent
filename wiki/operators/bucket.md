---
title: bucket 算子
type: operator
tags:
- operator
- transformational
sources:
- worldquantbrain-api
created: '2026-05-22'
operator_name: bucket
category: Transformational
operator_type: SCALAR
---

<!-- managed by wq-agent wiki import-wq; manual edits below -->

# `bucket`

**Category**：Transformational　**Type**：SCALAR

## 签名

```
bucket(rank(x), range=“0, 1, 0.1”, skipBoth=False, NaNGroup=False)
or
bucket(rank(x), buckets = “2,5,6,7,10”, skipBoth=False, NaNGroup=False)
```

## 官方说明

The bucket operator creates custom groups by dividing data into buckets (ranges) based on ranked values of any data field. These buckets can then be used with group operators like group_neutralize, group_rank, group_zscore etc.

## 使用提示（人工补充）

- TODO：典型适用场景、常配合的算子（用 [[wikilink]]）、参数范围、踩坑
