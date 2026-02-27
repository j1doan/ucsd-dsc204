# Performance Summary

## Runtime & Memory

| Metric | Value |
| --- | --- |
| Total runtime (sec) | 315.2443280220032 |
| Peak memory (MB) | 475.06640625 |

## Row Counts

| Metric | Value |
| --- | --- |
| Total input rows | 267424247 |
| Raw rows after parse/month filters | 267422585 |
| Intermediate pivoted rows (sum) | 144718 |
| Final wide table rows | 121480 |

## Discarded Rows

| Reason | Count | Percent of input |
| --- | --- | --- |
| Parse failures | 0 | 0.00% |
| Month mismatch | 1662 | 0.00% |
| Low-count cleanup (pivot stage) | 169778 | 0.06% |
| Total discarded (raw stage) | 1662 | 0.00% |
| Total discarded (overall) | 171440 | 0.06% |

## Date Consistency Issues

- Total month-mismatch rows: 1662
- Files with mismatches: 24

| Month | Mismatch Rows | Files Affected |
| --- | --- | --- |
| 2022-01 | 56 | 2 |
| 2022-02 | 87 | 2 |
| 2022-03 | 113 | 2 |
| 2022-04 | 193 | 2 |
| 2022-05 | 160 | 2 |
| 2022-06 | 521 | 2 |
| 2022-07 | 86 | 2 |
| 2022-08 | 67 | 2 |
| 2022-09 | 92 | 2 |
| 2022-10 | 104 | 2 |
| 2022-11 | 121 | 2 |
| 2022-12 | 62 | 2 |

## Row Breakdown by Year and Taxi Type

| Year | Taxi Type | Rows |
| --- | --- | --- |
| 2022 | fhv | 91092 |
| 2022 | green | 4089 |
| 2022 | yellow | 26299 |

## Schema Summary

- Output columns: 24

```
hour_0, hour_1, hour_2, hour_3, hour_4, hour_5, hour_6, hour_7, hour_8, hour_9, hour_10, hour_11, hour_12, hour_13, hour_14, hour_15, hour_16, hour_17, hour_18, hour_19, hour_20, hour_21, hour_22, hour_23
```
