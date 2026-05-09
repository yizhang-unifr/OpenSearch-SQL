# Landcover Golden SQL Pattern Summary

- total_rows: 2864
- scanned_files: Thessaly_NOA.xlsx, canton_of_zurich.xlsx, samples.xlsx, v3_full.xlsx

## Top Pattern Counts

| pattern | matched_rows | ratio |
|---|---:|---:|
| join_landcover_upscaled | 2853 | 0.996 |
| join_landcover_type | 2852 | 0.996 |
| extract_function | 2447 | 0.854 |
| round_function | 2381 | 0.831 |
| cast_function | 2375 | 0.829 |
| array_subscript | 2031 | 0.709 |
| cross_join_lateral | 1742 | 0.608 |
| unnest | 833 | 0.291 |
| with_ordinality | 833 | 0.291 |
| tuple_in_filter | 0 | 0.000 |

## Canonical Join Skeleton

```sql
SELECT ... FROM landcover_upscaled lu
CROSS JOIN LATERAL UNNEST(lu.ranks) WITH ORDINALITY AS rank_item(rank_pair, rank_idx)
JOIN landcover_type lt ON lt.code = rank_item.rank_pair[1]
WHERE ... -- optional temporal and geo filters
```
