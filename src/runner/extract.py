"""Column value matching and extraction utilities.

Removed sqlite_sequence filter (not applicable to PostgreSQL).
Changed backtick quoting to double-quote quoting for PostgreSQL.
"""

import re
import json
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class DES:

    def __init__(self, bert_model, DB_emb, col_values) -> None:
        self.model = bert_model
        self.DB_emb = DB_emb
        self.col_values = col_values

    def get_examples(self, target, topk=3):
        target_embedding = self.model.encode(target, show_progress_bar=False)
        all_pair = []
        for key in self.DB_emb:
            embs = self.DB_emb[key]
            distances = np.squeeze(euclidean_distances(target_embedding, embs)).tolist()
            distances = [distances] if np.isscalar(distances) else distances
            pairs = [
                (distance, index, key)
                for distance, index in zip(distances, range(len(distances)))
            ]
            pairs_sorted = sorted(pairs, key=lambda x: x[0])
            all_pair.extend(pairs_sorted[:topk])
            all_pair.sort(key=lambda x: x[0])
        return all_pair[:topk]


class DES_new(DES):

    def __init__(self, bert_model, DB_emb, col_values) -> None:
        super().__init__(bert_model, DB_emb, col_values)
        self.jump_l = {
            "how", "not", "what", "who", "which", "refer", "from", "with",
            "was", "were", "the", "and", "have", "many", "much", "list", "did",
        }

    def get_key_col_des(self, cols, values, shold=0.8, debug=False, topk=7, match_filter=0.3):
        des = []
        value_cols = []
        for value in values:
            value = value.strip(" '\"")
            if len(value) == 0 or re.fullmatch(r"\d+\.?\d*", value):
                continue
            self.get_key_col_des_single(value, topk, debug, des, value_cols, shold, match_filter)
            value_split = value.split(" ")
            if len(value_split) > 1:
                tmp_des = []
                tmp_value_cols = []
                for val in value_split:
                    if val == "":
                        continue
                    val = val.strip()
                    self.get_key_col_des_single(val, topk, debug, tmp_des, tmp_value_cols, max(shold / 2, 0.2), match_filter)
                if len(tmp_des) >= len(value_split):
                    des.extend(tmp_des)
                    value_cols.extend(tmp_value_cols)

            if len(value) < 3 or re.fullmatch(r"\d+\.?\d*", value) or value in values or value.lower() in self.jump_l:
                continue
            self.get_key_col_des_single(value, topk, debug, des, value_cols, shold, match_filter, tag=3, lcs=True)
        des = sorted(list(set(des)))
        cols = cols.union(set(value_cols))
        return cols, des

    def get_key_col_des_single(self, value, topk, debug, des, value_cols, shold, match_filter, tag=10, lcs=False):
        if len(value) == 0:
            return
        col_key_inst = self.get_examples([value], topk)
        if not col_key_inst:
            return

        val_col = []
        matchs = set()
        col_key_inst = same_str_sort(col_key_inst, self.col_values, value)[:topk]
        if not col_key_inst:
            return
        maxm_score = col_key_inst[0][1]
        same = col_key_inst[0][0]
        val_len = len(value)
        for match in col_key_inst:
            tab, col = match[2].split(".")
            # Skip PostgreSQL system tables and numeric-only values
            if re.fullmatch(r"\d+\.?\d*", match[3]) or abs(val_len - len(match[3])) > tag:
                continue
            if (
                (same == match[0] or match[3].lower() == value.lower())
                and match[1] < shold
                and match[2] not in matchs
                and match[1] - maxm_score < match_filter
            ):
                coll = tab + "." + quote_field(col)
                val_ans = match[3].replace("'", "''")
                val_col.append((coll, val_ans))
                value_cols.append(coll)
                matchs.add(match[2])
            else:
                continue
        if debug:
            print(value)
            print(col_key_inst)
            print(val_col)
        if val_col:
            des.extend(val_col)


def quote_field(field_name):
    """Quote field name for PostgreSQL (using double quotes instead of backticks)."""
    if re.search(r"\W", field_name):
        return f'"{field_name}"'
    return field_name


def col_update(cols_tmp, db_col):
    col_table = {}
    for x in db_col:
        table, col = x.split(".")
        col_table.setdefault(col, [])
        col_table[col].append(table)
    cols = []
    for c in cols_tmp:
        try:
            table, col_name = c.split(".")
            col_name = col_name.replace("`", "").replace('"', "")
            col_name = quote_field(col_name)
            for t in col_table[col_name]:
                cols.append(f"{t}.{col_name}")
        except Exception:
            pass
    return set(cols)


def same_str_sort(col_key_inst, col_values, value):
    col_key_tmp = []
    for match in col_key_inst:
        val = col_values[match[2]][match[1]]
        col_key_tmp.append((val.strip() != value.strip(), match[0], match[2], val))
    col_key_tmp.sort()
    return col_key_tmp
