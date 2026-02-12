"""SQL correction and alignment utilities – PostgreSQL version.

All ``sqlite3`` calls have been replaced with ``psycopg2`` via the
``execution`` module helpers.
"""

import os
import re
import json
import time
import random

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from func_timeout import func_timeout, FunctionTimedOut

from runner.execution import _get_pg_connection, _normalize_sql, sql_exec


# ── Raw SQL parsing ────────────────────────────────────────────────────────

def sql_raw_parse(sql: str, return_question: bool = False):
    sql = sql.split("/*")[0].strip().replace("```sql", "").replace("```", "")
    sql = re.sub("```.*?", "", sql)
    rwq = None
    if return_question:
        rwq, sql = sql.split("#SQL:")
    else:
        sql = sql.split("#SQL:")[-1]
    if sql.startswith('"') or sql.startswith("'"):
        sql = sql[1:-1]
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql, rwq


# ── LLM wrappers ──────────────────────────────────────────────────────────

def get_sql(chat_model, prompt, temp=1.0, return_question=False, top_p=None, n=1, single=True):
    sql = chat_model.get_ans(prompt, temp, top_p=top_p, n=n, single=single)
    if single:
        return sql_raw_parse(sql, return_question)
    else:
        return [x["message"]["content"] for x in sql], ""


# ── SQL manipulation helpers ──────────────────────────────────────────────

def retable(sql: str) -> str:
    """Expand T1/T2 aliases back to original table names."""
    table_as = re.findall(r" ([^ ]*) +AS +([^ ]*)", sql)
    for x in table_as:
        sql = sql.replace(f"{x[1]}.", f"{x[0]}.")
    return sql


def max_fun_check(sql_retable):
    fun_amb = re.findall(r"= *\( *SELECT *(MAX|MIN)\((.*?)\) +FROM +(\w+)", sql_retable)
    order_amb = set(re.findall(r"= (\(SELECT .* LIMIT \d\))", sql_retable))
    select_amb = set(re.findall(r"^SELECT[^\(\)]*? ((MIN|MAX)\([^\)]*?\)).*?LIMIT 1", sql_retable))
    return fun_amb, order_amb, select_amb


def foreign_pick(sql):
    matchs = re.findall(r"ON\s+(\w+\.\w+)\s*=\s*(\w+\.\w+) ", sql)
    ma_all = [x for y in matchs for x in y]
    return set(ma_all)


def column_pick(sql, db_col, foreign_set):
    matchs = foreign_pick(sql)
    cols = set()
    col_table = {}
    ans = set()
    sql_select = set(re.findall(r"SELECT (.*?) FROM ", sql))
    for x in db_col:
        if sql.find(x) != -1:
            cols.add(x)
        table, col = x.split(".")
        col_table.setdefault(col, [])
        col_table[col].append(table)
    for col in cols:
        table, col_name = col.split(".")
        flag = True
        for x in sql_select:
            if x.find(col) != -1:
                flag = False
                break
        if flag and (col in foreign_set or x in matchs):
            continue
        if col_table.get(col_name):
            Ambiguity = []
            for t in col_table[col_name]:
                tbc = f"{t}.{col_name}"
                if tbc != col:
                    Ambiguity.append(tbc)
            if len(Ambiguity):
                amb_des = col + ": " + ", ".join(Ambiguity)
                ans.add(amb_des)
    return sorted(list(ans))


def values_pick(vals, sql):
    val_dic = {}
    ans = set()
    try:
        for val in vals:
            val_dic.setdefault(val[1], [])
            val_dic[val[1]].append(val[0])
        for val in val_dic:
            in_sql, not_sql = [], []
            if sql.find(val):
                for x in val_dic[val]:
                    if sql.find(x) != -1:
                        in_sql.append(f"{x} = '{val}'")
                    else:
                        not_sql.append(f"{x} = '{val}'")
            if len(in_sql) and len(not_sql):
                ans.add(f"{', '.join(in_sql)}: {', '.join(not_sql)}")
        return sorted(list(ans))
    except Exception:
        return []


def func_find(sql):
    fun_amb = re.findall(r"\( *SELECT *(MAX|MIN)\((.*?)\) +FROM +(\w+)", sql)
    fun_str = []
    for fun in fun_amb:
        fuc = fun[0]
        col = fun[1]
        table = fun[2]
        order = "DESC" if fuc == "MAX" else "ASC"
        str_fun = f"(SELECT {fuc}({col}) FROM {table}): ORDER BY {table}.{col} {order} LIMIT 1"
        fun_str.append(str_fun)
    return "\n".join(fun_str)


t1_tabe_value = re.compile(r"(\w+\.[\w]+) =\s*'([^']+(?:''[^']*)*)'")
t2_tab_val = re.compile(r"(\w+\.`[^`]*?`) =\s*'([^']+(?:''[^']*)*)'")


# ── PostgreSQL-based filter & join exec ───────────────────────────────────

def filter_sql(b, bx, conn, SQL, chars=""):
    flag = False
    for x in b:
        sql_t = SQL.replace(bx, f"{chars}{x}")
        try:
            df = pd.read_sql_query(sql_t, conn)
        except Exception as e:
            print(e)
            df = []
        if len(df):
            SQL = sql_t
            flag = True
            break
    return SQL, flag


def join_exec(bx, al, question, SQL, chat_model):
    """Execute JOIN correction against PostgreSQL."""
    flag = False
    conn = _get_pg_connection()
    conn.autocommit = True
    try:
        if bx.startswith("IN"):
            b = bx[2:].strip(" ()").split(",")
            SQL, flag = filter_sql(b, bx, conn, SQL, chars="= ")
        elif al.find("OR") != -1:
            a = al.split("OR")
            SQL, flag = filter_sql(a, al, conn, SQL)
    finally:
        conn.close()
    return SQL, flag


def gpt_join_corect(SQL, question, chat_model):
    prompt = f"""The following SQL uses an incorrect JOIN with OR / IN conditions.
Keep only the highest-priority equality condition.

question: {question}
SQL: {SQL}

Return only the new SQL:
#SQL:"""
    SQL = get_sql(chat_model, prompt, 0.0)[0].split("SQL:")[-1]
    return SQL


def select_check(SQL, db_col, chat_model, question):
    select = re.findall(r"^SELECT.*?\|\| ' ' \|\| .*?FROM", SQL)
    if select:
        SQL = SQL.replace("|| ' ' ||", ", ")

    select_amb = re.findall(r"^SELECT.*? (\w+\.\*).*?FROM", SQL)
    if select_amb:
        prompt = f"""The database has the following columns:
{db_col}
Question: {question}
SQL: {SQL}
Replace {select_amb[0]} with the appropriate id column.
Return only the SQL:
#SQL:"""
        SQL = get_sql(chat_model, prompt, 0.0)[0].split("SQL:")[-1]
    return SQL


# ── soft_check class ──────────────────────────────────────────────────────

class soft_check:

    def __init__(self, bert_model, chat_model, soft_prompt, correct_dic, correct_prompt, vote_prompt=""):
        self.bert_model = bert_model
        self.chat_model = chat_model
        self.soft_prompt = soft_prompt
        self.correct_dic = correct_dic
        self.correct_prompt = correct_prompt
        self.vote_prompt = vote_prompt

    def vote_chose(self, SQLs, question):
        all_sql = "\n\n".join(SQLs)
        prompt = self.vote_prompt.format(question=question, sql=all_sql)
        SQL_vote = get_sql(self.chat_model, prompt, 0.0)[0]
        return SQL_vote

    def soft_correct(self, SQL, question, new_prompt, hint=""):
        soft_p = self.soft_prompt.format(SQL=SQL, question=question, hint=hint)
        soft_SQL = self.chat_model.get_ans(soft_p, 0.0)
        soft_SQL = re.sub(r"```\w*", "", soft_SQL)
        soft_json = json.loads(soft_SQL)
        if (soft_json["Judgment"] is False or soft_json["Judgment"] == "False") and soft_json["SQL"] != "":
            SQL = soft_json["SQL"]
            SQL = re.sub(r"\s+", " ", SQL).strip()
        elif soft_json["Judgment"] is False or soft_json["Judgment"] == "False":
            SQL = get_sql(self.chat_model, new_prompt, 1.0, False)[0]
        return SQL, soft_json["Judgment"]

    def double_check(self, new_prompt, values, values_final, SQL, question, new_db_info, db_col, hint=""):
        SQL = re.sub(r"(COUNT)(\([^\(\)]*? THEN 1 ELSE 0.*?\))", r"SUM\2", SQL)
        sql_retable = retable(SQL)
        SQL = self.values_check(sql_retable, values, values_final, SQL, question, new_db_info, db_col, hint)
        SQL = self.JOIN_error(SQL, question)
        SQL = self.func_check(sql_retable, SQL, question)
        SQL = self.func_check2(question, SQL)
        SQL = self.time_check(SQL)
        SQL = self.is_not_null(SQL)
        SQL = select_check(SQL, db_col, self.chat_model, question)
        return SQL, True

    def double_check_style_align(self, SQL, question, db_col, sql_retable):
        SQL = self.func_check(sql_retable, SQL, question)
        SQL = self.is_not_null(SQL)
        SQL = select_check(SQL, db_col, self.chat_model, question)
        return SQL, True

    def double_check_function_align(self, SQL, question):
        SQL = self.JOIN_error(SQL, question)
        SQL = self.func_check2(question, SQL)
        SQL = self.time_check(SQL)
        return SQL, True

    def double_check_agent_align(self, sql_retable, values, values_final, SQL, question, new_db_info, db_col, hint=""):
        SQL = self.values_check(sql_retable, values, values_final, SQL, question, new_db_info, db_col, hint)
        return SQL, True

    def JOIN_error(self, SQL, question):
        join_mutil = re.findall(
            r"JOIN\s+\w+(\s+AS\s+\w+){0,1}\s+ON(\s+\w+\.\w+\s*(=\s*\w+\.\w+(?:\s+OR\s+\w+\.\w+\s*=\s*\w+\.\w+)+|IN\s+\(.*?\)))",
            SQL,
        )
        flag = False
        if join_mutil:
            _, al, bx = join_mutil[0]
            try:
                _join_timeout = int(os.environ.get("JOIN_EXEC_TIMEOUT", "60"))
                SQL, flag = func_timeout(_join_timeout, join_exec, args=(bx, al, question, SQL, self.chat_model))
            except FunctionTimedOut:
                print("time out join")
            except Exception as e:
                print(e)
        if not flag and join_mutil:
            SQL = gpt_join_corect(SQL, question, self.chat_model)
            print("soft change JOIN gpt")
        return SQL

    def is_not_null(self, SQL):
        SQL = SQL.strip()
        inn = re.findall(r"ORDER BY .*?(?<!DESC )LIMIT +\d+;{0,1}", SQL)
        if not inn:
            return SQL
        for x in inn:
            if re.findall(r"SUM\(|COUNT\(", x):
                return SQL
        prompt = f"""Add WHERE IS NOT NULL for ORDER BY columns:
SQL: {SQL}

Return only the new SQL:
#SQL:"""
        SQL = get_sql(self.chat_model, prompt, 0.0)[0].split("SQL:")[-1]
        return SQL

    def time_check(self, sql):
        """Fix PostgreSQL date/time comparisons.

        Replace strftime patterns with EXTRACT and ensure proper quoting.
        """
        # Fix strftime → EXTRACT
        sql = re.sub(r"strftime\s*\(\s*'%Y'\s*,\s*(\w+)\s*\)", r"EXTRACT(YEAR FROM \1)", sql)
        sql = re.sub(r"strftime\s*\(\s*'%m'\s*,\s*(\w+)\s*\)", r"EXTRACT(MONTH FROM \1)", sql)
        sql = re.sub(r"strftime\s*\(\s*'%d'\s*,\s*(\w+)\s*\)", r"EXTRACT(DAY FROM \1)", sql)
        # Ensure numeric comparisons are quoted
        sql = re.sub(r"(EXTRACT\([^\)]+\)\s*[>=<]+\s*)(\d{4,})", r"\1'\2'", sql)
        return sql

    def func_check2(self, question, SQL):
        res = re.search(r"ORDER BY ((MIN|MAX)\((.*?)\)).*? LIMIT \d+", SQL)
        if res:
            prompt = f"""For the following question and SQL:
#question: {question}
#SQL: {SQL}

ERROR: {res.group()} is incorrect usage. Fix the SQL, check if GROUP BY needs SUM({res.groups()[2]}).

Return only the new SQL:"""
            SQL = get_sql(self.chat_model, prompt, 0.1)[0]
        return SQL

    def func_check(self, sql_retable, sql, question):
        fun_amb, order_amb, select_amb = max_fun_check(sql_retable)
        if not fun_amb and not order_amb and not select_amb:
            return sql
        fun_str = []
        origin_f = []
        for fun in fun_amb:
            fuc, col, table = fun
            order = "DESC" if fuc == "MAX" else "ASC"
            str_fun = f"WHERE {col} = (SELECT {fuc}({col}) FROM {table}): Use ORDER BY {table}.{col} {order} LIMIT 1"
            origin_f.append(f"WHERE {col} = (SELECT {fuc}({col}) FROM {table})")
            fun_str.append(str_fun)
        for fun in order_amb:
            origin_f.append(fun)
            fun_str.append(f"{fun}: Use JOIN instead of nesting")
        for fun in select_amb:
            origin_f.append(fun[0])
            fun_str.append(f"{fun[0]}: redundant {fun[1]} function, fix it")
        func_amb = "\n".join(fun_str)
        prompt = f"""Fix the SQL based on the ERROR and change ambiguity notes:
#question: {question}
#SQL: {sql}
ERROR: {','.join(origin_f)} does not meet requirements, use JOIN ORDER BY LIMIT instead
#change ambiguity: {func_amb}

Return only the new SQL:"""
        sql = get_sql(self.chat_model, prompt, 0.0)[0]
        return sql

    def values_check(self, sql_retable, values, values_final, sql, question, new_db_info, db_col, hint=""):
        dic_v = {}
        dic_c = {}
        l_v = list(set([x[1] for x in values]))
        tables = "( " + " | ".join(set([x.split(".")[0] for x in db_col])) + " )"
        for x in values:
            dic_v.setdefault(x[1], [])
            dic_v[x[1]].append(x[0])
            dic_c.setdefault(x[0], [])
            dic_c[x[0]].append(x[1])
        value_sql = re.findall(t1_tabe_value, sql_retable)
        value_sql.extend(re.findall(t2_tab_val, sql_retable))
        tabs = set(re.findall(tables, sql))
        if len(tabs) == 1:
            val_single = re.findall(r"[ \(]([\w]+) =\s*'([^']+(?:''[^']*)*)'", sql)
            val_single = set(val_single)
            tab = tabs.pop()[1:-1]
            for x in val_single:
                value_sql.append((f"{tab}.{x[0]}", x[1]))
        badval_l = []
        change_val = []
        value_sql = set(value_sql)
        for tab_val in value_sql:
            tab, val = tab_val
            if len(re.findall(r"\d", val)) / len(val) > 0.6:
                continue
            tmp_col = dic_v.get(val)
            if not tmp_col and len(l_v):
                val_close = self.bert_model.encode(val, show_progress_bar=False) @ self.bert_model.encode(l_v, show_progress_bar=False).T
                if val_close.max() > 0.95:
                    val_new = l_v[val_close.argmax()]
                    sql = sql.replace(f"'{val}'", f"'{val_new}'")
                    val = val_new
            tmp_col = dic_v.get(val)
            tmp_val = dic_c.get(tab, {})
            if tmp_col and tab not in tmp_col:
                lt = [f"{x} ='{val}'" for x in tmp_col]
                lt.extend([f"{x} ='{val}'" for x in tmp_val])
                rep = ", ".join(lt)
                badval_l.append(f"{tab} = '{val}'")
                change_val.append(f"{tab} = '{val}': {rep}")
        if badval_l:
            v_l = "\n".join(change_val)
            prompt = f"""Database Schema:
{new_db_info}

#question: {question}
#SQL: {sql}
ERROR: The following values do not exist in the database: {', '.join(badval_l)}
Rewrite SQL using the following conditions:
{v_l}

Return only the new SQL:
#SQL:"""
            sql = get_sql(self.chat_model, prompt, 0.0)[0]
        return sql

    def correct_sql(self, sql, query, db_info, hint, key_col_des, new_prompt, db_col=None, foreign_set=None, L_values=None):
        """Iteratively correct SQL by executing against PostgreSQL."""
        db_col = db_col or {}
        foreign_set = foreign_set or set()
        L_values = L_values or []

        conn = _get_pg_connection()
        conn.autocommit = True
        count = 0
        raw = sql
        none_case = False

        max_correction_rounds = int(os.environ.get("MAX_CORRECTION_ROUNDS", "2"))
        while count <= max_correction_rounds:
            try:
                df = pd.read_sql_query(_normalize_sql(sql), conn)
                if len(df) == 0:
                    raise ValueError("Error: Result: None")
                else:
                    break
            except Exception as e:
                if count >= max_correction_rounds:
                    sql = get_sql(self.chat_model, new_prompt, 0.2)[0]
                    none_case = True
                    break
                count += 1
                tag = str(e)
                e_s = str(e).split("':")[-1]
                result_info = f"{sql}\nError: {e_s}"

            if sql.find("SELECT") == -1:
                sql = get_sql(self.chat_model, new_prompt, 0.3)[0]
            else:
                fewshot = self.correct_dic.get("default", "")
                advice = ""
                for x in self.correct_dic:
                    if tag.find(x) != -1:
                        fewshot = self.correct_dic[x]
                        if e_s == "Result: None":
                            sql_re = retable(sql)
                            adv = column_pick(sql_re, db_col, foreign_set)
                            adv = "\n".join(adv)
                            val_advs = values_pick(L_values, sql_re)
                            val_advs = "\n".join(val_advs)
                            func_call = func_find(sql)
                            if len(adv) or len(val_advs) or len(func_call):
                                advice = "#Change Ambiguity: (replace or add)\n"
                                l = [x for x in [adv, val_advs, func_call] if len(x)]
                                advice += "\n\n".join(l)
                        elif x == "does not exist":
                            advice += "Please check if this column exists in other tables"
                        break
                fewshot = ""
                advice = ""
                cor_prompt = self.correct_prompt.format(
                    fewshot=fewshot,
                    db_info=db_info,
                    key_col_des=key_col_des,
                    q=query,
                    hint=hint,
                    result_info=result_info,
                    advice=advice,
                )
                sql = get_sql(self.chat_model, cor_prompt, 0.2 + count / 5, top_p=0.3)[0]
            raw = sql
        conn.close()
        return sql, none_case


# ── SQL execution wrapper (PostgreSQL) ────────────────────────────────────

def get_sql_ans(SQL):
    """Execute SQL against PostgreSQL, return (result_set, time_cost)."""
    try:
        _exec_timeout = int(os.environ.get("SQL_EXEC_TIMEOUT", "30"))
        ans, time_cost = func_timeout(_exec_timeout, sql_exec, args=(SQL,))
    except FunctionTimedOut:
        ans, time_cost = [], 100000
        print("time out")
    except Exception as e:
        ans, time_cost = [], 100000
        print(f"SQL execution error: {e}")
    return ans, time_cost


# ── Multi-candidate processing ────────────────────────────────────────────

def process_sql(Dcheck, SQL, L_values, values, question, new_db_info, db_col_keys, hint, key_col_des, tmp_prompt, db_col, foreign_set, align_methods):
    node_names = align_methods.split("+")
    align_functions = {
        "agent_align": Dcheck.double_check_agent_align,
        "style_align": Dcheck.double_check_style_align,
        "function_align": Dcheck.double_check_function_align,
    }
    SQL = re.sub(r"(COUNT)(\([^\(\)]*? THEN 1 ELSE 0.*?\))", r"SUM\2", SQL)
    sql_retable = retable(SQL)
    judgment = None
    sql_history = {}
    SQL_correct = SQL
    for node_name in node_names:
        if node_name in align_functions:
            if node_name == "agent_align":
                SQL, judgment = align_functions[node_name](sql_retable, L_values, values, SQL, question, new_db_info, db_col_keys, hint)
            elif node_name == "style_align":
                SQL, judgment = align_functions[node_name](SQL, question, db_col_keys, sql_retable)
            elif node_name == "function_align":
                SQL, judgment = align_functions[node_name](SQL, question)
            sql_history[node_name] = SQL
    align_SQL = SQL
    can_ex = True
    nocse = True
    ans = set()
    time_cost = 10000000

    try:
        # NOTE: func_timeout uses signals which only work in the main thread.
        # Since process_sql runs in ThreadPoolExecutor, call correct_sql directly
        # (it already has a bounded while loop with max iterations).
        SQL, nocse = Dcheck.correct_sql(SQL, question, new_db_info, hint, key_col_des, tmp_prompt, db_col, foreign_set, L_values)
    except Exception as e:
        print(f"correct_sql error: {e}")
        can_ex = False

    if can_ex:
        ans, time_cost = get_sql_ans(SQL)
        align_ans = None
        correct_ans = None
    return sql_history, SQL, ans, nocse, time_cost, align_SQL, align_ans, SQL_correct, correct_ans


def muti_process_sql(Dcheck, SQLs, L_values, values, question, new_db_info, hint, key_col_des, tmp_prompt, db_col, foreign_set, align_methods, n):
    vote = []
    none_case = False
    db_col_keys = db_col.keys()

    # Do NOT use `with ThreadPoolExecutor(...)` — its __exit__ calls
    # shutdown(wait=True), which blocks forever when a worker is stuck on
    # I/O (LLM call or DB query).  Manage the pool manually.
    executor = ThreadPoolExecutor(max_workers=n)
    try:
        future_to_sql = {
            executor.submit(
                process_sql,
                Dcheck,
                SQL,
                L_values,
                values,
                question,
                new_db_info,
                db_col_keys,
                hint,
                key_col_des,
                tmp_prompt,
                db_col,
                foreign_set,
                align_methods,
            ): (SQLs[SQL], SQL)
            for SQL in SQLs
        }
        _candidate_timeout = int(os.environ.get("CANDIDATE_TIMEOUT", "60"))
        _overall_timeout = int(os.environ.get("CANDIDATES_OVERALL_TIMEOUT", "90"))
        time_cost = 10000000
        for future in as_completed(future_to_sql, timeout=_overall_timeout):
            count, tmp_SQL = future_to_sql[future]
            try:
                sql_history, SQL, ans, none_c, time_cost, align_SQL, align_ans, SQL_correct, correct_ans = future.result(timeout=_candidate_timeout)
                vote.append({
                    "sql_history": sql_history,
                    "sql": SQL,
                    "answer": ans,
                    "count": count,
                    "time_cost": time_cost,
                    "align_sql": align_SQL,
                    "align_ans": align_ans,
                    "correct_sql": SQL_correct,
                    "correct_ans": correct_ans,
                })
                none_case = none_case or none_c
            except FunctionTimedOut:
                print(f"Error: Processing SQL timeout for SQL count {count}")
                vote.append({
                    "sql_history": tmp_SQL,
                    "sql": tmp_SQL,
                    "answer": [],
                    "count": 1,
                    "time_cost": time_cost,
                    "align_sql": tmp_SQL,
                    "correct_sql": tmp_SQL,
                    "align_ans": [],
                    "correct_ans": [],
                })
                none_case = True
            except Exception as e:
                print(f"Error processing SQL: {e}")
                vote.append({
                    "sql_history": tmp_SQL,
                    "sql": tmp_SQL,
                    "answer": [],
                    "count": 1,
                    "time_cost": time_cost,
                    "align_sql": tmp_SQL,
                    "correct_sql": tmp_SQL,
                    "align_ans": [],
                    "correct_ans": [],
                })
                none_case = True
    except TimeoutError:
        print(f"Error: muti_process_sql overall timeout ({_overall_timeout}s) – returning partial results")
        none_case = True
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return vote, none_case
