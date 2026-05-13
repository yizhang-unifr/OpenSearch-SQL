import logging
from typing import Any, Dict, List
from pipeline.utils import node_decorator,get_last_node_result
from runner.check_and_correct import sql_raw_parse


# ── landcover-join guard ──────────────────────────────────────────────────────

_LC_REQUIRED_KEYS = (
    "landcover", "land cover", "land use", "land-cover",
    "forest", "urban", "lake", "lakes", "river", "water",
    "cropland", "agricultural", "wetland", "grassland",
    "bare rock", "beach", "dune", "glacier",
)


def _requires_landcover_join(question: str) -> bool:
    """True if the question requires a JOIN to landcover_type to be correct."""
    q = question.lower()
    return any(k in q for k in _LC_REQUIRED_KEYS)


def _has_landcover_join(sql: str) -> bool:
    """True if the SQL accesses landcover data via ranks or landcover_type join."""
    s = sql.lower()
    return "ranks[" in s or "generate_subscripts" in s or "join landcover_type" in s


def _filter_to_landcover_votes(vote_list: List[Dict], question: str) -> List[Dict]:
    """Return only vote items with a landcover join; fall back to full list if none qualify."""
    if not _requires_landcover_join(question):
        return vote_list
    qualified = [v for v in vote_list if _has_landcover_join(v.get("sql", ""))]
    if qualified:
        logging.info(
            "vote: landcover filter kept %d/%d candidates for question=%r",
            len(qualified), len(vote_list), question[:60],
        )
        return qualified
    logging.warning(
        "vote: landcover filter found no qualifying candidates for question=%r; using all",
        question[:60],
    )
    return vote_list

def vote_single(vote_all,mod="answer",SQLs=[]):
    vote_M = [0] * len(vote_all)
    same_ans = {}
    for i, item in enumerate(vote_all):
        sql = item["sql"]
        ans = item[mod]  # ans 现在是一个列表
        count = item["count"]
        time_cost = item["time_cost"]
        
        vote_M[i] += count
        
        if same_ans.get(i, -1) == -1:  #父节点
            same_ans[i] = i
        if not ans:
            vote_M[i] = 0
            continue
        for j in range(i + 1, len(vote_all)):
            other_ans = vote_all[j][mod]  # 其他项的答案
            # 使用集合比较两个答案列表
            if ans ==  other_ans:
                same_ans[j] = same_ans[i]
                vote_M[i] += vote_all[j]["count"]
                vote_M[j] += vote_all[i]["count"]
            # if ans == vote_all[j][1]:
            #     same_ans[j] = same_ans[i]
            #     vote_M[i] += vote_all[j][2]
            #     vote_M[j] += vote_all[i][2]

    maxm = max(vote_M)
    # print(maxm,ans)
    min_t = 1000000
    sql_0=sql_raw_parse(SQLs[0], False)[0] 
    ans = sql_0 
    # ans=
    # print(vote_M)
    print("_______vote same best", same_ans)
    # result_all = {}
    for i, x in enumerate(vote_M):
        if maxm == x:
            if vote_all[i]["time_cost"]< min_t:
                ans = vote_all[i]['sql']
                min_t = vote_all[i]["time_cost"]
    return ans,maxm,min_t,vote_M
      


@node_decorator(check_schema_status=False)
def vote(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:

    vote = get_last_node_result(execution_history, "align_correct")["vote"]
    SQLs=get_last_node_result(execution_history, "candidate_generate")["SQL"]# 兜底
    if not vote:
        logging.error(
            "vote node received empty vote list; question_id=%s candidate_count=%s candidates=%s",
            getattr(task, "question_id", "unknown"),
            len(SQLs) if isinstance(SQLs, list) else 0,
            SQLs,
        )
        raise ValueError("align_correct returned empty vote list")

    question = getattr(task, "question", "") or ""
    vote = _filter_to_landcover_votes(vote, question)

    ans_correct,maxm,min_t,vote_M=vote_single(vote,"correct_ans",SQLs)
    # align_ans,maxm,min_t,vote_M=vote_single(vote,"align_ans",SQLs)
    ans,maxm,min_t,vote_M=vote_single(vote,"answer",SQLs)
    print(ans)
    print("_______")
    if maxm == 0:
        nonecase = True
    else:
        nonecase = False
    print(
        f"******votes:{vote_M} max vote: {maxm}, min_t:{min_t}, SQL vote is: {ans}"
    )
  
    response = {
        "SQL":ans,
        "SQL_correct_vote":ans_correct,
        # "SQL_align_vote":align_ans,
        "nonecase": nonecase
    }

    return response



