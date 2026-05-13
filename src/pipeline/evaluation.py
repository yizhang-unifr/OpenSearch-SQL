import logging
from typing import Any, Dict

from pipeline.utils import get_last_node_result, node_decorator
from runner.check_and_correct import sql_raw_parse
from runner.database_manager import DatabaseManager
from runner.execution import execute_sql
from runner.logger import Logger

@node_decorator(check_schema_status=False)
def evaluation(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the predicted SQL queries against the ground truth SQL query.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, Any]): The current tentative schema.
        execution_history (Dict[str, Any]): The history of executions.

    Returns:
        Dict[str, Any]: A dictionary containing the evaluation results.
    """
    # logging.info("Starting evaluation")

    ground_truth_sql = task.SQL

    # Execute the gold SQL once so every node's evaluation can include the reference result.
    gold_result = None
    try:
        gold_result = list(execute_sql(ground_truth_sql, fetch="all", timeout_s=180))
    except Exception as _e:
        gold_result = f"gold_exec_error: {_e}"

    to_evaluate = {
        "candidate_generate": get_last_node_result(execution_history, "candidate_generate"), 
        "align_correct": get_last_node_result(execution_history, "align_correct"),#align+纠错 
        # "align": get_last_node_result(execution_history, "vote"), #未纠错
        # "correct":get_last_node_result(execution_history, "vote"),
        "vote": get_last_node_result(execution_history, "vote")
    }
    result = {}
    for evaluation_for, node_result in to_evaluate.items():
        predicted_sql = "--"
        evaluation_result = {}

        try:
            if node_result["status"] == "success":

                if evaluation_for =="align" :
                    predicted_sql=node_result['SQL_align_vote']
                elif evaluation_for =="correct" :
                    predicted_sql=node_result["SQL_correct_vote"]
                elif evaluation_for =="align_correct":
                    vote_all=node_result['vote']
                    predicted_sql=vote_all[0]['sql']
                elif evaluation_for=="candidate_generate":
                    candidate_all=node_result['SQL']
                    predicted_sql=sql_raw_parse(candidate_all[0], False)[0]
                elif evaluation_for=="vote":
                    predicted_sql = node_result["SQL"]
                response = DatabaseManager().compare_sqls(
                    predicted_sql=predicted_sql,
                    ground_truth_sql=ground_truth_sql,
                    meta_time_out=180
                )

                evaluation_result.update({
                    "exec_res": response["exec_res"],
                    "exec_err": response["exec_err"],
                    "ves": response.get("ves", 0.0),
                })
            else:
                evaluation_result.update({
                    "exec_res": "generation error",
                    "exec_err": node_result["error"],
                    "ves": 0.0,
                })
        except Exception as e:
            Logger().log(
                f"Node 'evaluate_sql': {task.db_id}_{task.question_id}\n{type(e)}: {e}\n",
                "error",
            )
            evaluation_result.update({
                "exec_res": "error",
                "exec_err": str(e),
                "ves": 0.0,
            })

        # Also execute the predicted SQL and include the raw result for inspection.
        predicted_result = None
        try:
            predicted_result = list(execute_sql(predicted_sql, fetch="all", timeout_s=30))
        except Exception as _pe:
            predicted_result = f"pred_exec_error: {_pe}"

        evaluation_result.update({
            "Question": task.raw_question,
            "Evidence": task.evidence,
            "GOLD_SQL": ground_truth_sql,
            "GOLD_RESULT": gold_result,
            "PREDICTED_SQL": predicted_sql,
            "PREDICTED_RESULT": predicted_result,
        })
        result[evaluation_for] = evaluation_result

    logging.info("Evaluation completed successfully")
    return result
