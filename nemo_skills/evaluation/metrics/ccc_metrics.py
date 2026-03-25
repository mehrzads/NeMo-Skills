# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics


class CCCMetrics(BaseMetrics):
    def __init__(self, **kwargs):
        super().__init__()
        self.reset()

    def reset(self):
        super().reset()
        self.predictions_by_problem = defaultdict(list)

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions)
        if predictions:
            problem_id = predictions[0].get("problem_id", predictions[0]["name"])
            self.predictions_by_problem[problem_id].extend(predictions)

    def _get_score_dict(self, submission):
        subtask = submission.get("subtask")
        subtask_result = submission.get("test_case_results", {}).get(subtask, {})
        score = float(subtask_result.get("score", 0.0))
        max_score = float(submission.get("subtask_score", 0.0))
        normalized = score / max_score if max_score > 0 else 0.0
        return {"correct": 1 if max_score > 0 and score >= max_score else 0, "score": normalized}

    def _aggregate_row_group(self, submissions, mode: str, subtask_name: str, declared_max_score: float | None = None):
        scores = []
        sample_passed = []
        sample_total = []
        secret_passed = []
        secret_total = []
        compile_successes = []
        compile_attempts = []
        max_score = 0.0
        for submission in submissions[: self.max_k]:
            subtask_result = submission.get("test_case_results", {}).get(subtask_name, {})
            score = float(subtask_result.get("score", 0.0))
            outputs = subtask_result.get("outputs", [])
            # Use the declared subtask score as the scoring maximum.
            # Counting outputs only works for per-test ICPC tasks and breaks weighted IOI subtasks.
            if declared_max_score is not None:
                output_max_score = float(declared_max_score)
            elif "max_score" in subtask_result:
                output_max_score = float(subtask_result.get("max_score", 0.0))
            else:
                raise ValueError(
                    f"Max score is undefined for subtask '{subtask_name}'. "
                    "Expected declared subtask_score or test_case_results[*].max_score."
                )
            max_score = max(max_score, output_max_score)
            sample_tests = [out for out in outputs if out.get("test_group") == "sample"]
            secret_tests = [out for out in outputs if out.get("test_group") == "secret"]
            scores.append(score)
            sample_passed.append(sum(1 for out in sample_tests if float(out.get("score", 0.0)) > 0.0))
            sample_total.append(len(sample_tests))
            secret_passed.append(sum(1 for out in secret_tests if float(out.get("score", 0.0)) > 0.0))
            secret_total.append(len(secret_tests))
            compile_successes.append(sum(1 for out in outputs if out.get("compile_success")))
            compile_attempts.append(len(outputs))

        if not scores:
            return {
                "score": 0.0,
                "max_score": max_score,
                "sample_tests_passed": 0,
                "sample_tests_total": 0,
                "secret_tests_passed": 0,
                "secret_tests_total": 0,
                "compile_successes": 0,
                "compile_attempts": 0,
                "compile_success_rate": 0.0,
                "submission_stats": {
                    "num_submissions": 0,
                    "min_score": 0.0,
                    "avg_score": 0.0,
                    "max_score": 0.0,
                    "nonzero_submissions": 0,
                    "full_score_submissions": 0,
                    "avg_sample_tests_passed": 0.0,
                    "avg_secret_tests_passed": 0.0,
                    "avg_compile_successes": 0.0,
                    "avg_compile_attempts": 0.0,
                    "avg_compile_success_rate": 0.0,
                },
            }

        agg_score = max(scores) if mode == "best" else sum(scores) / len(scores)
        agg_sample_passed = max(sample_passed) if mode == "best" else sum(sample_passed) / len(sample_passed)
        agg_secret_passed = max(secret_passed) if mode == "best" else sum(secret_passed) / len(secret_passed)
        agg_compile_successes = max(compile_successes) if mode == "best" else sum(compile_successes) / len(compile_successes)
        agg_compile_attempts = max(compile_attempts) if mode == "best" else sum(compile_attempts) / len(compile_attempts)
        agg_compile_success_rate = (100.0 * agg_compile_successes / agg_compile_attempts) if agg_compile_attempts else 0.0

        return {
            "score": agg_score,
            "max_score": max_score,
            "sample_tests_passed": agg_sample_passed,
            "sample_tests_total": max(sample_total) if sample_total else 0,
            "secret_tests_passed": agg_secret_passed,
            "secret_tests_total": max(secret_total) if secret_total else 0,
            "compile_successes": agg_compile_successes,
            "compile_attempts": agg_compile_attempts,
            "compile_success_rate": agg_compile_success_rate,
            "submission_stats": {
                "num_submissions": len(scores),
                "min_score": min(scores),
                "avg_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "nonzero_submissions": sum(1 for s in scores if s > 0.0),
                "full_score_submissions": sum(1 for s in scores if max_score > 0 and s >= max_score),
                "avg_sample_tests_passed": sum(sample_passed) / len(sample_passed),
                "avg_secret_tests_passed": sum(secret_passed) / len(secret_passed),
                "avg_compile_successes": sum(compile_successes) / len(compile_successes),
                "avg_compile_attempts": sum(compile_attempts) / len(compile_attempts),
                "avg_compile_success_rate": (
                    100.0 * sum(compile_successes) / sum(compile_attempts) if sum(compile_attempts) else 0.0
                ),
            },
        }

    def _build_problem_reports(self, mode: str):
        total_score = 0.0
        total_max_score = 0.0
        per_problem_report = []
        total_sample_passed = 0.0
        total_sample_tests = 0
        total_secret_passed = 0.0
        total_secret_tests = 0
        total_submission_rows = 0
        total_nonzero_submission_rows = 0
        total_full_score_submission_rows = 0
        total_compile_successes = 0.0
        total_compile_attempts = 0.0
        problems_fully_solved = 0
        problems_sample_fully_solved = 0

        for problem_id, submissions in sorted(self.predictions_by_problem.items()):
            problem_name = submissions[0]["name"]
            grouped_rows = defaultdict(list)
            for submission in submissions:
                # Row id values are reused across problems and can be ambiguous in some dumps.
                # Include subtask label for stability while still grouping the same row across rs files.
                row_key = (submission.get("id"), submission.get("subtask"))
                grouped_rows[row_key].append(submission)
            declared_max_by_subtask = {}
            for submission in submissions:
                st = submission.get("subtask")
                if st is None:
                    continue
                declared = float(submission.get("subtask_score", 0.0))
                declared_max_by_subtask[st] = max(declared_max_by_subtask.get(st, 0.0), declared)
            all_subtasks = set()
            for submission in submissions:
                all_subtasks.update(submission.get("test_case_results", {}).keys())
            missing_max_subtasks = sorted(st for st in all_subtasks if st not in declared_max_by_subtask)
            if missing_max_subtasks:
                raise ValueError(
                    f"Problem '{problem_id}' has subtasks without defined max score: {missing_max_subtasks}. "
                    "Each subtask must have a declared subtask_score."
                )
            subtasks = {}
            for row_submissions in grouped_rows.values():
                # Each row has full subtask results in test_case_results.
                # Score this row against all subtasks, then keep the best row per subtask.
                row_subtasks = row_submissions[0].get("test_case_results", {}).keys()
                for subtask in row_subtasks:
                    row_report = self._aggregate_row_group(
                        row_submissions,
                        mode,
                        subtask_name=subtask,
                        declared_max_score=declared_max_by_subtask.get(subtask),
                    )
                    prev = subtasks.get(subtask)
                    if prev is None or row_report["score"] > prev["score"]:
                        subtasks[subtask] = row_report

            problem_score = 0.0
            problem_max_score = 0.0
            correct_subtasks = 0
            problem_sample_passed = 0.0
            problem_sample_tests = 0
            problem_secret_passed = 0.0
            problem_secret_tests = 0
            problem_compile_successes = 0.0
            problem_compile_attempts = 0.0
            problem_submission_rows = sum(s["submission_stats"]["num_submissions"] for s in subtasks.values())
            problem_nonzero_submission_rows = sum(s["submission_stats"]["nonzero_submissions"] for s in subtasks.values())
            problem_full_score_submission_rows = sum(
                s["submission_stats"]["full_score_submissions"] for s in subtasks.values()
            )

            for subtask_report in subtasks.values():
                correct = subtask_report["score"] >= subtask_report["max_score"] if subtask_report["max_score"] > 0 else False
                subtask_report["correct"] = correct
                problem_score += subtask_report["score"]
                problem_max_score += subtask_report["max_score"]
                correct_subtasks += int(correct)
                problem_sample_passed += subtask_report["sample_tests_passed"]
                problem_sample_tests += subtask_report["sample_tests_total"]
                problem_secret_passed += subtask_report["secret_tests_passed"]
                problem_secret_tests += subtask_report["secret_tests_total"]
                problem_compile_successes += subtask_report["compile_successes"]
                problem_compile_attempts += subtask_report["compile_attempts"]

            total_score += problem_score
            total_max_score += problem_max_score
            total_sample_passed += problem_sample_passed
            total_sample_tests += problem_sample_tests
            total_secret_passed += problem_secret_passed
            total_secret_tests += problem_secret_tests
            total_compile_successes += problem_compile_successes
            total_compile_attempts += problem_compile_attempts
            total_submission_rows += problem_submission_rows
            total_nonzero_submission_rows += problem_nonzero_submission_rows
            total_full_score_submission_rows += problem_full_score_submission_rows
            problems_fully_solved += int(problem_score >= problem_max_score and problem_max_score > 0)

            problem_report = {
                "problem_id": problem_id,
                "name": problem_name,
                "score": problem_score,
                "max_score": problem_max_score,
                "correct_subtasks": correct_subtasks,
                "num_subtasks": len(subtasks),
                "num_submission_rows": problem_submission_rows,
                "nonzero_submission_rows": problem_nonzero_submission_rows,
                "full_score_submission_rows": problem_full_score_submission_rows,
                "compile_successes": problem_compile_successes,
                "compile_attempts": problem_compile_attempts,
                "compile_success_rate": (
                    100.0 * problem_compile_successes / problem_compile_attempts if problem_compile_attempts else 0.0
                ),
                "subtasks": subtasks,
            }
            if problem_sample_tests or problem_secret_tests:
                problem_report["sample_tests_passed"] = problem_sample_passed
                problem_report["sample_tests_total"] = problem_sample_tests
                problem_report["secret_tests_passed"] = problem_secret_passed
                problem_report["secret_tests_total"] = problem_secret_tests
                problem_report["sample_fully_solved"] = problem_sample_tests > 0 and problem_sample_passed >= problem_sample_tests
                problems_sample_fully_solved += int(problem_report["sample_fully_solved"])
            per_problem_report.append(problem_report)

        return {
            "total_score": total_score,
            "total_max_score": total_max_score,
            "problems": per_problem_report,
            "num_problems": len(per_problem_report),
            "num_subtasks": sum(problem["num_subtasks"] for problem in per_problem_report),
            "num_submission_rows": total_submission_rows,
            "nonzero_submission_rows": total_nonzero_submission_rows,
            "full_score_submission_rows": total_full_score_submission_rows,
            "compile_successes": total_compile_successes,
            "compile_attempts": total_compile_attempts,
            "compile_success_rate": (100.0 * total_compile_successes / total_compile_attempts) if total_compile_attempts else 0.0,
            "problems_fully_solved": problems_fully_solved,
            "problems_sample_fully_solved": problems_sample_fully_solved,
            "problem_solve_rate": (100.0 * problems_fully_solved / len(per_problem_report)) if per_problem_report else 0.0,
            "sample_tests_passed": total_sample_passed,
            "sample_tests_total": total_sample_tests,
            "secret_tests_passed": total_secret_passed,
            "secret_tests_total": total_secret_tests,
        }

    def get_metrics(self):
        metrics_dict = super().get_metrics()
        keep_keys = [f"pass@1[avg-of-{self.max_k}]", f"pass@{self.max_k}"]
        metrics_dict = {k: v for k, v in metrics_dict.items() if k in keep_keys}

        avg_report = self._build_problem_reports(mode="avg")
        best_report = self._build_problem_reports(mode="best")
        report_by_key = {
            f"pass@1[avg-of-{self.max_k}]": avg_report,
            f"pass@{self.max_k}": best_report,
        }

        for key, metric in metrics_dict.items():
            report = report_by_key[key]
            total_score = int(report["total_score"]) if float(report["total_score"]).is_integer() else report["total_score"]
            total_max_score = (
                int(report["total_max_score"])
                if float(report["total_max_score"]).is_integer()
                else report["total_max_score"]
            )
            problem_table = []
            for problem in report["problems"]:
                ordered_subtasks = list(problem["subtasks"].items())
                score_array = [
                    (
                        int(subtask_info["score"])
                        if float(subtask_info["score"]).is_integer()
                        else subtask_info["score"]
                    )
                    for _, subtask_info in ordered_subtasks
                ]
                problem_table.append(
                    {
                        "problem_id": problem["problem_id"],
                        "name": problem["name"],
                        "status": "passed" if problem["score"] >= problem["max_score"] and problem["max_score"] > 0 else "failed",
                        "score": int(problem["score"]) if float(problem["score"]).is_integer() else problem["score"],
                        "max_score": int(problem["max_score"]) if float(problem["max_score"]).is_integer() else problem["max_score"],
                        "compile_success_rate": problem["compile_success_rate"],
                        "score_array": score_array,
                    }
                )
            summary = {
                "total_score": total_score,
                "total_max_score": total_max_score,
                "problems_fully_solved": report["problems_fully_solved"],
                "problems_sample_fully_solved": report["problems_sample_fully_solved"],
                "problem_solve_rate": report["problem_solve_rate"],
                "num_problems": report["num_problems"],
                "num_subtasks": report["num_subtasks"],
                "num_submission_rows": report["num_submission_rows"],
                "nonzero_submission_rows": report["nonzero_submission_rows"],
                "full_score_submission_rows": report["full_score_submission_rows"],
                "compile_successes": report["compile_successes"],
                "compile_attempts": report["compile_attempts"],
                "compile_success_rate": report["compile_success_rate"],
            }
            if report["sample_tests_total"] or report["secret_tests_total"]:
                summary["sample_tests_passed"] = report["sample_tests_passed"]
                summary["sample_tests_total"] = report["sample_tests_total"]
                summary["secret_tests_passed"] = report["secret_tests_passed"]
                summary["secret_tests_total"] = report["secret_tests_total"]
            summary["problem_table"] = problem_table

            metric.clear()
            metric["summary"] = summary
            metric["total_score"] = total_score
            metric["total_max_score"] = total_max_score
            metric["problems_fully_solved"] = report["problems_fully_solved"]
            metric["problems_sample_fully_solved"] = report["problems_sample_fully_solved"]
            metric["problem_solve_rate"] = report["problem_solve_rate"]
            metric["num_problems"] = report["num_problems"]
            metric["num_subtasks"] = report["num_subtasks"]
            metric["num_submission_rows"] = report["num_submission_rows"]
            metric["nonzero_submission_rows"] = report["nonzero_submission_rows"]
            metric["full_score_submission_rows"] = report["full_score_submission_rows"]
            metric["compile_successes"] = report["compile_successes"]
            metric["compile_attempts"] = report["compile_attempts"]
            metric["compile_success_rate"] = report["compile_success_rate"]
            if report["sample_tests_total"] or report["secret_tests_total"]:
                metric["sample_tests_passed"] = report["sample_tests_passed"]
                metric["sample_tests_total"] = report["sample_tests_total"]
                metric["secret_tests_passed"] = report["secret_tests_passed"]
                metric["secret_tests_total"] = report["secret_tests_total"]
            metric["problems"] = report["problems"]
        return metrics_dict

    def evaluations_to_print(self):
        return [f"pass@1[avg-of-{self.max_k}]", f"pass@{self.max_k}"]
