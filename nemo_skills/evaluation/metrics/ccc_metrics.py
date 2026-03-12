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

    def _aggregate_row_group(self, submissions, mode: str):
        scores = []
        sample_passed = []
        sample_total = []
        secret_passed = []
        secret_total = []
        max_score = 0.0
        for submission in submissions[: self.max_k]:
            subtask = submission["subtask"]
            subtask_result = submission.get("test_case_results", {}).get(subtask, {})
            score = float(subtask_result.get("score", 0.0))
            max_score = max(max_score, float(submission.get("subtask_score", 0.0)))
            outputs = subtask_result.get("outputs", [])
            sample_tests = [out for out in outputs if out.get("test_group") == "sample"]
            secret_tests = [out for out in outputs if out.get("test_group") == "secret"]
            scores.append(score)
            sample_passed.append(sum(1 for out in sample_tests if float(out.get("score", 0.0)) > 0.0))
            sample_total.append(len(sample_tests))
            secret_passed.append(sum(1 for out in secret_tests if float(out.get("score", 0.0)) > 0.0))
            secret_total.append(len(secret_tests))

        if not scores:
            return {
                "score": 0.0,
                "max_score": max_score,
                "sample_tests_passed": 0,
                "sample_tests_total": 0,
                "secret_tests_passed": 0,
                "secret_tests_total": 0,
                "submission_stats": {
                    "num_submissions": 0,
                    "min_score": 0.0,
                    "avg_score": 0.0,
                    "max_score": 0.0,
                    "nonzero_submissions": 0,
                    "full_score_submissions": 0,
                    "avg_sample_tests_passed": 0.0,
                    "avg_secret_tests_passed": 0.0,
                },
            }

        agg_score = max(scores) if mode == "best" else sum(scores) / len(scores)
        agg_sample_passed = max(sample_passed) if mode == "best" else sum(sample_passed) / len(sample_passed)
        agg_secret_passed = max(secret_passed) if mode == "best" else sum(secret_passed) / len(secret_passed)

        return {
            "score": agg_score,
            "max_score": max_score,
            "sample_tests_passed": agg_sample_passed,
            "sample_tests_total": max(sample_total) if sample_total else 0,
            "secret_tests_passed": agg_secret_passed,
            "secret_tests_total": max(secret_total) if secret_total else 0,
            "submission_stats": {
                "num_submissions": len(scores),
                "min_score": min(scores),
                "avg_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "nonzero_submissions": sum(1 for s in scores if s > 0.0),
                "full_score_submissions": sum(1 for s in scores if max_score > 0 and s >= max_score),
                "avg_sample_tests_passed": sum(sample_passed) / len(sample_passed),
                "avg_secret_tests_passed": sum(secret_passed) / len(secret_passed),
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
        problems_fully_solved = 0
        problems_sample_fully_solved = 0

        for problem_id, submissions in sorted(self.predictions_by_problem.items()):
            problem_name = submissions[0]["name"]
            grouped_rows = defaultdict(list)
            for submission in submissions:
                row_id = submission.get("id", f'{problem_id}:{submission["subtask"]}')
                grouped_rows[row_id].append(submission)

            subtasks = {}
            for row_submissions in grouped_rows.values():
                subtask = row_submissions[0]["subtask"]
                subtasks[subtask] = self._aggregate_row_group(row_submissions, mode)

            problem_score = 0.0
            problem_max_score = 0.0
            correct_subtasks = 0
            problem_sample_passed = 0.0
            problem_sample_tests = 0
            problem_secret_passed = 0.0
            problem_secret_tests = 0
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

            total_score += problem_score
            total_max_score += problem_max_score
            total_sample_passed += problem_sample_passed
            total_sample_tests += problem_sample_tests
            total_secret_passed += problem_secret_passed
            total_secret_tests += problem_secret_tests
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
            "num_submission_rows": total_submission_rows,
            "nonzero_submission_rows": total_nonzero_submission_rows,
            "full_score_submission_rows": total_full_score_submission_rows,
            "problems_fully_solved": problems_fully_solved,
            "problems_sample_fully_solved": problems_sample_fully_solved,
            "problems_total": len(per_problem_report),
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
            summary = {
                "total_score": total_score,
                "total_max_score": total_max_score,
                "problems_fully_solved": report["problems_fully_solved"],
                "problems_sample_fully_solved": report["problems_sample_fully_solved"],
                "problems_total": report["problems_total"],
                "problem_solve_rate": report["problem_solve_rate"],
                "num_problems": report["num_problems"],
                "num_submission_rows": report["num_submission_rows"],
                "nonzero_submission_rows": report["nonzero_submission_rows"],
                "full_score_submission_rows": report["full_score_submission_rows"],
            }
            if report["sample_tests_total"] or report["secret_tests_total"]:
                summary["sample_tests_passed"] = report["sample_tests_passed"]
                summary["sample_tests_total"] = report["sample_tests_total"]
                summary["secret_tests_passed"] = report["secret_tests_passed"]
                summary["secret_tests_total"] = report["secret_tests_total"]

            metric.clear()
            metric["summary"] = summary
            metric["total_score"] = total_score
            metric["total_max_score"] = total_max_score
            metric["problems_fully_solved"] = report["problems_fully_solved"]
            metric["problems_sample_fully_solved"] = report["problems_sample_fully_solved"]
            metric["problems_total"] = report["problems_total"]
            metric["problem_solve_rate"] = report["problem_solve_rate"]
            metric["num_problems"] = report["num_problems"]
            metric["num_submission_rows"] = report["num_submission_rows"]
            metric["nonzero_submission_rows"] = report["nonzero_submission_rows"]
            metric["full_score_submission_rows"] = report["full_score_submission_rows"]
            if report["sample_tests_total"] or report["secret_tests_total"]:
                metric["sample_tests_passed"] = report["sample_tests_passed"]
                metric["sample_tests_total"] = report["sample_tests_total"]
                metric["secret_tests_passed"] = report["secret_tests_passed"]
                metric["secret_tests_total"] = report["secret_tests_total"]
            metric["problems"] = report["problems"]
        return metrics_dict

    def evaluations_to_print(self):
        return [f"pass@1[avg-of-{self.max_k}]", f"pass@{self.max_k}"]
