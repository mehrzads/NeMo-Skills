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

    def get_metrics(self):
        total_score = 0.0
        total_max_score = 0.0
        per_problem_report = []
        total_sample_passed = 0
        total_sample_tests = 0
        total_secret_passed = 0
        total_secret_tests = 0
        total_submission_rows = 0
        total_nonzero_submission_rows = 0
        total_full_score_submission_rows = 0

        for problem_id, submissions in sorted(self.predictions_by_problem.items()):
            problem_name = submissions[0]["name"]
            subtasks = {}
            problem_submission_rows = 0
            problem_nonzero_submission_rows = 0
            problem_full_score_submission_rows = 0
            for submission in submissions:
                subtask = submission["subtask"]
                subtask_result = submission.get("test_case_results", {}).get(subtask, {})
                score = float(subtask_result.get("score", 0.0))
                max_score = float(submission.get("subtask_score", 0.0))
                outputs = subtask_result.get("outputs", [])
                sample_tests = [out for out in outputs if out.get("test_group") == "sample"]
                secret_tests = [out for out in outputs if out.get("test_group") == "secret"]
                sample_passed = sum(1 for out in sample_tests if float(out.get("score", 0.0)) > 0.0)
                secret_passed = sum(1 for out in secret_tests if float(out.get("score", 0.0)) > 0.0)
                problem_submission_rows += 1
                problem_nonzero_submission_rows += int(score > 0.0)
                problem_full_score_submission_rows += int(max_score > 0 and score >= max_score)

                if subtask not in subtasks:
                    subtasks[subtask] = {
                        "score": score,
                        "max_score": max_score,
                        "sample_tests_passed": sample_passed,
                        "sample_tests_total": len(sample_tests),
                        "secret_tests_passed": secret_passed,
                        "secret_tests_total": len(secret_tests),
                        "submission_stats": {
                            "num_submissions": 1,
                            "min_score": score,
                            "avg_score": score,
                            "max_score": score,
                            "nonzero_submissions": int(score > 0.0),
                            "full_score_submissions": int(max_score > 0 and score >= max_score),
                            "avg_sample_tests_passed": float(sample_passed),
                            "avg_secret_tests_passed": float(secret_passed),
                        },
                    }
                else:
                    subtasks[subtask]["score"] = max(subtasks[subtask]["score"], score)
                    subtasks[subtask]["max_score"] = max(subtasks[subtask]["max_score"], max_score)
                    subtasks[subtask]["sample_tests_passed"] = max(
                        subtasks[subtask]["sample_tests_passed"], sample_passed
                    )
                    subtasks[subtask]["sample_tests_total"] = max(
                        subtasks[subtask]["sample_tests_total"], len(sample_tests)
                    )
                    subtasks[subtask]["secret_tests_passed"] = max(
                        subtasks[subtask]["secret_tests_passed"], secret_passed
                    )
                    subtasks[subtask]["secret_tests_total"] = max(
                        subtasks[subtask]["secret_tests_total"], len(secret_tests)
                    )
                    stats = subtasks[subtask]["submission_stats"]
                    count = stats["num_submissions"] + 1
                    stats["min_score"] = min(stats["min_score"], score)
                    stats["avg_score"] = (stats["avg_score"] * stats["num_submissions"] + score) / count
                    stats["max_score"] = max(stats["max_score"], score)
                    stats["nonzero_submissions"] += int(score > 0.0)
                    stats["full_score_submissions"] += int(max_score > 0 and score >= max_score)
                    stats["avg_sample_tests_passed"] = (
                        stats["avg_sample_tests_passed"] * stats["num_submissions"] + sample_passed
                    ) / count
                    stats["avg_secret_tests_passed"] = (
                        stats["avg_secret_tests_passed"] * stats["num_submissions"] + secret_passed
                    ) / count
                    stats["num_submissions"] = count

            problem_score = 0.0
            problem_max_score = 0.0
            correct_subtasks = 0
            problem_sample_passed = 0
            problem_sample_tests = 0
            problem_secret_passed = 0
            problem_secret_tests = 0
            for subtask, subtask_report in subtasks.items():
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
            per_problem_report.append(
                problem_report
            )

        metrics_dict = super().get_metrics()
        for metric in metrics_dict.values():
            metric["total_score"] = int(total_score) if float(total_score).is_integer() else total_score
            metric["total_max_score"] = int(total_max_score) if float(total_max_score).is_integer() else total_max_score
            metric["problems"] = per_problem_report
            metric["num_problems"] = len(per_problem_report)
            metric["num_submission_rows"] = total_submission_rows
            metric["nonzero_submission_rows"] = total_nonzero_submission_rows
            metric["full_score_submission_rows"] = total_full_score_submission_rows
            if total_sample_tests or total_secret_tests:
                metric["sample_tests_passed"] = total_sample_passed
                metric["sample_tests_total"] = total_sample_tests
                metric["secret_tests_passed"] = total_secret_passed
                metric["secret_tests_total"] = total_secret_tests
        return metrics_dict

    def evaluations_to_print(self):
        return [f"pass@{self.max_k}"]
