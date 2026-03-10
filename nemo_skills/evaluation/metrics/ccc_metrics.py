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
        self.problem_scores = {}
        self.per_problem_subtask_scores = {}

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions)
        if predictions:
            self.predictions_by_problem[predictions[0]["name"]].extend(predictions)

    def _is_icpc_submission(self, submission: dict) -> bool:
        tcr = submission.get("test_case_results", {})
        return isinstance(tcr, dict) and "outputs" in tcr and "sample_score" in tcr and "score" in tcr

    def _icpc_submission_score(self, submission: dict) -> tuple[float, float]:
        outputs = submission.get("test_case_results", {}).get("outputs", [])
        tests = [o for o in outputs if o.get("test_type") == "test"]
        if not tests:
            tests = [o for o in outputs if o.get("test_type") != "sample"]
        max_score = float(len(tests))
        score = float(sum(1 for o in tests if float(o.get("score", 0.0)) > 0.0))
        return score, max_score

    def _get_score_dict(self, submission):
        if self._is_icpc_submission(submission):
            score, max_score = self._icpc_submission_score(submission)
            normalized = score / max_score if max_score > 0 else 0.0
            return {"correct": 1 if score == max_score and max_score > 0 else 0, "score": normalized}

        subtask = submission.get("subtask")
        subtask_result = submission.get("test_case_results", {}).get(subtask, {})
        score = float(subtask_result.get("score", 0.0))
        max_score = float(submission.get("subtask_score", 0.0))
        normalized = score / max_score if max_score > 0 else 0.0
        return {"correct": 1 if max_score > 0 and score >= max_score else 0, "score": normalized}

    def _get_ioi_problem_score(self, submissions):
        achieved_subtasks = {}
        max_subtasks = {}
        for submission in submissions:
            subtask = submission["subtask"]
            max_subtasks[subtask] = float(submission.get("subtask_score", 0.0))
            for st_name, result in submission.get("test_case_results", {}).items():
                achieved_subtasks[st_name] = max(achieved_subtasks.get(st_name, 0.0), float(result.get("score", 0.0)))
        achieved_total = sum(achieved_subtasks.values())
        max_total = sum(max_subtasks.values())
        return achieved_total, achieved_subtasks, max_total, max_subtasks

    def _get_icpc_problem_score(self, submissions):
        best_score = 0.0
        max_score = 0.0
        for submission in submissions:
            score, cur_max = self._icpc_submission_score(submission)
            best_score = max(best_score, score)
            max_score = max(max_score, cur_max)
        return best_score, {"all_tests": best_score}, max_score, {"all_tests": max_score}

    def get_metrics(self):
        total_score = 0.0
        per_problem_subtask_scores = {}

        for name, submissions in self.predictions_by_problem.items():
            if submissions and self._is_icpc_submission(submissions[0]):
                achieved_total, achieved_subtasks, max_total, max_subtasks = self._get_icpc_problem_score(submissions)
            else:
                achieved_total, achieved_subtasks, max_total, max_subtasks = self._get_ioi_problem_score(submissions)

            self.problem_scores[name] = (achieved_total, achieved_subtasks, max_total, max_subtasks)
            total_score += achieved_total
            per_problem_subtask_scores[name] = {
                "total": {"score": achieved_total, "max_score": max_total},
                "subtasks": {
                    subtask: {"score": achieved_subtasks.get(subtask, 0.0), "max_score": max_subtasks.get(subtask, 0.0)}
                    for subtask in max_subtasks
                },
            }

        metrics_dict = super().get_metrics()
        for m in metrics_dict.values():
            m["total_score"] = int(total_score) if float(total_score).is_integer() else total_score
            m["per_problem_subtask_scores"] = per_problem_subtask_scores
        self.per_problem_subtask_scores = per_problem_subtask_scores
        return metrics_dict

    def evaluations_to_print(self):
        return [f"pass@{self.max_k}"]
