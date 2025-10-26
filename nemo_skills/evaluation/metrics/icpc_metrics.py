# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics
from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage


class ICPCMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, predictions):
        super().update(predictions)
#        self._compute_pass_at_k(predictions)
        if predictions:            
            self.predictions_by_problem[predictions[0]["name"]].extend(predictions)

    def _get_score_dict(self, p):
        return {"correct": all(r["score"] > 0 for r in p["test_case_results"].values())}

    def get_problem_score(self, submissions) -> bool:
        submission = submissions[0]
        if not submission:
            return False
        if submission["test_case_results"]["score"]:
            return True
        else:
            return False

 

    def get_metrics(self):
        total_score = total_round_robin = 0.0
        self.problem_scores = {}
        self.correct_submissions = {}
        self.total_submissions = {}
        for name, submission in self.predictions_by_problem.items():
            if self.correct_submissions.get(name) is None:
                    self.correct_submissions[name] = 0
            if self.total_submissions.get(name) is None:
                self.total_submissions[name] = 0
            if self.problem_scores.get(name) is None:
                self.problem_scores[name] = False
            if self.get_problem_score(submission):                
                self.correct_submissions[name] += 1
                self.problem_scores[name] = True
            self.total_submissions[name] += 1    
        self.print_problem_scores()
        metrics_dict = {}
        for name, scores in self.problem_scores.items():
            metrics_dict[name] = {
                "correct": self.correct_submissions[name],
                "total": self.total_submissions[name]
            }
        return metrics_dict

    def evaluations_to_print(self):
        """Returns all problem names."""
        return list(self.problem_scores.keys())

    def metrics_to_print(self):
        metrics_to_print = {"correct": as_int, "total": as_int}
        return metrics_to_print


    def reset(self):
        super().reset()
        self.predictions_by_problem = defaultdict(list)
        self.problem_scores = {}

    def print_problem_scores(self):
        print("---------------------------------Problem and subtask scores---------------------------------")       
        for name, scores in self.problem_scores.items():
            print(f"# {name}: {scores} self.correct_submissions[name]: {self.correct_submissions[name]} self.total_submissions[name]: {self.total_submissions[name]}")