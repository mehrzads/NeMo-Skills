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


class ICPCMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, predictions):
        super().update(predictions)
#        self._compute_pass_at_k(predictions)
        if predictions:
            print("1. predictions[0]["name"]:", predictions[0]["name"])
            print("2. len(predictions):", len(predictions))
            self.predictions_by_problem[predictions[0]["name"]].extend(predictions)

    def _get_score_dict(self, p):
        return {"correct": all(r["score"] > 0 for r in p["test_case_results"].values())}

    def get_problem_score(self, submissions) -> float:
        """
        For a given problem (list of submissions), compute the score as follows:
          - For each subtask, take the maximum score over all submissions.
          - Sum these maximum scores to get the problem score.
        """
        if not submissions:
            return 0.0
        subtask_scores = {}

        for submission in submissions:
            name = submission["name"]
            result = submission["test_case_results"]
            if result["score"]:
                subtask_scores[name] = True;  
            else:
                if subtask_scores.get(name) is None:
                    subtask_scores[name] = False;    
            print("3. name, subtask_scores[name]:", name, subtask_scores[name])      
        return subtask_scores

 

    def get_metrics(self):
        total_score = total_round_robin = 0.0
        self.problem_scores = {}
        for name, submissions in self.predictions_by_problem.items():
            scores = self.get_problem_score(submissions)
            self.problem_scores[name] = scores
            print("4. name, scores:", name, scores)
        self.print_problem_scores()
        metrics_dict = super().get_metrics()
        for m in metrics_dict.values():
            m["total_score"] = str(total_score)
        return metrics_dict

    def reset(self):
        super().reset()
        self.predictions_by_problem = defaultdict(list)
        self.problem_scores = {}

    def print_problem_scores(self):
        print("---------------------------------Problem and subtask scores---------------------------------")       
        for name, scores in self.problem_scores.items():
            print(f"# {name}: {scores}")