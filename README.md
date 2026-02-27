# Use of LLM to Analyze Math Strategy Learning in ITS


## Description

This project feeds historical student problem-solving data into a Gemini LLM, asking it to role-play as a specific student and predict whether that student would:

- Use the **optimal strategy** (Equivalent Ratios or Means and Extremes)
- **Solve for the unknown** without errors
- Obtain the **correct final answer** without errors

After each prediction, the pipeline feeds back the ground-truth outcomes so Gemini can adapt its simulation to the student's evolving learning trajectory. Results are compared against ground-truth labels and saved to CSV.

---

## Requirements

Install dependencies with pip:

```
pip install google-genai pandas
```

**Python:** 3.10+

**API access:** A valid Google Gemini API key with access to `gemini-2.5-flash`.

---

## Input Data

The pipeline reads `student_problem_info.csv` with the following columns:

| Column | Description |
|---|---|
| `student_id` | Unique identifier for each student |
| `cluster_number` | Cluster/group the student belongs to |
| `problem_text` | The proportion problem presented to the student |
| `correct_strategy` | Ground truth: did the student use the optimal strategy? (`yes`/`no`) |
| `correct_unknown` | Ground truth: did the student solve the unknown without errors? (`yes`/`no`) |
| `correct_answer` | Ground truth: did the student get the correct final answer? (`yes`/`no`) |

---

## System Prompt

The model is instructed to simulate a 7th-grade student solving ratios and proportions problems of the form:

```
PercentageChange / 100 = AmountChange / AmountOriginal
```

Two strategies are described:

- **Equivalent Ratios (ER):** Scale one ratio to the other by multiplying/dividing by a common integer. Best when the scaling factor is a whole number.
- **Means and Extremes (ME):** Cross-multiply and solve algebraically. Best when the scaling factor is non-integer.

After solving each problem, the model self-assesses on the three behavioral dimensions and returns them in a structured summary block:

```
---SUMMARY---
optimal_strategy: yes or no
solved_unknown: yes or no
correct_final_answer: yes or no
---END---
```

---

## Feedback Loop

After each problem, the pipeline sends the ground-truth outcomes back to Gemini:

```
Here is the correct information for this problem:
- The student used the optimal strategy: <yes/no>
- The student solved for the unknown with no errors: <yes/no>
- The student obtained the correct final answer with no errors: <yes/no>
Use this to adjust your reasoning for the next problem based on the student's learning trajectory.
```

This allows the model to refine its simulation as it learns more about the student's tendencies across problems.

---

## Output

Results are saved incrementally to `evaluation_results.csv` after each student. The output includes:

| Column | Description |
|---|---|
| `student_id` | Student identifier |
| `problem_number` | Problem index within the session |
| `problem_text` | The problem that was posed |
| `cluster_number` | Student cluster |
| `gemini_optimal_strategy` | Gemini's prediction (`yes`/`no`) |
| `gemini_solved_unknown` | Gemini's prediction (`yes`/`no`) |
| `gemini_correct_answer` | Gemini's prediction (`yes`/`no`) |
| `correct_strategy` | Ground truth (`yes`/`no`) |
| `correct_unknown` | Ground truth (`yes`/`no`) |
| `correct_answer` | Ground truth (`yes`/`no`) |
| `strategy_match` | Whether prediction matched ground truth |
| `unknown_match` | Whether prediction matched ground truth |
| `answer_match` | Whether prediction matched ground truth |
| `gemini_raw_response` | Full raw model response for debugging |

---

## Usage

```python
python gemini_pipline.py
```

Or use `run_pipeline()` directly:

```python
import pandas as pd
from gemini_pipline import run_pipeline

df = pd.read_csv("student_problem_info.csv")
results = run_pipeline(df, api_key="YOUR_KEY", save_path="results.csv")
```