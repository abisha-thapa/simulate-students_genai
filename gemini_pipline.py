
import time
import pandas as pd
from google import genai

# ── Configuration ──────────────────────────────────────────────────────────────
API_KEY = "your_api_key"
MODEL = "gemini-2.5-flash"
RETRY_LIMIT = 3
RETRY_DELAY = 5

SYSTEM_PROMPT = (
    "You are simulating a 7th-grade student interacting with an intelligent tutoring system (ITS) "
    "whose learning trajectory I will reveal to you problem by problem. Solve each problem as this "
    "student would, based on what you have learned about their tendencies so far. Your answers should "
    "reflect the student's likely behavior, not ideal performance.\n\n"
    "THE TASK: For each problem, solve for the unknown variable in a proportion of the form:\n"
    "PercentageChange/100 = AmountChange/AmountOriginal\n\n"
    "STRATEGIES: There are two strategies to solve for the unknown:\n"
    "i) Equivalent Ratios (ER): Scale one ratio to the other by multiplying or dividing by a common "
    "integer to directly infer the unknown. E.g., x/20 = 50/100 → divide the right side by 5 → x = 10. "
    "ER is efficient when the scaling factor is an integer.\n"
    "ii) Means and Extremes (ME): Perform cross multiplication and solve for the unknown. "
    "ME is more efficient when the scaling factor is non-integer.\n"
    "You may use ER, ME, or BOTH.\n\n"
    "AFTER SOLVING EACH PROBLEM, assess whether the student you are simulating would have:\n"
    "i) used the optimal strategy (yes or no),\n"
    "ii) solved for the unknown using that strategy without making errors (yes or no), and\n"
    "iii) after solving for the unknown, used this value to obtain the correct final answer "
    "without making errors or requiring additional help (yes or no).\n\n"
    "IMPORTANT: At the end of EVERY response, you MUST include a summary block in EXACTLY this format:\n"
    "---SUMMARY---\n"
    "optimal_strategy: yes or no\n"
    "solved_unknown: yes or no\n"
    "correct_final_answer: yes or no\n"
    "---END---\n\n"
    "Once you do this, I will tell you whether the student you are simulating used the optimal strategy, "
    "solved for the unknown without errors, and obtained the correct final answer without errors. "
    "Use this to adjust your reasoning for the next problem based on the student's learning trajectory."
)


# ── Parsing ────────────────────────────────────────────────────────────────────
def parse_yes_no(s: str):
    """Extract yes/no from a string."""
    if s is None:
        return None
    s = s.strip().lower()
    if "yes" in s:
        return "yes"
    elif "no" in s:
        return "no"
    return None


def parse_response(response_text: str) -> dict:
    """
    Extract from Gemini's response:
      - 3 yes/no self-assessments
    """
    result = {
        "gemini_optimal_strategy": None,
        "gemini_solved_unknown": None,
        "gemini_correct_answer": None
    }

    if "---SUMMARY---" not in response_text:
        return result

    block = response_text.split("---SUMMARY---")[1]
    if "---END---" in block:
        block = block.split("---END---")[0]

    for line in block.strip().splitlines():
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        val_part = line_stripped.split(":")[-1].strip()

        # Yes/no self-assessments
        if "optimal_strategy" in line_lower:
            result["gemini_optimal_strategy"] = parse_yes_no(val_part)

        elif "solved_unknown" in line_lower:
            result["gemini_solved_unknown"] = parse_yes_no(val_part)

        elif "correct_final_answer" in line_lower:
            result["gemini_correct_answer"] = parse_yes_no(val_part)

    return result


# ── Core Pipeline ──────────────────────────────────────────────────────────────
def call_gemini(client, model, conversation_history, message_text):
    """Send a message and get a response, with retry logic."""
    conversation_history.append({
        "role": "user",
        "parts": [{"text": message_text}],
    })

    for attempt in range(RETRY_LIMIT):
        try:
            response = client.models.generate_content(
                model=model,
                contents=conversation_history,
            )
            response_text = response.text
            conversation_history.append({
                "role": "model",
                "parts": [{"text": response_text}],
            })
            return response_text
        except Exception as e:
            print(f"    [Retry {attempt + 1}/{RETRY_LIMIT}] Error: {e}")
            if attempt < RETRY_LIMIT - 1:
                time.sleep(RETRY_DELAY)
            else:
                print("    [FAILED] Skipping this call.")
                return None


def run_student_session(client, model, student_problems: pd.DataFrame) -> list[dict]:
    """
    Run one session for a single student (group of problems in order).
    Returns a list of result dicts, one per problem.
    """
    conversation_history = [
        {"role": "user", "parts": [{"text": f"[Instructions]\n{SYSTEM_PROMPT}"}]},
        {"role": "model", "parts": [{"text": (
            "Understood. I will simulate a student solving problems."
        )}]},
    ]

    results = []

    for idx, row in student_problems.iterrows():
        problem_num = len(results) + 1
        student_id = row["student_id"]
        problem_text = row["problem_text"]
        gt_strategy = str(row["correct_strategy"]).strip().lower()
        gt_unknown = str(row["correct_unknown"]).strip().lower()
        gt_answer = str(row["correct_answer"]).strip().lower()

        print(f"  Problem {problem_num}...")

        # Step 1: Send problem
        response_text = call_gemini(client, model, conversation_history, problem_text)

        if response_text is None:
            parsed = {k: None for k in [
                "gemini_optimal_strategy",
                "gemini_solved_unknown",
                "gemini_correct_answer"
            ]}
        else:
            parsed = parse_response(response_text)

        # Compare yes/no self-assessments against ground truth
        strategy_match = (
            parsed["gemini_optimal_strategy"] == gt_strategy
            if parsed["gemini_optimal_strategy"] is not None else None
        )
        unknown_match = (
            parsed["gemini_solved_unknown"] == gt_unknown
            if parsed["gemini_solved_unknown"] is not None else None
        )
        answer_match = (
            parsed["gemini_correct_answer"] == gt_answer
            if parsed["gemini_correct_answer"] is not None else None
        )

        results.append({
            "student_id": student_id,
            "problem_number": problem_num,
            "problem_text": problem_text,
            "cluster_number": row['cluster_number'],
            # Gemini self-assessments (yes/no)
            "gemini_optimal_strategy": parsed["gemini_optimal_strategy"],
            "gemini_solved_unknown": parsed["gemini_solved_unknown"],
            "gemini_correct_answer": parsed["gemini_correct_answer"],
            # Ground truth (yes/no)
            "correct_strategy": gt_strategy,
            "correct_unknown": gt_unknown,
            "correct_answer": gt_answer,
            # Match flags
            "strategy_match": strategy_match,
            "unknown_match": unknown_match,
            "answer_match": answer_match,
            # Raw response for debugging
            "gemini_raw_response": response_text,
        })

        # Step 2: Feed ground truth back
        feedback = (
            f"Here is the correct information for this problem:\n"
            f"- The student used the optimal strategy: {gt_strategy}\n"
            f"- The student solved for the unknown with no errors: {gt_unknown}\n"
            f"- The student obtained the correct final answer with no errors: {gt_answer}\n"
            f"Use this to adjust your reasoning for the next problem based on "
            f"the student's learning trajectory."
        )
        call_gemini(client, model, conversation_history, feedback)

    return results



def run_pipeline(
    df: pd.DataFrame,
    api_key: str = API_KEY,
    model: str = MODEL,
    save_path: str = "evaluation_results.csv"
) -> pd.DataFrame:
    """
    Main entry point.

    Input DataFrame columns:
        student_id, problem_text, correct_strategy (yes/no),
        correct_unknown (yes/no), correct_answer (yes/no)

    Parameters
    ----------
    save_path : str
        CSV path for incremental saves after each student.

    Returns a results DataFrame.
    """
    client = genai.Client(api_key=api_key)

    all_results = []
    students = list(df["student_id"].unique())

    print(f"Running pipeline for {len(students)} students...\n")

    for i, student_id in enumerate(students):
        student_df = df[df["student_id"] == student_id].copy()
        student_results = run_student_session(client, model, student_df)
        all_results.extend(student_results)

        # Save after each student
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(save_path, index=False)

    results_df = pd.DataFrame(all_results)
    return results_df


# ── Calling Gemini for each sampled student from each cluster in new session ────────────────────────────────────────────
if __name__ == "__main__":

    df = pd.read_csv("student_problem_info.csv")
    results_df = run_pipeline(df)

    print("\n" + "=" * 80)
    print("  SAVING RESULTS")
    print("=" * 80)

    print(results_df.head())


