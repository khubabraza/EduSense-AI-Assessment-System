import requests

# Paste the public URL printed by your MCQ v3 Colab notebook.
# Example: "https://abc-def-xyz.trycloudflare.com"
MCQ_API_BASE_URL = "https://styles-wet-correct-interference.trycloudflare.com"

# This is for the justification scoring adapter.
# If your scoring adapter Colab notebook is not running right now, temporarily use the same v3 URL
# so quiz submission does not break. But this will use fallback scoring, not the trained scoring adapter.
SCORING_API_BASE_URL = "https://styles-wet-correct-interference.trycloudflare.com"


def predict_justification_score_api(
    student_justification="",
    model_justification="",
    student_answer="",
    expected_answer="",
    question="",
    options="",
    **kwargs
):
    """
    Calls the external scoring API.

    Important:
    - Return a numeric score only when the LLM API gives a valid score.
    - Return None when the API fails, times out, or gives invalid output.
    - app.py will then safely fall back to cosine similarity.
    """

    try:
        base_url = SCORING_API_BASE_URL.rstrip("/")
        url = f"{base_url}/score"

        payload = {
            "question": question or "",
            "options": options or "",
            "student_justification": student_justification or student_answer or "",
            "model_justification": model_justification or expected_answer or "",
            "student_answer": student_answer or student_justification or "",
            "expected_answer": expected_answer or model_justification or ""
        }

        response = requests.post(url, json=payload, timeout=25)

        if response.status_code != 200:
            print("LLM scoring API returned non-200 status:", response.status_code)
            return None

        try:
            data = response.json()
        except Exception as e:
            print("LLM scoring API did not return valid JSON:", e)
            return None

        score = (
            data.get("score")
            or data.get("predicted_score")
            or data.get("llm_score")
        )

        if score is None:
            print("LLM scoring API response has no score field:", data)
            return None

        score = float(score)

        if score < 0:
            score = 0.0

        if score > 100:
            score = 100.0

        return score

    except Exception as e:
        print("LLM scoring API failed. App will use cosine fallback:", e)
        return None

def predict_justification_evaluation_api(
    student_justification="",
    model_justification="",
    student_answer="",
    expected_answer="",
    question="",
    options="",
    **kwargs
):
    """
    Returns:
        {
            "score": float,
            "feedback": str
        }

    Supports different possible Colab /score response formats.
    If feedback is not returned by Colab, a safe academic fallback feedback is generated.
    """
    try:
        url = f"{SCORING_API_BASE_URL.rstrip('/')}/score"

        payload = {
            "question": question or "",
            "options": options or "",
            "student_justification": student_justification or student_answer or "",
            "model_justification": model_justification or expected_answer or "",
            "student_answer": student_answer or student_justification or "",
            "expected_answer": expected_answer or model_justification or ""
        }

        response = requests.post(url, json=payload, timeout=120)
        data = response.json()

        score = (
            data.get("score")
            or data.get("predicted_score")
            or data.get("llm_score")
            or data.get("justification_score")
            or 0
        )

        feedback = (
            data.get("feedback")
            or data.get("llm_feedback")
            or data.get("explanation")
            or data.get("reason")
            or data.get("comment")
            or ""
        )

        score = float(score)

        if not feedback:
            if score >= 80:
                feedback = "The student demonstrated strong conceptual understanding and explained the answer clearly."
            elif score >= 50:
                feedback = "The student showed partial understanding, but the explanation needs more clarity and completeness."
            elif score > 0:
                feedback = "The response shows limited understanding and needs a more accurate explanation of the concept."
            else:
                feedback = "The response does not provide enough relevant reasoning for the selected answer."

        return {
            "score": score,
            "feedback": feedback
        }

    except Exception as e:
        print("LLM evaluation API failed:", e)

        return {
            "score": 0.0,
            "feedback": "Detailed AI feedback could not be prepared at this moment."
        }

def generate_mcqs_from_notes_api(
    notes_text="",
    num_mcqs=3,
    topic_name="",
    target_topic="",
    generation_mode="pasted_notes"
):
    try:
        base_url = MCQ_API_BASE_URL.rstrip("/")
        url = f"{base_url}/generate_mcqs"

        payload = {
            "notes_text": notes_text or "",
            "num_mcqs": int(num_mcqs),
            "topic_name": topic_name or "",
            "target_topic": target_topic or "",
            "generation_mode": generation_mode or "pasted_notes"
        }

        print("Calling MCQ API:", url)
        print("Payload mode:", payload["generation_mode"])
        print("Payload notes length:", len(payload["notes_text"]))
        print("Payload target topic:", payload["target_topic"])

        response = requests.post(url, json=payload, timeout=240)

        print("MCQ API status:", response.status_code)
        print("MCQ API response preview:", response.text[:500])

        try:
            data = response.json()
        except Exception:
            return {
                "success": False,
                "partial_success": False,
                "exact_success": False,
                "requested_count": num_mcqs,
                "returned_count": 0,
                "mcqs": [],
                "message": "",
                "error": (
                    f"MCQ API did not return valid JSON. "
                    f"Status={response.status_code}. "
                    f"Preview={response.text[:300]}"
                ),
                "raw_output": ""
            }

        return {
            "success": data.get("success", False),
            "partial_success": data.get("partial_success", False),
            "exact_success": data.get("exact_success", False),
            "requested_count": data.get("requested_count", num_mcqs),
            "returned_count": data.get("returned_count", len(data.get("mcqs", []))),
            "mcqs": data.get("mcqs", []),
            "message": data.get("message", ""),
            "error": data.get("error"),
            "raw_output": data.get("raw_output", "")
        }

    except Exception as e:
        return {
            "success": False,
            "partial_success": False,
            "exact_success": False,
            "requested_count": num_mcqs,
            "returned_count": 0,
            "mcqs": [],
            "message": "",
            "error": str(e),
            "raw_output": ""
        }

# -------------------------------------------------------
# Direct test
# Run: python llm_client.py
# -------------------------------------------------------

if __name__ == "__main__":
    score = predict_justification_score_api(
        question="Which data type is used to store decimal numbers?",
        options="A) int B) string C) float D) boolean",
        model_justification="Float data type is used to store numbers with decimal points.",
        student_justification="Float is used for decimal values because it can store numbers with fractional parts."
    )

    print("Predicted score from Colab API:", score)

    sample_notes = """
    A stack is a linear data structure that follows the Last In First Out principle, also called LIFO.
    The main operations of stack are push, pop, and peek.
    Push inserts an element at the top of the stack.
    Pop removes the top element from the stack.
    Peek returns the top element without removing it.
    """

    generated_mcqs = generate_mcqs_from_notes_api(
        notes_text=sample_notes,
        num_mcqs=3,
        topic_name="Stack",
        target_topic="Stack operations",
        generation_mode="pasted_notes"
    )

    print("\nGenerated MCQs from Colab API:")
    print(generated_mcqs)