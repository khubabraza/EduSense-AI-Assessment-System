import os
import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# -------------------------------------------------------
# EduSense LLM Scorer Configuration
# -------------------------------------------------------

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ADAPTER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "edusense_mistral_lora_adapter"
)

_model = None
_tokenizer = None


# -------------------------------------------------------
# Score Extraction
# -------------------------------------------------------

def extract_score(text):
    """
    Extract the first valid score from model output.
    Returns an integer between 0 and 100.
    """
    if not text:
        return 0

    match = re.search(r"\b(100|[0-9]{1,2})\b", text)

    if match:
        score = int(match.group(1))
        return max(0, min(100, score))

    return 0


# -------------------------------------------------------
# Model Loading
# -------------------------------------------------------

def load_llm_model():
    """
    Loads base Mistral 7B + EduSense LoRA adapter.
    Model is loaded only once and reused for all scoring calls.
    """

    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(
            f"EduSense LoRA adapter not found at: {ADAPTER_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. Mistral 7B 4-bit inference requires an NVIDIA GPU. "
            "Use cosine fallback or run the LLM scorer on a GPU machine."
        )

    print("Loading EduSense LLM scorer...")
    print("Adapter path:", ADAPTER_PATH)
    print("GPU:", torch.cuda.get_device_name(0))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager"
    )

    base_model.config.use_cache = True
    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    _model = model
    _tokenizer = tokenizer

    print("EduSense LLM scorer loaded successfully.")

    return _model, _tokenizer


# -------------------------------------------------------
# Prompt Builder
# -------------------------------------------------------

def build_scoring_prompt(question, options, model_justification, student_justification):
    """
    Builds the same style prompt used during fine-tuning.
    """

    return f"""### Instruction:
Compare the student justification with the model justification and give a quality score from 0 to 100 based on how well the student understood the concept. Return ONLY the number.

### Input:
Question: {question}
Options: {options}
Model Justification: {model_justification}
Student Justification: {student_justification}

### Response:
"""


# -------------------------------------------------------
# Main Scoring Function
# -------------------------------------------------------

def predict_justification_score(question, options, model_justification, student_justification):
    """
    Returns only the justification quality score from 0 to 100.
    This score will later be converted in app.py:
        justification_marks = (score / 100) * 0.6
    """

    if not student_justification or not student_justification.strip():
        return 0

    model, tokenizer = load_llm_model()

    prompt = build_scoring_prompt(
        question=question,
        options=options,
        model_justification=model_justification,
        student_justification=student_justification
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    raw_output = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    score = extract_score(raw_output)

    return score


# -------------------------------------------------------
# Direct Test
# Run: python llm_scorer.py
# -------------------------------------------------------

if __name__ == "__main__":
    result = predict_justification_score(
        question="Which data type is used to store decimal numbers?",
        options="A) int B) string C) float D) boolean",
        model_justification="Float data type is used to store numbers with decimal points.",
        student_justification="Float is used for decimal values because it can store numbers with fractional parts."
    )

    print("Predicted justification score:", result)