import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def _extract_code_from_response(response: str, entry_point: str = "") -> str:
    """Extract the code block most likely to be the solution from markdown fences."""
    blocks = []
    for marker in ("```python", "```py", "```"):
        for part in response.split(marker)[1:]:
            if "```" in part:
                code = part.split("```", 1)[0].strip("\n")
            else:
                code = part.strip("\n")
            blocks.append(code)
        if blocks:
            break

    if not blocks:
        return response

    # 1. Last block with the exact entry point definition (unused for MBPP+, no entry_point)
    if entry_point:
        for block in reversed(blocks):
            if f"def {entry_point}" in block:
                return block

    # 2. Last block containing any function definition
    for block in reversed(blocks):
        if "def " in block:
            return block

    # 3. Last block that doesn't look like a test snippet
    for block in reversed(blocks):
        stripped = block.strip()
        if not stripped.startswith(("print(", "assert ", "# Test")):
            return block

    return blocks[-1]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    # MBPP+ has no entry_point column; extraction falls back to the last def block.
    return [
        [_extract_code_from_response(r, doc.get("entry_point", "")) for r in resp]
        for resp, doc in zip(resps, docs)
    ]
