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


def process_results_pass64(doc, results):
    """Process all repeated results for pass@k computation.

    The filter build_predictions_instruct groups all 64 responses into a single
    instance, so results = [[code_0, code_1, ..., code_63]] (1 element containing
    all solutions).
    """
    test_case = doc["test"] + "\ncheck(" + doc["entry_point"] + ")"
    # Flatten: results is [[sol_0, ..., sol_63]] from the filter
    if len(results) == 1 and isinstance(results[0], list):
        solutions = results[0]
    else:
        solutions = [r[0] if isinstance(r, list) else r for r in results]
    # pass_at_k expects references=[test_case], predictions=[[sol_0, ..., sol_63]]
    return pass_at_k(references=[test_case], predictions=[solutions], k=[1, 64])


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def _extract_code_from_response(response: str, entry_point: str = "") -> str:
    """Extract the code block most likely to be the solution from markdown fences."""
    # Collect all code blocks
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

    # 1. Last block with the exact entry point definition
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
    return [
        [_extract_code_from_response(r, doc.get("entry_point", "")) for r in resp]
        for resp, doc in zip(resps, docs)
    ]
