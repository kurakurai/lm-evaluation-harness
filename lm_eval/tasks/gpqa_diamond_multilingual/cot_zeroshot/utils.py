import re


BOXED_LETTER_RE = re.compile(r"\\boxed\{([A-D])\}")

OPTION_RE = re.compile(
    r"\(A\)\s*(.+?)\s*\(B\)\s*(.+?)\s*\(C\)\s*(.+?)\s*\(D\)\s*(.+?)(?=\n\s*\n|\\boxed|\Z)",
    re.DOTALL,
)


def doc_to_target(doc):
    match = BOXED_LETTER_RE.search(doc["solution"])
    return match.group(1) if match else doc["solution"]


def process_docs(dataset):
    def _add_choices(doc):
        m = OPTION_RE.search(doc["problem"])
        if m:
            choices = [m.group(i).strip() for i in range(1, 5)]
        else:
            choices = []
        doc["choices"] = choices
        return doc

    dataset = dataset.map(_add_choices)
    return dataset.filter(lambda d: len(d["choices"]) == 4)
