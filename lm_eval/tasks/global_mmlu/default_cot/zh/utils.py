from functools import partial


CATEGORIES = ["Business", "Humanities", "Medical", "Other", "STEM", "Social Sciences"]

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def doc_to_target(doc):
    ans = doc["answer"]
    # Handle both int index (0-3) and string letter ("A"-"D")
    if isinstance(ans, int):
        return "(%s)" % ANSWER_MAP[ans]
    return "(%s)" % ans


def _add_choices(doc):
    doc["choices"] = ["(A)", "(B)", "(C)", "(D)"]
    return doc


def process_docs(dataset, category):
    return dataset.filter(lambda x: x["subject_category"] == category).map(_add_choices)


process_functions = {
    f"process_{category.lower().replace(' ', '_')}": partial(
        process_docs, category=category
    )
    for category in CATEGORIES
}

globals().update(process_functions)
