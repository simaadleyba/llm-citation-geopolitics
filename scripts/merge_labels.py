"""Join citations with domain labels and question factors into the analysis table.

Input:  data/processed/citations.csv
        data/labels/domain_labels.csv
        data/questions.jsonl
Output: data/processed/analysis.csv
        one row per (call, url) with: model origin, lang, topic_country,
        controversy, retriever, and domain labels (country, language, type,
        state_affiliated, us_lean), plus derived flags:
        home_country_outlet, lang_match.
"""


def load_inputs():
    raise NotImplementedError


def add_derived_flags(df):
    """home_country_outlet: outlet country == model origin.
    lang_match: outlet language == query language.
    """
    raise NotImplementedError


def check_coverage(df):
    """Report share of cited urls with no label; fail if above a threshold."""
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
