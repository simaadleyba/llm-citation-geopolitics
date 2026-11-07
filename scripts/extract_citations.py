"""Turn raw responses into one row per (call, url), separating retrieved from cited.

Input:  data/raw/responses_*.jsonl
Output: data/processed/citations.csv
        columns: run_id, question_id, lang, model, retriever, rep, url, domain,
                 retrieved (bool), cited (bool), cite_rank
        data/processed/domains_to_label.csv  (unique domains not yet in labels)
"""


def load_raw(raw_dir="data/raw"):
    """Read all raw jsonl files into a list of records."""
    raise NotImplementedError


def to_domain(url):
    """Registrable domain via tldextract, e.g. https://www.bbc.co.uk/x -> bbc.co.uk."""
    raise NotImplementedError


def explode_record(record):
    """One row per unique url in the record, flagging retrieved and cited."""
    raise NotImplementedError


def unlabeled_domains(citations, labels_path="data/labels/domain_labels.csv"):
    """Domains present in citations but missing from the label file."""
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
