### input data
from modules.constants_settings import ConstantsSettings as CS
import pandas as pd
from modules.validator import Validator


def infer_source_files_technology(technology, INPUT_DIR, METADATA_DIR):
    if technology == CS.TECHNOLOGY_minion:
        all_reports_file = INPUT_DIR + "all_reports_ont.tsv"
        all_parameters_file = INPUT_DIR + "all_parameters_ont.tsv"
        all_references_file = INPUT_DIR + "all_references_ont.tsv"
        validation_file = METADATA_DIR + "benchmark_ont_validation.tsv"
    elif technology == CS.TECHNOLOGY_illumina_old:
        all_reports_file = INPUT_DIR + "all_reports_illumina.tsv"
        all_parameters_file = INPUT_DIR + "all_parameters_illumina.tsv"
        all_references_file = INPUT_DIR + "all_references_illumina.tsv"
        validation_file = METADATA_DIR + "benchmark_illumina_validation.tsv"
    else:
        raise ValueError("Technology not supported")

    return all_reports_file, all_parameters_file, all_references_file, validation_file


def read_references_filter(references_file, benchmark_prefix="benchmark_batch"):
    raw_refs = pd.read_csv(references_file, sep="\t")
    raw_refs = raw_refs[raw_refs.project.str.contains(benchmark_prefix)]

    return raw_refs


def raw_refs_validate(raw_refs: pd.DataFrame, validator: Validator):
    raw_ref_unique = raw_refs.drop_duplicates(subset=["accid"])[
        ["taxid", "accid", "description"]
    ]

    raw_ref_unique["samples_found"] = raw_ref_unique.apply(
        validator.assess_ignorant, axis=1
    )

    raw_refs = pd.merge(
        raw_refs, raw_ref_unique[["accid", "samples_found"]], on="accid", how="left"
    )
    raw_refs["found"] = raw_refs.apply(
        lambda x: True if x.sample_name in x.samples_found.split(";") else False, axis=1
    )
    return raw_refs
