import numpy as np
import pandas as pd
import logging


class Validator:
    validation: pd.DataFrame

    def __init__(self, filepath, references: pd.DataFrame):
        self.validation_set = self.load_validation(filepath)
        self.reference_df = references

        self.logger = logging.getLogger(__name__)
        self.logger.info("Validator initialized")
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
            datefmt="%m-%d %H:%M",
            filename="logs/validator.log",
            filemode="w",
        )

    def check_content(self, df):
        checksum = 0
        if "taxid" not in df.columns:
            df["taxid"] = np.nan
            checksum += 1

        if "accid" not in df.columns:
            df["accid"] = np.nan
            checksum += 1

        if "description" not in df.columns:
            df["description"] = np.nan
            checksum += 1

        if checksum == 3:
            self.logger.info(
                "all columns absent. provide at least one validator: taxid, accid or description."
            )

        return df

    def load_validation(self, file_path):
        df = pd.read_csv(file_path, sep="\t")

        df = self.check_content(df)

        df = df.dropna()
        df["taxid"] = df.taxid.apply(lambda x: [y for y in x.split(";")])
        df["accid"] = df.accid.apply(lambda x: [y for y in x.split(";")])

        df.set_index("sample_name", inplace=True)

        return df

    def assess(self, x):
        if x.taxid in self.validation_set.loc[x.sample_name].taxid:
            return True
        if x.accid in self.validation_set.loc[x.sample_name].accid:
            return True

        for subdesc in self.validation_set.loc[x.sample_name].description.split(";"):
            if subdesc.lower() in x.description.lower():
                return True

        return False

    def assess_ignorant(self, x):
        samples = []

        for sample_name in self.validation_set.index:
            if x.taxid in self.validation_set.loc[sample_name].taxid:
                samples.append(sample_name)
            if x.accid in self.validation_set.loc[sample_name].accid:
                samples.append(sample_name)

            added = 0
            for subdesc in self.validation_set.loc[sample_name].description.split(";"):
                if added > 1:
                    break
                if subdesc.lower() in x.description.lower():
                    samples.append(sample_name)
                    added += 1

        if len(samples):
            return ";".join(samples)

        return "None"

    def assess_assembly_classification(self, x):
        run_references = self.reference_df[
            self.reference_df.run_id == x.run_id
        ].reset_index(drop=True)

        classification_source = run_references[
            run_references.taxid == x.taxid
        ].classification_source.unique()

        if len(classification_source) == 0:
            self.logger.info(
                f"no classification source found for taxid: {x.taxid} in run: {x.run_id}, sample: {x.sample_name}"
            )
            return False

        classification_source = classification_source[0]

        if classification_source in [2, 3]:
            return True

        return False

    def assess_read_classification(self, x):
        run_references = self.reference_df[
            self.reference_df.run_id == x.run_id
        ].reset_index(drop=True)
        classification_source = run_references[
            run_references.taxid == x.taxid
        ].classification_source.unique()

        if len(classification_source) == 0:
            return False

        classification_source = classification_source[0]

        if classification_source in [1, 3]:
            return True

        return False

    def assess_complete(self, x):
        if x.rhelp == True and x.ahelp == True:
            return True

        return False
