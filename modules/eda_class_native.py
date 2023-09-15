import itertools as it
import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ipywidgets import fixed, interact, interact_manual, interactive

# from params import *
import logging


class Validator:
    validation: pd.DataFrame

    def __init__(self, filepath, references: pd.DataFrame):
        self.validation_set = self.load_validation(filepath)
        self.reference_df = references
        self.logger = logging.getLogger(__name__)
        self.logger.info("Validator initialized")
        # set up logging to file - see previous section for more details
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
            self.logger.error(
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


def standardize_runs_df(df, filter=False):
    dt = []

    def split_if_not_nan(x):
        if not pd.isna(x) and isinstance(x, str):
            return float(x.split(" ")[0])
        else:
            return x

    df["runtime"] = df["runtime"].apply(split_if_not_nan)

    for source in df.source.unique():
        sub = df[df.source == source].reset_index(drop=True)

        sub = sub.replace("NA", np.NaN)
        for cols in ["coverage", "ref_proportion", "depth", "depthR", "runtime"]:
            # print(sub[cols])
            if np.nanstd(sub[cols]) == 0 or sub.shape[0] == 0:
                sub[cols] = np.NaN
            else:
                sub[cols] = sub[cols].astype(float)
                # sub[cols] = sub[cols].apply(lambda x: np.log10(x))

                # sub[cols] = (sub[cols] - np.nanmean(sub[cols])) / np.nanstd(sub[cols])
                sub[cols] = sub[cols] / np.max(sub[cols])

            if filter:
                sub = sub[(sub[cols] < 4) & (sub[cols] > -4)]

        dt.append(sub)

    dt = pd.concat(dt, axis=0)
    dt_cols = list(dt.columns)

    dt.columns = dt_cols
    dt = dt.rename(columns={"run_x": "run"})
    return dt


def df_runid_summary(run_df, analysis_dir: str, technology: str):
    technology_text = technology.lower()
    report_file = os.path.join(analysis_dir, f"reports_runid.{technology_text}.tsv")
    if os.path.isfile(report_file):
        run_assess = pd.read_csv(report_file, sep="\t")
    else:
        run_assess = []
        for rid in run_df.runid.unique():
            rclub = run_df[run_df.runid == rid].reset_index(drop=True)

            source = rclub.source.unique()[0]
            run = rclub.run.unique()[0]

            csuc = sum(rclub.success) / rclub.shape[0]
            cass = sum(rclub.ahelp) / rclub.shape[0]
            cread = sum(rclub.rhelp) / rclub.shape[0]

            rclub_succ = rclub[rclub.success == True].reset_index(drop=True)
            # print(rclub_succ)

            ccov = 0
            cdepth = 0
            cdepthr = 0
            mapped_proportion = 0
            cruntime = 0
            crsucc = 0

            if len(rclub_succ) == 1:
                ccov = rclub_succ.coverage.values[0]
                cdepth = rclub_succ.depth.values[0]
                cdepthr = rclub_succ.depthR.values[0]
                mapped_proportion = rclub_succ.ref_proportion.values[0]
                cruntime = rclub_succ.runtime.values[0]
                crsucc = 1
            elif len(rclub_succ) > 1:
                ccov = np.mean(rclub_succ.coverage.values)
                cdepth = np.mean(rclub_succ.depth.values)
                cdepthr = np.mean(rclub_succ.depthR.values)
                mapped_proportion = np.mean(rclub_succ.ref_proportion.values)
                cruntime = np.mean(rclub_succ.runtime.values)
                crsucc = 1

            runsum = pd.DataFrame(
                [
                    [
                        run,
                        rid,
                        source,
                        csuc,
                        cass,
                        cread,
                        ccov,
                        cdepth,
                        cdepthr,
                        mapped_proportion,
                        cruntime,
                        crsucc,
                        sum(rclub.complete) / rclub.shape[0],
                    ]
                ],
                columns=[
                    "run",
                    "runid",
                    "source",
                    "precision",
                    "ahelp",
                    "rhelp",
                    "coverage",
                    "depth",
                    "depthR",
                    "ref_proportion",
                    "runtime",
                    "finished",
                    "complete",
                ],
            )

            run_assess.append(runsum)

        run_assess = pd.concat(run_assess).reset_index(drop=True)
        run_assess.to_csv(report_file, sep="\t", header=True, index=False)

    return run_assess


class run_eda:
    data_total: pd.DataFrame
    softs: pd.DataFrame
    source_total: dict
    sources: list
    run_summaries_filename: str = "run_summaries.tsv"
    combined_reports_filename: str = "combined_reports.tsv"
    reports_combined_full_filename: str = "reports.comb_full.tsv"

    def __init__(
        self,
        validator: Validator,
        report_file: str,
        params_file: str,
        intermediate_output_dir: str,
        samples_keep: list = [],
        technology: str = "illumina",
    ):
        self.validator = validator
        self.report_file = report_file
        self.params_file = params_file
        self.samples_keep = samples_keep
        self.intermediate_output_dir = intermediate_output_dir
        self.dir = os.path.dirname(report_file)
        self.run_summaries_path = os.path.join(
            self.intermediate_output_dir, self.run_summaries_filename
        )
        self.combined_reports_path = os.path.join(
            self.intermediate_output_dir, self.combined_reports_filename
        )
        self.reports_combined_full_path = os.path.join(
            self.intermediate_output_dir, self.reports_combined_full_filename
        )

        self.run_input()
        self.remove_samples_unvalidated()
        self.remove_samples_filter()
        self.complement_input()
        self.clean_softs()

        self.all_runs = self.softs.run.unique()
        self.all_samples = self.softs.sample_name.unique()
        with open(
            intermediate_output_dir + f"samples_analyzed_{technology}.txt", "w"
        ) as f:
            f.write("\n".join(self.all_samples))

    def run_input(self) -> None:
        """
        return report data sets
        """

        self.data_total = pd.read_csv(self.report_file, sep="\t", low_memory=False)
        self.softs = pd.read_csv(self.params_file, sep="\t", low_memory=False)

    def clean_softs(self):
        """
        replace nan with empty string"""
        self.softs = self.softs.replace("nan", "")
        self.softs = self.softs.fillna("")

    def filter_samples(self, samples_keep: list):
        if samples_keep == []:
            return
        ## samples not in validation set
        absent_samples_idx = ~self.data_total.sample_name.isin(samples_keep)
        absent_samples = self.data_total.sample_name[absent_samples_idx].unique()
        ## samples not in reports
        missing_samples = self.validator.validation_set.index[
            ~self.validator.validation_set.index.isin(self.data_total.sample_name)
        ]

        absent_runs = self.data_total.run_id[absent_samples_idx].unique()

        self.data_total = self.data_total[
            ~self.data_total.sample_name.isin(absent_samples)
        ]

        self.softs = self.softs[~self.softs.run_id.isin(absent_runs)]

    def remove_samples_unvalidated(self):
        self.filter_samples(self.validator.validation_set.index.tolist())

    def remove_samples_filter(self):
        self.filter_samples(self.samples_keep)

    def complement_input(self):
        self.data_total["runid"] = self.data_total.apply(
            lambda x: f"{x.project_id}_{x.sample_id}_{x.run_id}", axis=1
        )

        self.softs["runid"] = self.softs.apply(
            lambda x: f"{x.project_id}_{x.sample_id}_{x.run_id}", axis=1
        )

        self.data_total["source"] = self.data_total.apply(
            lambda x: f"{x.project_id}_{x.sample_id}", axis=1
        )

        self.data_total["success"] = self.data_total.apply(
            self.validator.assess, axis=1
        )

        self.data_total["ahelp"] = self.data_total.apply(
            self.validator.assess_assembly_classification, axis=1
        )

        self.data_total["rhelp"] = self.data_total.apply(
            self.validator.assess_read_classification, axis=1
        )

        self.data_total["complete"] = self.data_total.apply(
            self.validator.assess_complete, axis=1
        )

        self.softs["source"] = self.softs.apply(
            lambda x: f"{x.project_id}_{x.sample_id}", axis=1
        )

    def split(self):
        self.source_total = {
            x: g.reset_index(drop=True) for x, g in self.data_total.groupby("source")
        }

        self.sources = list(self.source_total.keys())

        self.args_dict = {
            x: g.reset_index(drop=True) for x, g in self.softs.groupby("runid")
        }

    def run_summary(self, stotal):
        run_summaries = []
        sample_name = stotal.sample_name.unique()[0]
        source = stotal.source.unique()[0]

        for run in self.all_runs:
            percent_over = 0
            Hdepth = 0
            HdepthR = 0
            mapped_reads = 0
            ref_proportion = 0
            mapped_proportion = 0
            ngaps = 0
            assh = False
            readh = False
            focus = 0
            found_success = False
            runtime = np.NaN
            description = ""

            if run in stotal.run_name.unique():
                rda = stotal[stotal.run_name == run].reset_index(drop=True)
                found_success = np.nansum(rda.success == True) > 0
                if found_success:
                    description = "; ".join(
                        rda.description[rda.success == True].unique()
                    )
                    dsuc = rda[rda.success == True]

                    runtime = max(rda.runtime)

                    focus = (
                        dsuc[dsuc.classification_success == True].shape[0]
                        / rda.shape[0]
                    )
                    percent_over = np.mean(dsuc["coverage"])
                    Hdepth = np.mean(dsuc["depth"])
                    HdepthR = np.mean(dsuc["depthR"])
                    mapped_reads = np.mean(dsuc["mapped_reads"])
                    ref_proportion = np.mean(dsuc["ref_proportion"])
                    mapped_proportion = np.mean(dsuc["mapped_proportion"])
                    ngaps = np.mean(dsuc["ngaps"])
                    assh = sum(dsuc.classification_success.str.contains("contigs")) > 0
                    readh = sum(dsuc.classification_success.str.contains("reads")) > 0

            run_summaries.append(
                [
                    sample_name,
                    source,
                    run,
                    found_success,
                    runtime,
                    percent_over,
                    Hdepth,
                    HdepthR,
                    mapped_reads,
                    ref_proportion,
                    mapped_proportion,
                    ngaps,
                    readh,
                    assh,
                    focus,
                    description,
                ]
            )

        run_summaries = pd.DataFrame(
            run_summaries,
            columns=[
                "sample_name",
                "source",
                "run",
                "runid",
                "success",
                "runtime",
                "coverage",
                "depth",
                "depthR",
                "mapped_reads",
                "ref_proportion",
                "mapped_proportion",
                "ngaps",
                "rhelp",
                "ahelp",
                "focus",
                "description",
            ],
        )
        return run_summaries

    def summarize_runs(self):
        if os.path.isfile(self.run_summaries_path):
            pass
            self.summaries = pd.read_csv(self.run_summaries_path, sep="\t")

        else:
            # self.split()
            self.summaries = {}
            self.descriptions = {}

            for file in self.sources:
                stotal = self.source_total[file]

                stotal["success"] = stotal.apply(self.validator.assess, axis=1)
                run_summaries = self.run_summary(stotal)
                self.summaries[file] = run_summaries

            self.summaries = pd.concat(list(self.summaries.values()), axis=0)
            self.summaries.to_csv(self.run_summaries_path, sep="\t")

    def summarize_samples(self):
        if os.path.isfile(self.run_summaries_path):
            self.summaries = pd.read_csv(self.run_summaries_path, sep="\t")

        else:
            # self.split()
            self.summaries = {}

            self.descriptions = {}
            for file in self.sources:
                stotal = self.source_total[file]

                stotal["success"] = stotal.apply(self.validator.assess, axis=1)
                run_summaries = self.run_summary(stotal)
                self.summaries[file] = run_summaries

            self.summaries = pd.concat(list(self.summaries.values()), axis=0)
            self.summaries.to_csv(self.run_summaries_path, sep="\t")

    def describe(self, file=""):
        self.descriptions = {}

        for file in self.sources:
            stotal = self.source_total[file]

            sample = stotal.sample_name.unique()[0]
            infection = self.validator.validation_set.loc[sample][0]
            run_summaries = self.summaries[self.summaries.sample_name == sample]
            ssums = run_summaries[run_summaries.success == True]
            ###
            psc_runs = sum(run_summaries.success == True) / run_summaries.shape[0]
            psc_runs = round(psc_runs * 100, 3)
            report_runs = f"{run_summaries.shape[0]}"  # f"total number of runs: {run_summaries.shape[0]}"
            report_success = f"{psc_runs} %,  ({ssums.shape[0]})"  # f"N runs that find {infection} : {ssums.shape[0]},  ({psc_runs} %)"

            species_unique = [x.split("; ") for x in ssums.description.unique()]
            species_unique = list(it.chain(*species_unique))
            species_unique = "\n".join(list(set(species_unique)))
            allcover = (ssums.success == True) & (ssums.ahelp == True)
            allcover = sum(allcover)
            if ssums.shape[0]:
                succass = round(100 * allcover / ssums.shape[0], 3)
            else:
                succass = 0

            succtotal = round(100 * allcover / run_summaries.shape[0], 3)

            report1 = f"{succass} %"  # f"percent of successfull runs where assembly is also successful : {succass} %"
            report2 = f"{succtotal} %"  # f"percent of runs that find {infection} with both reads and assembly : {succtotal} %"

            self.descriptions[file] = [
                infection,
                report_runs,
                report_success,
                species_unique,
                report1,
                report2,
            ]

    def pretty_print(self, file):
        text = self.descriptions[file]
        return text

    def describe_print(self, file=""):
        text = [
            [
                "infection",
                "Number of runs",
                "success",
                "species",
                "% success assembled",
                "% success + assembly",
            ]
        ]

        if file:
            header = ["summary \ source"] + [file]
            text.append(self.pretty_print(file))
        else:
            text.extend([self.pretty_print(file) for file in self.sources])
            header = ["summary \ source"] + self.sources

        text = pd.DataFrame(text).T
        text.columns = header
        return text

    def eda_plots(self, file, cols=["Hdepth%", "coverage", "runtime"]):
        run_summaries = self.summaries[file]
        ssums = run_summaries[run_summaries.success == True]

        for col in cols:
            sns.histplot(data=ssums, x=col).set_title(file)
            plt.show()

    def combine_data(self):
        if os.path.isfile(self.combined_reports_path):
            combdat = pd.read_csv(self.combined_reports_path, sep="\t")
            combdat.success = combdat.success.astype(bool)

        else:
            total_softs = self.softs
            total_reports = self.data_total

            # for s, g in total_reports.items():
            #    g["source"] = s

            combdat = (
                pd.merge(total_softs, total_reports, on="runid")
                .rename({"source_x": "source"}, axis=1)
                .drop("source_y", axis=1)
            )

            combdat["complete"] = (combdat["success"] == True) & (
                combdat["ahelp"] == True
            ) & combdat["rhelp"] == True

            combdat.to_csv(
                self.combined_reports_path, sep="\t", header=True, index=False
            )

        self.combdat = combdat

    def get_combd_data_success(self):
        if self.combdat is None:
            self.combine_data()

        return self.combdat[self.combdat.success == True]

    def get_combd_data_failed(self):
        if self.combdat is None:
            self.combine_data()

        return self.combdat[self.combdat.success == False]

    def combine_data_full(self):
        if os.path.isfile(self.reports_combined_full_path):
            combdat_full = pd.read_csv(self.reports_combined_full_path, sep="\t")

        else:
            args_all = []
            for x, g in self.args_dict.items():
                newg = g.copy()
                newg["id"] = [x]
                newg.reset_index(inplace=True, drop=True)

                if len(args_all):
                    args_all = pd.concat([args_all, newg], axis=0)
                    args_all.reset_index(inplace=True, drop=True)
                else:
                    args_all = newg

            combdat_full = pd.merge(args_all, self.summaries, on="runid")
            combdat_full["complete"] = (
                (combdat_full["success"] == True)
                & (combdat_full["ahelp"] == True)
                & (combdat_full["rhelp"] == True)
            )

            combdat_full.to_csv(
                self.reports_combined_full_path, sep="\t", header=True, index=False
            )

        self.combdat_full = combdat_full

    def interactive_1d(self, cols=["Hdepth%", "%>2", "runtime"]):
        @interact(file=self.sources)
        def eda_plots_interact(file):
            f, axes = plt.subplots(1, 3)
            f.set_size_inches(15, 7)

            run_summaries = self.summaries[self.summaries.source == file]
            ssums = run_summaries[run_summaries.success == True]

            for ix, col in enumerate(cols):
                sns.histplot(data=ssums, x=col, ax=axes[ix]).set_title(col)

    def interactive_2d(self, cols=["Hdepth%", "%>2", "runtime"], kind="hist"):
        @interact(file=self.sources)
        def eda_plots_interact(file):
            run_summaries = self.summaries[file]
            ssums = run_summaries[run_summaries.success == True]
            f, axes = plt.subplots(1, 3)
            f.set_size_inches(15, 7)

            for ix, comb in enumerate(it.combinations(cols, 2)):
                plt.hist2d(
                    ssums[comb[0]], ssums[comb[1]], bins=(15, 15), cmap=plt.cm.BuPu
                )
                axes[ix].set(xlabel=comb[0], ylabel=comb[1])

            plt.show()


class plot_interact:
    def single_data_plot(scomb, module, rx):
        # scomb=scomb.sort_values(rx, ascending= False)
        source = scomb.source.unique()[0]

        sns.set(rc={"figure.figsize": (15, 8)})

        sns.boxplot(data=scomb, x=module, y=rx).set_title(source)

        plt.show()

    def explain(self, dataset, Rx, modules):
        dropdown_source = widgets.Dropdown(
            options=sorted(dataset.source.unique()),
            value=sorted(dataset.source.unique())[0],
            description="source:",
        )
        dropdown_variable = widgets.Dropdown(
            options=Rx, value=Rx[0], description="Var:"
        )
        dropdown_module = widgets.Dropdown(
            options=modules, value=modules[0], description="Module:"
        )

        def dropdown_source_eventhandler(change):
            """
            Eventhandler for the state dropdown widget
            """
            # display(input_widgets)
            source_choice = change.new
            # output_by_state(source_choice, dropdown_variable.value, dropdown_module.value)
            # IPython.display.clear_output(wait=True)

        def dropdown_variable_eventhandler(change):
            """
            Eventhandler for the question dropdown widget
            """
            # display(input_widgets)
            variable_choice = change.new
            # output_by_state(dropdown_source.value, variable_choice, dropdown_module.value)
            # IPython.display.clear_output(wait=True)

        def dropdown_module_eventhandler(change):
            """
            Event handler for the stratification dropdown widget
            """
            # display(input_widgets)
            module_choice = change.new
            # output_by_state(dropdown_source.value, dropdown_variable.value, module_choice)
            #

        dropdown_source.observe(dropdown_source_eventhandler, names="value")
        dropdown_variable.observe(dropdown_variable_eventhandler, names="value")
        dropdown_module.observe(dropdown_module_eventhandler, names="value")

        @interact(
            source=dropdown_source, variable=dropdown_variable, module=dropdown_module
        )
        def output_by_state(source, variable, module):
            """
            Takes in a state value, the specific question from the for loop and the specified stratification category.
            This function is called by the dropdown handlers below to pull the data based on user-input.
            """
            # IPython.display.clear_output(wait=True)

            output_data = dataset[dataset.source == source]
            output_data = single_data_plot(output_data, module, variable)


def single_data_plot(scomb, module, rx):
    # scomb=scomb.sort_values(rx, ascending= False)
    source = scomb.source.unique()[0]

    sns.set(rc={"figure.figsize": (15, 8)})

    sns.boxplot(data=scomb, x=module, y=rx).set_title(source)

    plt.show()


def dropdown_menu_eda(dataset, Rx, modules):
    output = widgets.Output()
    dropdown_source = widgets.Dropdown(
        options=sorted(dataset.source.unique()),
        value=sorted(dataset.source.unique())[0],
        description="source:",
    )
    dropdown_variable = widgets.Dropdown(options=Rx, value=Rx[0], description="Var:")
    dropdown_module = widgets.Dropdown(
        options=modules, value=modules[0], description="Module:"
    )

    def output_by_state(source, variable, module):
        """
        Takes in a state value, the specific question from the for loop and the specified stratification category.
        This function is called by the dropdown handlers below to pull the data based on user-input.
        """

        output_data = dataset[dataset.source == source]

        output_data = single_data_plot(output_data, module, variable)

        with output:
            display(output_data)

    def dropdown_source_eventhandler(change):
        """
        Eventhandler for the state dropdown widget
        """
        display(input_widgets)
        source_choice = change.new
        output_by_state(source_choice, dropdown_variable.value, dropdown_module.value)
        IPython.display.clear_output(wait=True)

    def dropdown_variable_eventhandler(change):
        """
        Eventhandler for the question dropdown widget
        """
        display(input_widgets)
        variable_choice = change.new
        output_by_state(dropdown_source.value, variable_choice, dropdown_module.value)
        IPython.display.clear_output(wait=True)

    def dropdown_module_eventhandler(change):
        """
        Event handler for the stratification dropdown widget
        """
        display(input_widgets)
        module_choice = change.new
        output_by_state(dropdown_source.value, dropdown_variable.value, module_choice)
        IPython.display.clear_output(wait=True)

    dropdown_source.observe(dropdown_source_eventhandler, names="value")
    dropdown_variable.observe(dropdown_variable_eventhandler, names="value")
    dropdown_module.observe(dropdown_module_eventhandler, names="value")

    input_widgets = widgets.HBox([dropdown_source, dropdown_variable, dropdown_module])

    display(input_widgets)
    output_by_state(sorted(dataset.source.unique())[0], Rx[0], modules[0])

    IPython.display.clear_output(wait=True)
