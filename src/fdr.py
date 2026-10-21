import os
import sys
import argparse
import datetime
from collections import defaultdict
import re
from html import escape
import json
from pathlib import Path
import yaml
import asyncio

from dotenv import load_dotenv
import jinja2
import pandas as pd
import docker
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from playwright.async_api import async_playwright

import common

# environmental variable S3_BUCKET is loaded from the environment at startup.   If it's not available, the module will intentionally fail.
load_dotenv()
s3_bucket = os.environ.get("S3_BUCKET")
assert s3_bucket is not None, "This module requires an S3 bucket to be defined in the environmental variable S3_BUCKET"


# ---------------------------------------------------------------------------
# Default values.
#
# These defaults are the values of the constants that previously lived inline
# in this module. Each is exposed as an optional command line option below.
# ---------------------------------------------------------------------------

DEFAULT_RUN_ID = "verification-15-percentile-large-a"
DEFAULT_SAMPLE_TRIAL = "trial0001"  # Used to extract common infrastructure details

# SLO defaults. Note: the original module overrode the initial SLO values
# (100 ms / 0.0001) with the baseline-run means below, so those are used here.
DEFAULT_SLO_RT_MS = 26.063374
DEFAULT_SLO_FR = 0.000245

DEFAULT_TOLERANCE = 0.20

DEFAULT_TOTAL_CPU = 32
DEFAULT_TOTAL_MEMORY = 64_000_000_000

DEFAULT_TEMPLATE_DIR = "./src/templates"
DEFAULT_FIGURE_DIR = "../figures"

DEFAULT_AUTOSCALER_LOOKUP = {
    "hpa": "Kubernetes Horizontal Pod Autoscaler (HPA)",
    "vpa": "Kubernetes Vertical Pod Autoscaler (VPA)",
    "keda": "Kubernetes Event-Driven Autoscaler (KEDA)",
    "none": "No autoscaler (control condition)",
}

DEFAULT_TIME_SLICE_SECONDS = 30

DEFAULT_TITLE = "KASBench Full Disclosure Report"

DEFAULT_MICROSERVICES = [
    "globeco-allocation-service",
    "globeco-confirmation-service",
    "globeco-execution-service",
    "globeco-fix-engine",
    "globeco-order-generation-service",
    "globeco-order-service",
    "globeco-portfolio-accounting-service",
    "globeco-portfolio-management-portal",
    "globeco-portfolio-service",
    "globeco-pricing-service",
    "globeco-security-service",
    "globeco-trade-service",
]

DEFAULT_ALPHA = 0.50
DEFAULT_BETA = 0.50
DEFAULT_K = 5
DEFAULT_W_MAX = 1
W_THETA_A = W_TAU_A = 0.50
W_THETA_S = W_TAU_S = 0.50
W_S = 0.40
W_A = 0.40
W_C = 0.20
P = 3


def _set_parameters(
        params,
        metadata,
        autoscaler_lookup,
        tolerance,
        time_slice_seconds,
        alpha,
        beta,
        k,
        w_max,
        w_theta_a,
        w_tau_a,
        w_theta_s,
        w_tau_s,
        w_s,
        w_a,
        w_c,
        p,
    ):
    
    # convert metatadata into a table and store as a report param
    df = pd.DataFrame(metadata).T
    metadata_table_html = df.to_html(index=False, classes="data-table", 
        border=0, header=False)
    params["metadata_table_html"] = metadata_table_html

    # extract and format the start and end times
    start_time = metadata["start_time"]["value"]
    end_time = metadata["end_time"]["value"]
    start_time_formatted = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').strftime('%B %d, %Y at %H:%M:%S UTC')
    params["start_date"] = start_time_formatted
    end_time_formatted = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S').strftime('%B %d, %Y at %H:%M:%S UTC')
    params["end_date"] = end_time_formatted

    #  set the autoscaler list
    params["autoscaler_list"] = [v for v in autoscaler_lookup.values()]

    # set the remaining command line parameters
    params["time_slice_seconds"] = time_slice_seconds

    params["tolerance_percent"] = str(int(tolerance * 100))

    params["tolerance"] = tolerance

    params["alpha"] = alpha
    params["beta"] = beta
    params["k"] = k

    words_dict = {
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"
    }

    params["k_word"] = words_dict[k]

    params["w_max"] = w_max

    params["w_theta_a"] = W_THETA_A
    params["w_tau_a"] = W_TAU_A
    params["w_theta_s"] = W_THETA_S
    params["w_tau_s"] = W_TAU_S
    params["w_s"] = W_S
    params["w_a"] = W_A
    params["w_c"] = W_C
    params["p"] = P


    return params



def run_benchmark(
    run_id=DEFAULT_RUN_ID,
    sample_trial=DEFAULT_SAMPLE_TRIAL,
    data_dir=None,
    slo_rt_ms=DEFAULT_SLO_RT_MS,
    slo_fr=DEFAULT_SLO_FR,
    tolerance=DEFAULT_TOLERANCE,
    total_cpu=DEFAULT_TOTAL_CPU,
    total_memory=DEFAULT_TOTAL_MEMORY,
    template_dir=DEFAULT_TEMPLATE_DIR,
    figure_dir=DEFAULT_FIGURE_DIR,
    output_path=None,
    autoscaler_lookup=None,
    time_slice_seconds=DEFAULT_TIME_SLICE_SECONDS,
    title=DEFAULT_TITLE,
    microservices=None,
    alpha=DEFAULT_ALPHA,
    beta=DEFAULT_BETA,
    k=DEFAULT_K,
    w_max=DEFAULT_W_MAX,
    w_theta_a=W_THETA_A,
    w_tau_a=W_TAU_A,
    w_theta_s=W_THETA_S,
    w_tau_s=W_TAU_S,
    w_s=W_S,
    w_a=W_A,
    w_c=W_C,
    p=P,
):
    """Run the full disclosure report benchmark.

    All parameters are optional and default to the values that were previously
    hard-coded in this module. Parameters whose defaults depend on ``run``
    (``data_dir`` and ``output_path``) are derived from ``run`` when left as
    ``None``.
    """

    # Derived defaults that depend on `run`.
    if data_dir is None:
        data_dir = f"./data/{run_id}"
        os.makedirs(data_dir, exist_ok=True)
    if output_path is None:
        output_path = f"./output/fdr-{run_id}.pdf"
    if autoscaler_lookup is None:
        autoscaler_lookup = dict(DEFAULT_AUTOSCALER_LOOKUP)
    if microservices is None:
        microservices = list(DEFAULT_MICROSERVICES)

    run_db = f"{data_dir}/logs.db"

    report_date = datetime.datetime.now(datetime.UTC)

    params = {}
    params["title"] = title

    # step 1: Download the the benchmark database created by the benchmark controller
    benchmark_db_path =f"{data_dir}/benchmark.db"
    common.download_benchmark_db(s3_bucket, run_id, benchmark_db_path)

    # step 2: Download the experiment progress
    experiment_progress = common.get_experiment_progress(s3_bucket, run_id)

    # step 3: Download the tofu outputs
    tofu_outputs = common.get_tofu_outputs(s3_bucket, run_id, sample_trial)

    # step 4: Download the run details
    run_details = common.get_run_details(s3_bucket, run_id, sample_trial)

    # step 5: populate the metadata
    metadata = common.populate_metadata(
        experiment_progress, 
        tofu_outputs, 
        run_details, 
        run_id, 
        autoscaler_lookup, 
        s3_bucket, 
        report_date,
    )
    # save metadata to a file
    with open(f"{data_dir}/{run_id}-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


    # step 6: Set parameters
    params = _set_parameters(
        params,
        metadata,
        autoscaler_lookup,
        tolerance,
        time_slice_seconds,
        alpha,
        beta,
        k,
        w_max,
        w_theta_a,
        w_tau_a,
        w_theta_s,
        w_tau_s,
        w_s,
        w_a,
        w_c,
        p=P,
    )

    # save params to json file
    with open(f"{data_dir}/{run_id}-params.json", "w") as f:
        json.dump(params, f, indent=2)

    # step 7: get Kubernetes-deployed container images

    deployments_table_html, statefulsets_table_html = common.compile_container_images(s3_bucket, run_id, sample_trial)
    params["deployments_table_html"] = deployments_table_html
    params["statefulsets_table_html"] = statefulsets_table_html
    
    # step 8: get the docker images of benchmark framework components

    benchmark_images = common.get_benchmark_images_df(s3_bucket, run_id, sample_trial)
    params["benchmark_images_table_html"] = benchmark_images.to_html(
        index=False,
        escape=False,
        header=True,
        classes="data-table compact",
        border=0
    )

    # step 9: get Helm chart versions
    helm_df = common.get_helm_versions(s3_bucket, run_id)
    params["helm_versions_table_html"] = helm_df.to_html(
        index=False,
        escape=False,
        header=True,
        classes="data-table compact",
        border=0
    )



    # step 10 download the Locust database files and load them into a single database
    common.download_and_load_locust_dbs(s3_bucket, run_id, run_db, data_dir)
    
    # step 11: get the log as a dataframe
    log_df = common.get_log_as_dataframe(run_db)

    # save log_df in parquet format
    log_df.to_parquet(f"{data_dir}/{run_id}-log.parquet")

    # step 12: get the files in the S3 database for the run_id
    files = common.get_s3_file_listing(s3_bucket, run_id)

    # step 13: generate performance plots and tables
    response_time_filename, failure_rate_filename, round_trip_filename, merged_trial_summary_df, autoscaler_summary_df = \
        common.generate_performance_plots(log_df, files, s3_bucket, run_id)
   
    # save merged_trial_summary_df in parquet format
    merged_trial_summary_df.to_parquet(f"{data_dir}/{run_id}-trial-summary.parquet")

    # save autoscaler_summary_df in parquet format
    autoscaler_summary_df.to_parquet(f"{data_dir}/{run_id}-autoscaler-summary.parquet")

    params["response_time_plot_path"] =  os.path.abspath(response_time_filename)
    params["failure_rate_plot_path"] = os.path.abspath(failure_rate_filename)
    params["round_trip_plot_path"] = os.path.abspath(round_trip_filename)
    params["merged_trial_summary_table_html"] = common.merged_trial_summary_df_to_html(merged_trial_summary_df)
    params["autoscaler_summary_table_html"] = common.autoscaler_summary_df_to_html(autoscaler_summary_df)


    # step 14: get earliest and latest run times
    earliest_run_time, latest_run_time = common.get_run_start_and_end_times_from_db(run_db)

    # step 15: calculate synchronous theta and tau
    weighted_theta_results, theta_results, theta_df = common.calculate_weighted_synchronous_theta(log_df, earliest_run_time, latest_run_time,
        slo_rt_ms=slo_rt_ms, slo_fr=slo_fr, k=k, alpha=alpha)
    params["theta_table_html"] = common.weighted_synchronous_theta_to_html(weighted_theta_results)
    
    weighted_tau_results, tau_results = common.calculate_weighted_synchronous_tau(log_df, earliest_run_time, latest_run_time,
        slo_rt_ms=slo_rt_ms, slo_fr=slo_fr, tolerance=tolerance, alpha=alpha)
    params["tau_table_html"] = common.weighted_synchronous_tau_to_html(weighted_tau_results)

    # save weighted_theta_results to parquet format
    common.to_parquet_safe(weighted_theta_results, f"{data_dir}/{run_id}-weighted-synchronous-theta-results.parquet")
    
    # save theta_results.time_slices to parquet
    common.to_parquet_safe(theta_results.time_slices, f"{data_dir}/{run_id}-synchronous-theta-time-slices.parquet")

    # save theta_results.trials to parquet format
    common.to_parquet_safe(theta_results.trials, f"{data_dir}/{run_id}-synchronous-theta-trials.parquet")

    # save theta_results.autoscalers to parquet format
    common.to_parquet_safe(theta_results.autoscalers, f"{data_dir}/{run_id}-synchronous-theta-autoscalers.parquet")

    # save theta_df to parquet format
    common.to_parquet_safe(theta_df, f"{data_dir}/{run_id}-synchronous-theta-results.parquet")

    # save weighted_tau_results to parquet format
    common.to_parquet_safe(weighted_tau_results, f"{data_dir}/{run_id}-weighted-synchronous-tau-results.parquet")

    # save tau_results.time_slices to parquet format
    common.to_parquet_safe(tau_results.time_slices, f"{data_dir}/{run_id}-synchronous-tau-time-slices.parquet")

    # save tau_results.trials to parquet format
    common.to_parquet_safe(tau_results.trials, f"{data_dir}/{run_id}-synchronous-tau-trials.parquet")

    # save tau_results.autoscalers to parquet format
    common.to_parquet_safe(tau_results.autoscalers, f"{data_dir}/{run_id}-synchronous-tau-autoscalers.parquet")


    # step 16: Get missing kafka metrics from TSDB snapshots (add this query to KASBench Runner)
    query = "sum(rate(kafka_publish_success_total[2m]))"
    client = docker.from_env()
    port = 9099

    arrival_rate_df = common.get_metric_from_tsdb(
        s3_bucket,
        files,
        query,
        client=client,
        port=port)

    # step 17: calculate asynchronous theta and tau
    lag_df, agg_df, agg_trial_df = common.get_asynchronous_metrics(files, run_id, s3_bucket, earliest_run_time, latest_run_time,
        time_slice_seconds=time_slice_seconds,
        arrival_rate_df=arrival_rate_df, w_max=w_max, tolerance=tolerance, k=k)

    # Save lag_df and agg_df in parquet format using common.to_parquet_safe
    common.to_parquet_safe(lag_df, f"{data_dir}/{run_id}-asynchronous-lag.parquet")
    common.to_parquet_safe(agg_df, f"{data_dir}/{run_id}-asynchronous-aggregated.parquet")
    common.to_parquet_safe(agg_trial_df, f"{data_dir}/{run_id}-asynchronous-trial.parquet")

    params["asynchronous_metrics_table_html"] = common.weighted_asynchronous_metrics_df_to_html(agg_df)

    # step 18: calculate the cost metric
    cpu_baseline, memory_baseline = common.get_minimum_memory_from_helm_project()
    c_bar_df, c_bar_trial_df = common.calculate_c_metric(
        s3_bucket, 
        run_id, 
        earliest_run_time, 
        latest_run_time, 
        cpu_baseline,
        memory_baseline,
        microservices,
        time_slice_seconds, 
        total_cpu,
        total_memory,
        beta)

    common.to_parquet_safe(c_bar_df, f"{data_dir}/{run_id}-cost-metric.parquet")
    common.to_parquet_safe(c_bar_trial_df, f"{data_dir}/{run_id}-cost-metric-trials.parquet")

    params["cost_metrics_table_html"] = common.cost_metrics_df_to_html(c_bar_df)

    # step 19: calculate the overall score

    summary_df, summary_table_html = common.calculate_summary(
        weighted_theta_results, 
        weighted_tau_results, 
        agg_df, 
        c_bar_df,
        w_theta_a,
        w_tau_a,
        w_theta_s,
        w_tau_s,
        w_s,
        w_a,
        w_c,
        p,
    )
    params["summary_table_html"] = summary_table_html

    common.to_parquet_safe(summary_df, f"{data_dir}/{run_id}-summary.parquet")

    # step 20: generate the summary plots
    
    filename=os.path.abspath("./figures/summary-metrics-bar-chart.png")
    common.plot_summary_metrics_bar_chart(
        summary_df,
        autoscaler_order=["hpa", "keda", "none", "vpa"],
        filename=filename,
    )
    params["summary_metrics_bar_chart"] = filename

    filename = os.path.abspath("./figures/summary-metrics-radar-chart.png")
    common.plot_summary_metrics_radar_chart(
        summary_df,
        filename=filename,
    )
    params["summary_metrics_radar_chart"] = filename


    # step 21: load MathML macros

    params = common.load_mathml_macros(params)


    # step 22: load autoscaler descriptions
    params = common.load_autoscaler_descriptions(params)

    # step 23: render the report


    asyncio.run(common.render_report(template_dir, output_path, params, metadata))




    config = {
        "run": run_id,
        "sample_trial": sample_trial,
        "data_dir": data_dir,
        "run_db": run_db,
        "slo_rt_ms": slo_rt_ms,
        "slo_fr": slo_fr,
        "tolerance": tolerance,
        "total_cpu": total_cpu,
        "total_memory": total_memory,
        "template_dir": template_dir,
        "figure_dir": figure_dir,
        "output_path": output_path,
        "autoscaler_lookup": autoscaler_lookup,
        "time_slice_seconds": time_slice_seconds,
        "title": title,
        "microservices": microservices,
        "alpha": alpha,
        "beta": beta,
        "k": k,
        "w_max": w_max,
        "w_theta_a": w_theta_a,
        "w_tau_a": w_tau_a,
        "w_theta_s": w_theta_s,
        "w_tau_s": w_tau_s,
        "w_s": w_s,
        "w_a": w_a,
        "w_c": w_c,
        "p": p,
        "params": params,
    }

    return config


def _parse_key_value_pairs(values):
    """Parse a list of ``key=value`` strings into a dict."""
    result = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Expected key=value, got: {item!r}"
            )
        key, value = item.split("=", 1)
        result[key] = value
    return result


def build_arg_parser():
    """Build the command line argument parser for the benchmark."""
    parser = argparse.ArgumentParser(
        description="Generate the KASBench Full Disclosure Report.",
    )

    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help="Run identifier. (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-trial",
        default=DEFAULT_SAMPLE_TRIAL,
        help="Sample trial used to extract common infrastructure details. "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Data directory. (default: ../data/<run>)",
    )
    parser.add_argument(
        "--slo-rt-ms",
        type=float,
        default=DEFAULT_SLO_RT_MS,
        help="Response-time SLO in milliseconds. (default: %(default)s)",
    )
    parser.add_argument(
        "--slo-fr",
        type=float,
        default=DEFAULT_SLO_FR,
        help="Failure-rate SLO as a proportion. (default: %(default)s)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Tolerance. (default: %(default)s)",
    )
    parser.add_argument(
        "--total-cpu",
        type=int,
        default=DEFAULT_TOTAL_CPU,
        help="Total CPU. (default: %(default)s)",
    )
    parser.add_argument(
        "--total-memory",
        type=int,
        default=DEFAULT_TOTAL_MEMORY,
        help="Total memory in bytes. (default: %(default)s)",
    )
    parser.add_argument(
        "--template-dir",
        default=DEFAULT_TEMPLATE_DIR,
        help="Template directory. (default: %(default)s)",
    )
    parser.add_argument(
        "--figure-dir",
        default=DEFAULT_FIGURE_DIR,
        help="Figure directory. (default: %(default)s)",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output PDF path. (default: ../output/fdr-<run>.pdf)",
    )
    parser.add_argument(
        "--autoscaler-lookup",
        nargs="+",
        metavar="KEY=VALUE",
        default=None,
        help="Autoscaler lookup entries as key=value pairs. "
        "(default: the built-in autoscaler lookup table)",
    )
    parser.add_argument(
        "--time-slice-seconds",
        type=int,
        default=DEFAULT_TIME_SLICE_SECONDS,
        help="Time slice duration in seconds. (default: %(default)s)",
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="Report title. (default: %(default)s)",
    )
    parser.add_argument(
        "--microservices",
        nargs="+",
        metavar="SERVICE",
        default=None,
        help="List of microservice names. "
        "(default: the built-in microservice list)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Alpha weight. (default: %(default)s)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help="Beta weight. (default: %(default)s)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="K parameter. (default: %(default)s)",
    )
    parser.add_argument(
        "--w-max",
        type=int,
        default=DEFAULT_W_MAX,
        help="Maximum weight. (default: %(default)s)",
    )
    parser.add_argument(
        "--w-theta-a",
        type=float,
        default=W_THETA_A,
        help="Weight of asynchronous theta. (default: %(default)s)",
    )
    parser.add_argument(
        "--w-tau-a",
        type=float,
        default=W_TAU_A,
        help="Weight of asynchronous tau. (default: %(default)s)",
    )
    parser.add_argument(
        "--w-theta-s",
        type=float,
        default=W_THETA_S,
        help="Weight of synchronous theta. (default: %(default)s)",
    )
    parser.add_argument(
        "--w-tau-s",
        type=float,
        default=W_TAU_S,
        help="Weight of synchronous tau. (default: %(default)s)",
    )
    parser.add_argument(
        "--w-s",
        type=float,
        default=W_S,
        help="Weight of the synchronous component. (default: %(default)s)",
    )
    parser.add_argument(
        "--w-a",
        type=float,
        default=W_A,
        help="Weight of the asynchronous component. (default: %(default)s)",
    )
    parser.add_argument(
        "--w-c",
        type=float,
        default=W_C,
        help="Weight of the cost component. (default: %(default)s)",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=P,
        help="P parameter. (default: %(default)s)",
    )

    return parser


def main(argv=None):
    """Command line entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    autoscaler_lookup = None
    if args.autoscaler_lookup is not None:
        autoscaler_lookup = _parse_key_value_pairs(args.autoscaler_lookup)

    return run_benchmark(
        run_id=args.run_id,
        sample_trial=args.sample_trial,
        data_dir=args.data_dir,
        slo_rt_ms=args.slo_rt_ms,
        slo_fr=args.slo_fr,
        tolerance=args.tolerance,
        total_cpu=args.total_cpu,
        total_memory=args.total_memory,
        template_dir=args.template_dir,
        figure_dir=args.figure_dir,
        output_path=args.output_path,
        autoscaler_lookup=autoscaler_lookup,
        time_slice_seconds=args.time_slice_seconds,
        title=args.title,
        microservices=args.microservices,
        alpha=args.alpha,
        beta=args.beta,
        k=args.k,
        w_max=args.w_max,
        w_theta_a=args.w_theta_a,
        w_tau_a=args.w_tau_a,
        w_theta_s=args.w_theta_s,
        w_tau_s=args.w_tau_s,
        w_s=args.w_s,
        w_a=args.w_a,
        w_c=args.w_c,
        p=args.p,
    )


if __name__ == "__main__":
    main()
