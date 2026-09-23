from __future__ import annotations

import io
import os
import datetime
import sqlite3
import json
from dataclasses import dataclass
import re
import requests
import time
import shutil
import tempfile
from urllib.parse import urlparse
import subprocess


import boto3
import pandas as pd
import numpy as np
from scipy.stats import t
from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame
import matplotlib.pyplot as plt
from scipy import stats


s3 = boto3.client("s3")


def download_file_from_s3(s3_bucket, s3_key, local_path):
    """Download a file from S3 to a local path."""
    s3.download_file(s3_bucket, s3_key, local_path)
    print(f"Downloaded {s3_key} to {local_path}")


def get_s3_file_listing(s3_bucket, s3_prefix=''):
    """Get a listing of files in an S3 bucket."""
    file_list = []
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                file_list.append(obj)
    return file_list

def get_s3_file(s3_bucket, s3_filename, local_filename, force=False):
    """Download a file from S3 to a local path."""
    # skip if file exists and not force
    if os.path.exists(local_filename) and not force: 
        return
    # download the tile
    s3.download_file(s3_bucket, s3_filename, local_filename)



def download_locust_db(data_dir, s3_bucket, s3_key, trial, role, filename, force=False, verbose=False):
    """Downloads the Locust file to local disk"""
    # the local path is data_dir/trial/db/role/filename
    local_file_path = f"{data_dir}/{trial}/db/{role}/{filename}"
    
    # skip if the file already exists
    if os.path.exists(local_file_path) and not force:
        if verbose:
            print(f"File {local_file_path} already exists, skipping...")
        return local_file_path
    
    # create the directory structure if it doesn't exist
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

    if verbose:
        print(f"Downloading {s3_key} to {local_file_path}")
    
    # download the file
    s3.download_file(s3_bucket, s3_key, local_file_path)
    return local_file_path


def get_experiment_progress(s3_bucket, run_id):
    """ Get the experiment-progress.json from S3"""
    key = f"{run_id}/experiment-progress.json"
    obj = s3.get_object(Bucket=s3_bucket, Key=key)
    return json.loads(obj['Body'].read())


def download_benchmark_db(s3_bucket, run_id, local_file_path):
    """ Downloads the benchmark database from S3"""
    # if local_file_path exists, return
    if os.path.exists(local_file_path):
        return
    key = f"{run_id}/benchmark.db"
    s3.download_file(s3_bucket, key, local_file_path)


def get_benchmark_start_end(benchmark_db_path):
    """Get the start and end time of the benchmark"""
    conn = sqlite3.connect(benchmark_db_path)
    cursor = conn.cursor()

    # Get the minimum and maximum timestamps
    cursor.execute("SELECT MIN(event_time), MAX(event_time) FROM events")
    start_time, end_time = cursor.fetchone()

    # Convert timestamps to datetime objects with timezone
    # Note: the timestamps in the database are already in UTC.
    # start_time and end_time are strings in the form "2026-09-04T15:26:11Z".  
    
    end_time = end_time.replace(tzinfo=datetime.timezone.utc)
    start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%SZ")
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%SZ")
    conn.close()

    return start_time, end_time


def get_tofu_outputs(s3_bucket, run_id, trial_id):
    """Get the tofu outputs from S3"""
    key = f"{run_id}/{trial_id}/infrastructure/tofu_outputs.json"
    obj = s3.get_object(Bucket=s3_bucket, Key=key)
    return json.loads(obj['Body'].read())


def is_locust_db(file_info):
    """Check if the file info refers to a locust database file"""
    s3_key = file_info["Key"]
    is_db_path = s3_key.split("/")[-2] == "db" 
    return is_db_path and s3_key.endswith(".db")

def parse_locust_db_file_info(file_info):
    """Parse a locust DB key to extract the trial, role, and filename"""
    s3_key = file_info["Key"]
    parts = s3_key.split("/")
    trial, filename = parts[-3], parts[-1]
    role = filename.split(".")[0]
    return trial, role, filename

def parse_roundtrip_file_info(file_info):
    """Parse a roundtrip file key to extract the trial and filename"""
    s3_key = file_info["Key"]
    parts = s3_key.split("/")
    trial, filename = parts[-3], parts[-1]
    return trial, filename

def parse_run_details_file_info(file_info):
    """Parse a run details file key to extract the trial and filename"""
    s3_key = file_info["Key"]
    parts = s3_key.split("/")
    trial, filename = parts[-2], parts[-1]
    return trial, filename

def parse_snapshot_name(snapshot: str) -> tuple[str, str]:
    """Parses a snapshot name into its components."""
    # Example: exp-2026-09-04-a/trial0001/tsdb-snapshots/20260904T160520Z-543178ecef36b42c/
    parts = snapshot.split("/")
    return parts[:2]

    
def create_log_database(db_path):
    """Creates the log database"""

    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            run_id TEXT,
            trial_id TEXT, 
            role TEXT,
            autoscaler TEXT,
            user_id TEXT,
            request_type TEXT,
            name TEXT,
            response_time REAL,
            response_length INTEGER,
            response TEXT,
            status_code TEXT,
            reason TEXT,
            exception TEXT,
            start_time REAL,
            url TEXT,
            method_and_name TEXT,
            success BOOLEAN,
            failure BOOLEAN
        )
    ''')

def get_raw_log_as_dataframe(filename):
    """Gets the Locust database output from each trial as a dataframe"""
    conn = sqlite3.connect(filename)
    df = pd.read_sql_query("SELECT * FROM logs", conn)
    conn.close()

    df["method_and_name"] = df.request_type + " " + df.name
    
    df.start_time = pd.to_datetime(df.start_time, unit='s')
    df.set_index("start_time")

    return df

def get_log_as_dataframe(filename):
    """Gets the aggregate Locust database as a dataframe"""
    conn = sqlite3.connect(filename)
    df = pd.read_sql_query("SELECT * FROM logs", conn)
    conn.close()

    # df["method_and_name"] = df.request_type + " " + df.name
    
    # df.start_time = pd.to_datetime(df.start_time, unit='s')
    # df.set_index("start_time")

    return df


def get_run_details(s3_bucket, run_id, trial_id, verbose=False):
    """Get the run details from S3"""
    key = f"{run_id}/{trial_id}/run_details.json"
    obj = s3.get_object(Bucket=s3_bucket, Key=key)
    return json.loads(obj['Body'].read())


def get_autoscaler(run_details):
    """ Get the autoscaler from the run details"""
    autoscaler = run_details["initialization"]["autoscaler"]
    return autoscaler    


def get_trial_summary_df(df):
    """ Computes aggregates for each trial """
    df = df.copy()
    df["is_success"] = pd.to_numeric(df["status_code"], errors="coerce").between(200, 299)
    response_time_summary = ( df[df.is_success].groupby(["autoscaler", "trial_id"], as_index=False)
        .agg(
            successful_requests=("status_code", "size"),
            mean_response_time=("response_time", "mean"),
        ))
    failure_rate_summary = ( df.groupby(["autoscaler", "trial_id"], as_index=False)
        .agg(
            requests=("status_code", "size"),
            failures=("failure", "sum"),
            failure_rate=("failure", "mean"),
        ))
    return response_time_summary.merge(failure_rate_summary, on=["autoscaler", "trial_id"], how="outer")


    return (
        df.groupby(["autoscaler", "trial_id"], as_index=False)
        .agg(
            requests=("status_code", "size"),
            mean_response_time=("response_time", "mean"),
            failures=("failure", "sum"),
            failure_rate=("failure", "mean"),
        )
    )


def get_autoscaler_summary(
    trial_summary: pd.DataFrame,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Compute trial-level aggregates for each autoscaler."""

    autoscaler_summary = (
        trial_summary
        .groupby("autoscaler", as_index=False)
        .agg(
            trials=("trial_id", "count"),

            mean_rt=("mean_response_time", "mean"),
            median_rt=("mean_response_time", "median"),
            sd_rt=("mean_response_time", "std"),

            mean_failure_rate=("failure_rate", "mean"),
            median_failure_rate=("failure_rate", "median"),
            sd_failure_rate=("failure_rate", "std"),

            mean_roundtrip_completion_percentage=("roundtrip_completion_percentage", "mean"),
            median_roundtrip_completion_percentage=("roundtrip_completion_percentage", "median"),
            sd_roundtrip_completion_percentage=("roundtrip_completion_percentage", "std"),

            total_quantity_ordered=("quantity_ordered", "sum"),
            total_quantity_placed=("quantity_placed", "sum"),
            total_quantity_filled=("quantity_filled", "sum"),
        )
    )

    # Student-t critical value for each autoscaler's number of trials.
    autoscaler_summary["t_critical"] = autoscaler_summary["trials"].apply(
        lambda n: (
            t.ppf(
                1 - (1 - confidence_level) / 2,
                df=n - 1,
            )
            if n >= 2
            else np.nan
        )
    )

    autoscaler_summary["ci95_rt"] = (
        autoscaler_summary["t_critical"]
        * autoscaler_summary["sd_rt"]
        / np.sqrt(autoscaler_summary["trials"])
    )

    autoscaler_summary["ci95_failure_rate"] = (
        autoscaler_summary["t_critical"]
        * autoscaler_summary["sd_failure_rate"]
        / np.sqrt(autoscaler_summary["trials"])
    )

    autoscaler_summary["ci95_roundtrip_completion_percentage"] = (
        autoscaler_summary["t_critical"]
        * autoscaler_summary["sd_roundtrip_completion_percentage"]
        / np.sqrt(autoscaler_summary["trials"])
    )

    # Optional pooled percentage. This weights trials by quantity ordered.
    autoscaler_summary["pooled_roundtrip_completion_percentage"] = (
        100
        * autoscaler_summary["total_quantity_filled"]
        / autoscaler_summary["total_quantity_ordered"]
    )

    autoscaler_summary = autoscaler_summary.drop(columns="t_critical")

    return autoscaler_summary


def merge_roundtrip_completion_percentage(files, s3_bucket, trial_summary_df, run_id=None):
        
    roundtrip_df = None

    if files is None:
        files = get_s3_file_listing(s3_bucket, run_id)

    for file in files:
        # select the roundtrip json results
        if "roundtrip/trade_orders.json" in file["Key"]:
            # parse the trial and filename from the s3 key
            trial, filename = parse_roundtrip_file_info(file)

            # Fetch object content directly from S3 using boto3
            response = s3.get_object(Bucket=s3_bucket, Key=file["Key"])
            content_bytes = response["Body"].read()

            # Read JSON into pandas via an in-memory BytesIO buffer
            roundtrip = pd.read_json(io.BytesIO(content_bytes))

            #  add trial_id column to roundtrip df
            roundtrip["trial_id"] = trial

            #  concat roundtrip df with existing roundtrip_df
            if roundtrip_df is None:
                roundtrip_df = roundtrip
            else:
                roundtrip_df = pd.concat([roundtrip_df, roundtrip], ignore_index=True)

    # join trial_summary_df and roundtrip_df on trial_id
    merged_trial_summary_df = trial_summary_df.merge(roundtrip_df, on="trial_id", how="left")

    # calculate roundtrip percentage
    merged_trial_summary_df["roundtrip_completion_percentage"] = merged_trial_summary_df.quantity_filled/merged_trial_summary_df.quantity_ordered * 100
    
    # remove unnecessary columns
    # merged_trial_summary_df = merged_trial_summary_df[['autoscaler', 'trial_id', 'requests', 'mean_response_time', 'failures',
    #     'failure_rate', 'roundtrip_completion_percentage']]
    
    return merged_trial_summary_df


@dataclass(frozen=True)
class ThetaResults:
    """Results produced by calculate_theta_metrics."""

    time_slices: pd.DataFrame
    trials: pd.DataFrame
    autoscalers: pd.DataFrame


def calculate_theta_metrics(
    requests_df: pd.DataFrame,
    *,
    slo_rt_ms: float = 100.0,
    slo_fr: float = 0.01,
    autoscaler_col: str = "autoscaler",
    trial_col: str = "trial_id",
    time_slice_col: str = "time_slice",
    response_time_col: str = "response_time",
    failure_col: str = "failure",
) -> ThetaResults:
    """
    Calculate all four theta metrics:

        theta_u_rt: response time above SLO
        theta_o_rt: response time below SLO
        theta_u_fr: failure rate above SLO
        theta_o_fr: failure rate below SLO

    The input dataframe contains one row per request and is already assigned
    to fixed-duration time slices.

    For each time slice:

        RT_t = mean response time of requests in the slice

        FR_t = number of failed requests / total number of request rows

    Because all time slices have equal duration, Delta-t cancels during
    normalization, and each theta is 100 times the mean normalized deviation
    across time slices.

    Parameters
    ----------
    requests_df:
        Request-level dataframe.

    slo_rt_ms:
        Response-time SLO in milliseconds.

    slo_fr:
        Failure-rate SLO expressed as a proportion. For example, use 0.01
        for a 1% failure-rate SLO.

    Returns
    -------
    ThetaResults
        time_slices:
            One row per autoscaler, trial, and time slice.

        trials:
            One row per autoscaler and trial.

        autoscalers:
            Experiment-level mean, standard deviation, SEM, and trial count
            for each autoscaler.
    """
    if slo_rt_ms <= 0:
        raise ValueError("slo_rt_ms must be greater than zero.")

    if not 0 < slo_fr <= 1:
        raise ValueError("slo_fr must be greater than 0 and no greater than 1.")

    required_columns = {
        autoscaler_col,
        trial_col,
        time_slice_col,
        response_time_col,
        failure_col,
    }

    missing_columns = required_columns.difference(requests_df.columns)

    if missing_columns:
        raise ValueError(
            "The dataframe is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    df = requests_df.copy()

    df[response_time_col] = pd.to_numeric(
        df[response_time_col],
        errors="coerce",
    )

    df[failure_col] = pd.to_numeric(
        df[failure_col],
        errors="coerce",
    )

    if df[response_time_col].isna().any():
        raise ValueError(
            f"Column {response_time_col!r} contains missing or nonnumeric values."
        )

    if df[failure_col].isna().any():
        raise ValueError(
            f"Column {failure_col!r} contains missing or nonnumeric values."
        )

    invalid_failure_values = ~df[failure_col].isin([0, 1])

    if invalid_failure_values.any():
        invalid_values = sorted(
            df.loc[invalid_failure_values, failure_col].unique().tolist()
        )
        raise ValueError(
            f"Column {failure_col!r} must contain only 0 and 1. "
            f"Invalid values: {invalid_values}"
        )

    group_columns = [
        autoscaler_col,
        trial_col,
        time_slice_col,
    ]

    # .size() counts request rows independently of missing values in any
    # particular data column.
    time_slices = (
        df.groupby(group_columns, observed=True, dropna=False)
        .agg(
            mean_response_time_ms=(response_time_col, "mean"),
            failures=(failure_col, "sum"),
            requests=(failure_col, "size"),
        )
        .reset_index()
    )

    time_slices["failure_rate"] = (
        time_slices["failures"] / time_slices["requests"]
    )

    # Underprovisioning: observed result is worse than the SLO.
    time_slices["normalized_u_rt"] = (
        time_slices["mean_response_time_ms"] - slo_rt_ms
    ).clip(lower=0) / slo_rt_ms

    time_slices["normalized_u_fr"] = (
        time_slices["failure_rate"] - slo_fr
    ).clip(lower=0) / slo_fr

    # Overprovisioning: observed result is better than the SLO.
    time_slices["normalized_o_rt"] = (
        slo_rt_ms - time_slices["mean_response_time_ms"]
    ).clip(lower=0) / slo_rt_ms

    time_slices["normalized_o_fr"] = (
        slo_fr - time_slices["failure_rate"]
    ).clip(lower=0) / slo_fr

    trial_group_columns = [
        autoscaler_col,
        trial_col,
    ]

    trials = (
        time_slices
        .groupby(trial_group_columns, observed=True, dropna=False)
        .agg(
            theta_u_rt=("normalized_u_rt", lambda values: 100.0 * values.mean()),
            theta_o_rt=("normalized_o_rt", lambda values: 100.0 * values.mean()),
            theta_u_fr=("normalized_u_fr", lambda values: 100.0 * values.mean()),
            theta_o_fr=("normalized_o_fr", lambda values: 100.0 * values.mean()),
            time_slices=(time_slice_col, "size"),
            requests=("requests", "sum"),
        )
        .reset_index()
    )

    metric_columns = [
        "theta_u_rt",
        "theta_o_rt",
        "theta_u_fr",
        "theta_o_fr",
    ]

    autoscalers = (
        trials
        .groupby(autoscaler_col, observed=True, dropna=False)[metric_columns]
        .agg(["mean", "std", "count"])
    )

    autoscalers.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in autoscalers.columns
    ]

    autoscalers = autoscalers.reset_index()

    for metric in metric_columns:
        autoscalers[f"{metric}_sem"] = (
            autoscalers[f"{metric}_std"]
            / np.sqrt(autoscalers[f"{metric}_count"])
        )

    return ThetaResults(
        time_slices=time_slices,
        trials=trials,
        autoscalers=autoscalers,
    )    


def append_time_slice(df, earliest_start_time, latest_end_time, time_slice_seconds=30):
    """Append a time_slice column to the dataframe."""
    
    # Make a copy of the dataframe
    df = df.copy()

    # Convert start_time to datetime and subtract the minimum start_time to get a timedelta
    df["start_time_dt"] = pd.to_datetime(df.start_time, format="mixed")
    latest_end_time = latest_end_time + pd.Timedelta(seconds=time_slice_seconds)

    # Calculate interval_range from the lowest start time to the highest start time of the dataframe
    time_slice_index = pd.interval_range(
        start=earliest_start_time,
        end=latest_end_time,
        freq=pd.Timedelta(seconds=time_slice_seconds),
        name="time_slice",
    )

    # Assign each row to a time slice based on start_time
    df["time_slice"] = pd.cut(
        df.start_time_dt,
        bins=time_slice_index,
        include_lowest=True, 
        labels=time_slice_index[:],
    )

    # Remove the df.start_time_dt column
    df = df.drop(columns=["start_time_dt"])

    # Return the dataframe
    return df


def add_weighted_theta(summary_df_by_autoscaler, gamma=0.50, omega=0.50):
    """Add weighted theta metrics to the summary dataframe."""

    # Make a copy of the dataframe
    summary_df_by_autoscaler = summary_df_by_autoscaler.copy()

    # Weight over and under provisioning by gamma
    summary_df_by_autoscaler["theta_rt_mean"] = gamma * summary_df_by_autoscaler["theta_u_rt_mean"] + (1 - gamma) * summary_df_by_autoscaler["theta_o_rt_mean"]
    summary_df_by_autoscaler["theta_fr_mean"] = gamma * summary_df_by_autoscaler["theta_u_fr_mean"] + (1 - gamma) * summary_df_by_autoscaler["theta_o_fr_mean"]
    
    # Weight response time and failure rate by omega
    summary_df_by_autoscaler["theta_mean"] = omega * summary_df_by_autoscaler["theta_rt_mean"] + (1 - omega) * summary_df_by_autoscaler["theta_fr_mean"]
    return summary_df_by_autoscaler



@dataclass(frozen=True)
class TauResults:
    """Results produced by calculate_tau_metrics."""

    time_slices: pd.DataFrame
    trials: pd.DataFrame
    autoscalers: pd.DataFrame


def calculate_tau_metrics(
    requests_df: pd.DataFrame,
    *,
    slo_rt_ms: float = 100.0,
    slo_fr: float = 0.01,
    autoscaler_col: str = "autoscaler",
    trial_col: str = "trial_id",
    time_slice_col: str = "time_slice",
    response_time_col: str = "response_time",
    failure_col: str = "failure",
    tolerance: float = 0.20
) -> TauResults:
    """
    Calculate all four tau time-share metrics:

        tau_u_rt:
            Percentage of time slices in which mean response time exceeds
            the response-time SLO.

        tau_o_rt:
            Percentage of time slices in which mean response time is below
            the response-time SLO.

        tau_u_fr:
            Percentage of time slices in which failure rate exceeds the
            failure-rate SLO.

        tau_o_fr:
            Percentage of time slices in which failure rate is below the
            failure-rate SLO.

    The input dataframe contains one row per request and must already assign
    each request to a fixed-duration time slice.

    For each time slice:

        RT_t = mean response time of requests in the slice

        FR_t = failed request rows / total request rows

    Since all time slices have equal duration, each tau metric is 100 times
    the mean of its corresponding zero-or-one indicator.

    Slices exactly equal to an SLO contribute zero to both the under- and
    over-SLO time shares, consistent with sgn(0) = 0.

    Parameters
    ----------
    requests_df:
        Request-level dataframe.

    slo_rt_ms:
        Response-time SLO in milliseconds.

    slo_fr:
        Failure-rate SLO expressed as a proportion. For example, 0.01 means
        a 1% failure-rate SLO.

    Returns
    -------
    TauResults
        time_slices:
            One row per autoscaler, trial, and time slice, including the
            calculated RT and FR values and indicator variables.

        trials:
            One row per autoscaler and trial containing the four tau values.

        autoscalers:
            Experiment-level mean, standard deviation, trial count, and
            standard error for each autoscaler.
    """
    if slo_rt_ms <= 0:
        raise ValueError("slo_rt_ms must be greater than zero.")

    if not 0 < slo_fr <= 1:
        raise ValueError("slo_fr must be greater than 0 and no greater than 1.")

    required_columns = {
        autoscaler_col,
        trial_col,
        time_slice_col,
        response_time_col,
        failure_col,
    }

    missing_columns = required_columns.difference(requests_df.columns)

    if missing_columns:
        raise ValueError(
            "The dataframe is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    df = requests_df.copy()

    df[response_time_col] = pd.to_numeric(
        df[response_time_col],
        errors="coerce",
    )

    df[failure_col] = pd.to_numeric(
        df[failure_col],
        errors="coerce",
    )

    if df[response_time_col].isna().any():
        raise ValueError(
            f"Column {response_time_col!r} contains missing or nonnumeric values."
        )

    if df[failure_col].isna().any():
        raise ValueError(
            f"Column {failure_col!r} contains missing or nonnumeric values."
        )

    invalid_failure_values = ~df[failure_col].isin([0, 1])

    if invalid_failure_values.any():
        invalid_values = sorted(
            df.loc[invalid_failure_values, failure_col].unique().tolist()
        )
        raise ValueError(
            f"Column {failure_col!r} must contain only 0 and 1. "
            f"Invalid values: {invalid_values}"
        )

    group_columns = [
        autoscaler_col,
        trial_col,
        time_slice_col,
    ]

    # .size() counts all request rows, independently of missing values in
    # any particular measurement column.
    time_slices = (
        df.groupby(group_columns, observed=True, dropna=False)
        .agg(
            mean_response_time_ms=(response_time_col, "mean"),
            failures=(failure_col, "sum"),
            requests=(failure_col, "size"),
        )
        .reset_index()
    )

    time_slices["failure_rate"] = (
        time_slices["failures"] / time_slices["requests"]
    )

    # Equivalent to max(sgn(RT_t - SLO_RT), 0).
    time_slices["indicator_u_rt"] = (
        time_slices["mean_response_time_ms"] > (slo_rt_ms * (1.0 + tolerance))
    ).astype(np.int8)

    # Equivalent to max(sgn(SLO_RT - RT_t), 0).
    time_slices["indicator_o_rt"] = (
        time_slices["mean_response_time_ms"] < (slo_rt_ms * (1.0 - tolerance))
    ).astype(np.int8)

    # Equivalent to max(sgn(FR_t - SLO_FR), 0).
    time_slices["indicator_u_fr"] = (
        time_slices["failure_rate"] > (slo_fr * (1.0 + tolerance))
    ).astype(np.int8)

    # Equivalent to max(sgn(SLO_FR - FR_t), 0).
    time_slices["indicator_o_fr"] = (
        time_slices["failure_rate"] < (slo_fr * (1.0 - tolerance))
    ).astype(np.int8)

    trial_group_columns = [
        autoscaler_col,
        trial_col,
    ]

    trials = (
        time_slices
        .groupby(trial_group_columns, observed=True, dropna=False)
        .agg(
            tau_u_rt=("indicator_u_rt", lambda values: 100.0 * values.mean()),
            tau_o_rt=("indicator_o_rt", lambda values: 100.0 * values.mean()),
            tau_u_fr=("indicator_u_fr", lambda values: 100.0 * values.mean()),
            tau_o_fr=("indicator_o_fr", lambda values: 100.0 * values.mean()),
            time_slices=(time_slice_col, "size"),
            requests=("requests", "sum"),
        )
        .reset_index()
    )

    metric_columns = [
        "tau_u_rt",
        "tau_o_rt",
        "tau_u_fr",
        "tau_o_fr",
    ]

    autoscalers = (
        trials
        .groupby(autoscaler_col, observed=True, dropna=False)[metric_columns]
        .agg(["mean", "std", "count"])
    )

    autoscalers.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in autoscalers.columns
    ]

    autoscalers = autoscalers.reset_index()

    for metric in metric_columns:
        autoscalers[f"{metric}_sem"] = (
            autoscalers[f"{metric}_std"]
            / np.sqrt(autoscalers[f"{metric}_count"])
        )

    return TauResults(
        time_slices=time_slices,
        trials=trials,
        autoscalers=autoscalers,
    )


def add_weighted_tau(summary_df_by_autoscaler, gamma=0.50, omega=0.50):
    """Add weighted tau metrics to the summary dataframe."""

    # Make a copy of the dataframe
    summary_df_by_autoscaler = summary_df_by_autoscaler.copy()

    # Weight over and under provisioning by gamma
    summary_df_by_autoscaler["tau_rt_mean"] = gamma * summary_df_by_autoscaler["tau_u_rt_mean"] + (1 - gamma) * summary_df_by_autoscaler["tau_o_rt_mean"]
    summary_df_by_autoscaler["tau_fr_mean"] = gamma * summary_df_by_autoscaler["tau_u_fr_mean"] + (1 - gamma) * summary_df_by_autoscaler["tau_o_fr_mean"]
    
    # Weight response time and failure rate by omega
    summary_df_by_autoscaler["tau_mean"] = omega * summary_df_by_autoscaler["tau_rt_mean"] + (1 - omega) * summary_df_by_autoscaler["tau_fr_mean"]
    return summary_df_by_autoscaler

def parse_metric_file_info(file_info):
    """Parse a metric file key to extract the trial and filename"""
    s3_key = file_info["Key"]
    parts = s3_key.split("/")
    trial, filename = parts[-3], parts[-1]
    return trial, filename


def get_merged_range_metric_df(files, run_id, s3_bucket, metric, column_label):
        
    metrics_df = None

    if files is None:
        files = get_s3_file_listing(s3_bucket, run_id)

    for file in files:
        # select the metrics file to process
        if f"metrics/{metric}" in file["Key"]:
            # parse the trial and filename from the s3 key
            trial, filename = parse_metric_file_info(file)

            # Fetch object content directly from S3 using boto3
            response = s3.get_object(Bucket=s3_bucket, Key=file["Key"])
            content_bytes = response["Body"].read()

            # Decode bytes to string and then parse as JSON
            content = json.loads(content_bytes)["data"]["result"]
            # content = content["data"]["result"]
            
            # Read JSON into pandas 
            temp_df = MetricRangeDataFrame(content)

            # Reset index
            temp_df = temp_df.reset_index()

            # Add trial_id column to roundtrip df
            temp_df["trial_id"] = trial

            # Get the run details
            run_details = get_run_details(s3_bucket, run_id, trial)

            # Get the autoscaler
            autoscaler = get_autoscaler(run_details)
            temp_df["autoscaler"] = autoscaler

            #  concat roundtrip df with existing roundtrip_df
            if metrics_df is None:
                metrics_df = temp_df
            else:
                metrics_df = pd.concat([metrics_df, temp_df], ignore_index=True)

    return metrics_df.rename(columns={"value": column_label})


def calculate_async_mean_time_by_autoscaler(async_df, column_label):
    async_df = async_df.copy()
    async_df["time"] = pd.to_datetime(async_df["timestamp"])
    async_df = async_df.sort_values(by=["trial_id", "time"])

    # group by autoscaler and trial_id and calculate the mean of the async_processing_time
    mean_time_by_trial_df = async_df.groupby(["autoscaler", "trial_id"])["value"].mean().to_frame()
    
    # # merge the mean time back to the original dataframe
    # async_df = async_df.merge(mean_time_by_trial, on="trial_id", how="left", suffixes=("", "_mean"))

    # group by autoscaler and calculate the mean of the mean_time_by_trial
    mean_time_by_autoscaler_df = mean_time_by_trial_df.groupby("autoscaler")["value"].mean().to_frame()

    # rename the value column to the column_label
    mean_time_by_autoscaler_df = mean_time_by_autoscaler_df.rename(columns={"value": column_label})

    return mean_time_by_autoscaler_df

def calculate_async_utilization_by_autoscaler(files, run_id, s3_bucket, earliest_start_time, latest_end_time, time_slice_seconds=15):
    metrics = ["kafka_consumer_idle_seconds_total-service_name-topic",
                "kafka_consumer_processing_seconds_total-service_name-topic"]
    labels = ["idle", "processing"]
    utilization_df = None
    for metric, label in zip(metrics, labels):
        print(metric, label)
        temp_df = get_merged_range_metric_df(files, run_id, s3_bucket, metric, label)
        if temp_df is None or len(temp_df) == 0:
            print(f"temp_df does not exist for metric: {metric}, label: {label}.")
            continue
        temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"], unit="s", utc=True).astype('datetime64[us, UTC]')

        
        
        # Calculate interval_range from the lowest start time to the highest start time of the dataframe
        time_slice_index = pd.interval_range(
            # start=min_timestamp,
            # end=max_timestamp,
            start=earliest_start_time,
            end=latest_end_time,
            freq=pd.Timedelta(seconds=time_slice_seconds),
            name="time_slice",
        )

        # Assign each row to a time slice based on start_time
        temp_df["time_slice"] = pd.cut(
            temp_df.timestamp,
            bins=time_slice_index,
            include_lowest=True, 
            labels=time_slice_index[:],
        )

        
        if utilization_df is None:
            utilization_df = temp_df
        else:
            # merge idle and processing times
            utilization_df = utilization_df.merge(temp_df, on=["autoscaler", "service_name", "topic", "trial_id",  "time_slice"], how="outer")
            # drop column timestamp_y
            utilization_df = utilization_df.drop(columns=["timestamp_y"])
            # rename timestamp_x to timestamp
            utilization_df = utilization_df.rename(columns={"timestamp_x": "timestamp"})
            # reorder columns to timestamp, trial_id, autoscaler, service_name, topic, time_slice, idle, processing
            utilization_df = utilization_df[["timestamp", "trial_id", "autoscaler", "service_name", "topic", "time_slice", "idle", "processing"]]
            # calculate u_t as processing/(processing + idle)
            utilization_df["u_t"] = utilization_df["processing"]/(utilization_df["processing"] + utilization_df["idle"])
    return utilization_df, time_slice_seconds


def get_run_start_and_end_times(files, s3_bucket, run_id):
    """ Retrieves the earliest benchmark start time and the latest benchmark end time.  
    Used to determine consistent time slices for merging data by time slice. """
    
    earliest_start = datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)
    latest_end = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    
    # Iterate file infos
    for file in files:
        if "run_details.json" in file["Key"]:
            # parse the file info for the trial id
            trial_id, _ = parse_run_details_file_info(file)
            # get the run details for the trial
            run_details = get_run_details(s3_bucket, run_id, trial_id, verbose=False)
            # extract the start and end time
            start = datetime.datetime.fromisoformat(run_details['status']['startTime'].replace('Z', '+00:00'))
            end = datetime.datetime.fromisoformat(run_details['status']['endTime'].replace('Z', '+00:00'))
            # compare to the current earliest start and latest end time
            earliest_start = min(earliest_start, start)
            latest_end = max(latest_end, end)
    
    return earliest_start, latest_end


def get_trial_start_and_end_times(files, s3_bucket, run_id, trial_id):
    """ Retrieves the earliest benchmark start time and the latest benchmark end time.  
    Used to determine consistent time slices for merging data by time slice. """
    
    earliest_start = datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)
    latest_end = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    
    # Iterate file infos
    for file in files:
        if "run_details.json" in file["Key"]:
            # parse the file info for the trial id
            trial_id_for_run, _ = parse_run_details_file_info(file)
            if trial_id_for_run != trial_id:
                continue
            # get the run details for the trial
            run_details = get_run_details(s3_bucket, run_id, trial_id, verbose=False)
            # extract the start and end time
            start = datetime.datetime.fromisoformat(run_details['status']['startTime'].replace('Z', '+00:00'))
            end = datetime.datetime.fromisoformat(run_details['status']['endTime'].replace('Z', '+00:00'))
            # compare to the current earliest start and latest end time
            earliest_start = min(earliest_start, start)
            latest_end = max(latest_end, end)
    
    return earliest_start, latest_end




def get_asynchronous_metrics(files, run_id, s3_bucket, earliest_start_time, latest_end_time,
    l_max=5, l_target=5, n_min=0, u_target=0.50, gamma=0.50, omega=0.50, time_slice_seconds=15):
    """Calculate the asynchronous metrics for a given run."""
    
    # Get the async processing time
    async_utilization, _ = calculate_async_utilization_by_autoscaler(files, run_id, s3_bucket, earliest_start_time, 
        latest_end_time, time_slice_seconds=time_slice_seconds)

    # Get the lag dataframe
    lag_df = get_merged_range_metric_df(files, run_id, s3_bucket, "kafka_consumer_group_lag_sum_ratio", "lag")
    
    # Get the replicas dataframe
    n_df = get_merged_range_metric_df(files, run_id, s3_bucket, "kube_deployment_status_replicas-deployment", "replicas")
    
    # Filter columns
    lag_df = lag_df[['timestamp', 'topic', 'autoscaler','trial_id', 'lag']]
    n_df = n_df[['timestamp', 'autoscaler','trial_id', 'replicas', 'deployment']]
    
    # Add column "deployment" to lag_df
    lag_df["deployment"] = np.where(lag_df.topic == "orders", "globeco-fix-engine", "globeco-confirmation-service")
    
    # Merge lag and n dataframes
    lag_df = lag_df.merge(n_df, on=["timestamp", "autoscaler", "trial_id", "deployment"], suffixes=["_lag", "_n"])
    
    # Convert timestamp to UTC 
    lag_df["timestamp"] = pd.to_datetime(lag_df["timestamp"], unit="s", utc=True).astype('datetime64[us, UTC]')
    
    # Mege lag_df and utilization on timestamp autoscaler, trial_id, and topic
    lag_df = lag_df.merge(async_utilization, on=["timestamp", "autoscaler", "trial_id", "topic"], suffixes=["_lag", "_utilization"])
    print("Displaying lag_df")
    display(lag_df)
    
    # Calculate interval_range from the ealiest to latest start time
    time_slice_index = pd.interval_range(
        start=earliest_start_time, 
        end=latest_end_time,
        freq=pd.Timedelta(seconds=time_slice_seconds),
        name="time_slice",
    )


    # Assign each row to a time slice based on start_time
    lag_df["time_slice"] = pd.cut(
        lag_df.timestamp,
        bins=time_slice_index,
        include_lowest=True, 
        labels=time_slice_index[:],
    )

    # Add "theta_a_u" column
    lag_df["theta_a_u"] = (lag_df.lag - l_max).clip(lower=0)/l_max

    # Add "tau_a_u" time-slice column
    lag_df["tau_a_u"] = np.where(lag_df.lag > l_max, 1, 0)

    # Add overprovisioned time-slice column
    lag_df["overprovisioned_ts"] = np.where(lag_df.lag == 0, 1, 0)

    # Add "lag_at_or_below_target" column (ones)
    lag_df["lag_at_or_below_target"] = np.where(lag_df.lag <= l_target, 1, 0)

    # Add "replicas_above_minimum" column (ones)
    lag_df["replicas_above_minimum"] = np.where(lag_df.replicas > n_min, 1, 0)

    # Add "overprovisioned_ratio" column
    lag_df["overprovisioned_ratio"] = np.maximum(u_target - lag_df.u_t, 0)/u_target

    # Add "theta_a_o" column
    lag_df["theta_a_o"] = lag_df["overprovisioned_ratio"] * lag_df["lag_at_or_below_target"] * lag_df["replicas_above_minimum"]

    # Add "underutilization" column (ones)
    lag_df["underutilization"] = np.where(lag_df.u_t < u_target, 1, 0)

    # Add "lag_under_target" column (ones
    lag_df["lag_under_target"] = np.where(lag_df.lag <= l_target, 1, 0)

    # Add "tau_a_o" column
    lag_df["tau_a_o"] = lag_df["overprovisioned_ts"] * lag_df["lag_at_or_below_target"] * lag_df["replicas_above_minimum"]

    agg_df = (
        lag_df
        .groupby(["autoscaler"], observed=True, dropna=False)
        .agg(
            lag_mean=("lag","mean"),
            theta_a_u= ("theta_a_u", lambda values: 100.0 * values.mean()),
            theta_a_o= ("theta_a_o", lambda values: 100.0 * values.mean()),
            tau_a_u = ("tau_a_u", lambda values: 100.0 * values.mean()),
            tau_a_o = ("tau_a_o", lambda values: 100.0 * values.mean()),
        )
        .reset_index()
    )

    agg_df

    agg_df["theta_a"] =  gamma * agg_df.theta_a_u + (1 - gamma) * agg_df.theta_a_o
    agg_df["tau_a"] =  omega * agg_df.tau_a_u + (1 - omega) * agg_df.tau_a_o
    agg_df
        
    
    return lag_df, agg_df


def get_tsdb_snapshots(files):
    """Get the tsdb snapshots from the files."""
    snapshots = set()
    for file in files:
        if match := re.search(r"(^.*tsdb-snapshots/\d{8}T\d{6}Z-[0-9a-f]{16}/)", file["Key"]):
            snapshots.add(match.group(1))
    
    # sort and return as a list
    return sorted(list(snapshots))


def get_tsdb_local_dir(run_id: str, trial_id: str) -> str:
    # Docker volume binds require an absolute host path. Resolve relative to
    # the project root (the parent of this file's directory) so the path is
    # stable regardless of the notebook's current working directory.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "data", "tsdb_snapshot", f"{run_id}-{trial_id}")


def tsdb_snapshot_exists(tsdb_local_dir: str):
    """Check if a local tsdb snapshot exists."""
    return os.path.isdir(tsdb_local_dir)
       

def execute_promql_range(
    base_url: str,
    query: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    step: str = "15s",
):
    """Execute a PromQL range query using datetime objects.

    :param base_url: Prometheus root URL (e.g. 'http://localhost:9090')
    :param query: PromQL expression string
    :param start_time: datetime object representing start of range
    :param end_time: datetime object representing end of range
    :param step: Resolution step width (e.g., '10s', '1m', '5m')
    """
    endpoint = f"{base_url}/api/v1/query_range"
    params = {
        "query": query,
        "start": start_time.timestamp(),  # Converts to float epoch seconds
        "end": end_time.timestamp(),
        "step": step,
    }

    try:
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        try:
            print("Prometheus Error Detail:", response.json().get("error"))
        except Exception:
            print(response.text)
        raise e    
    return response.json()


def parse_s3_uri(s3_uri: str):
    """Extract bucket name and prefix from an S3 URI."""
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {s3_uri}. Expected format: s3://bucket-name/path/to/snapshot")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket, prefix


def download_s3_folder(bucket_name: str, prefix: str, local_dir: str):
    """Download all objects under an S3 prefix preserving directory structure."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    print(f"Downloading s3://{bucket_name}/{prefix} to {local_dir}...")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative_path = os.path.relpath(key, prefix)
            target_path = os.path.join(local_dir, relative_path)

            if key.endswith("/"):
                os.makedirs(target_path, exist_ok=True)
                continue

            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            s3.download_file(bucket_name, key, target_path)

    # Prometheus container runs as 'nobody' (UID 65534). Ensure full read/write access.
    for root, dirs, files in os.walk(local_dir):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            os.chmod(os.path.join(root, f), 0o666)
    os.chmod(local_dir, 0o777)


def wait_for_prometheus(base_url: str, timeout: int = 45):
    """Poll Prometheus until the ready endpoint responds 200 OK."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            res = requests.get(f"{base_url}/-/ready", timeout=2)
            if res.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("Prometheus failed to become ready within the timeout period.")


def execute_promql(base_url: str, query: str):
    """Execute an instant PromQL query and return JSON response."""
    endpoint = f"{base_url}/api/v1/query"
    response = requests.get(endpoint, params={"query": query}, timeout=15)
    response.raise_for_status()
    return response.json()


def run_snapshot_analysis(s3_uri: str, queries: list[str], port: int = 9090):
    client = docker.from_env()
    temp_dir = tempfile.mkdtemp(prefix="prom_tsdb_")
    container = None
    base_url = f"http://localhost:{port}"

    try:
        bucket, prefix = parse_s3_uri(s3_uri)
        download_s3_folder(bucket, prefix, temp_dir)

        print("Starting Prometheus container...")
        container = client.containers.run(
            image="prom/prometheus:latest",
            command=[
                "--storage.tsdb.path=/prometheus",
                "--web.enable-lifecycle",
                "--storage.tsdb.retention.time=100y",
            ],
            volumes={
                temp_dir: {"bind": "/prometheus", "mode": "rw"}
            },
            ports={"9090/tcp": port},
            detach=True,
            remove=False,
        )

        wait_for_prometheus(base_url)

        print("\n--- Query Results ---")
        for q in queries:
            print(f"\nQuery: {q}")
            try:
                result = execute_promql(base_url, q)
                data = result.get("data", {}).get("result", [])
                print(f"Returned {len(data)} series/results:")
                for item in data:
                    print(item)
            except Exception as err:
                print(f"Query execution failed: {err}")

    finally:
        if container:
            print("\nStopping and removing container...")
            try:
                container.stop(timeout=5)
                container.remove(v=True)
            except Exception as e:
                print(f"Error removing container: {e}")

        if os.path.exists(temp_dir):
            print(f"Cleaning up temporary data directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


def get_metric_from_tsdb(s3_bucket, files, query, client, port, step="15s"):
    """Execute a PromQL query against the tsdb snapshots."""
    base_url = f"http://localhost:{port}"

    # Get a list of snapshot files from the full list of files
    snapshots = get_tsdb_snapshots(files)

    # Initialize the dataframe
    df = None

    # For each snapshot, download the files, start a container, execute the query, and stop the container
    for snapshot in sorted(snapshots):
        # Parse the snapshot name to get the run_id and trial_id
        run_id, trial_id = parse_snapshot_name(snapshot)

        # Get the local directory for the tsdb snapshot
        tsdb_local_dir = get_tsdb_local_dir(run_id, trial_id)

        # Download the snapshot if it doesn't already exist locally
        if not tsdb_snapshot_exists(tsdb_local_dir):
            print(f"Downloading {snapshot}...")
            download_s3_folder(s3_bucket, snapshot, tsdb_local_dir )
        
        # Prometheus requires a config file even when only serving historical
        # TSDB data for queries. Write a minimal empty config into the bind dir.
        config_path = os.path.join(tsdb_local_dir, "prometheus.yml")
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                f.write("global: {}\n")

        # Start the container
        container = client.containers.run(
            image="prom/prometheus:latest",
            command=[
                "--config.file=/prometheus/prometheus.yml",
                "--storage.tsdb.path=/prometheus",
                "--web.enable-lifecycle",
                "--storage.tsdb.retention.time=100y",
            ],
            volumes={
                tsdb_local_dir: {"bind": "/prometheus", "mode": "rw"}
            },
            ports={"9090/tcp": port},
            detach=True,
            remove=True,
        )

        # Wait for the container to be ready
        wait_for_prometheus(base_url)

        # Get the start and end times for the range query
        start_time, end_time = get_trial_start_and_end_times(files, s3_bucket, run_id, trial_id)
        
        # Execute the query
        try:
            result = execute_promql_range(base_url, query, start_time, end_time)
            data = result.get("data", {}).get("result", [])
            # print("Data: ", data)
        except Exception as err:
            raise Exception(f"Query execution failed: {err}")

        # Stop the container (will automatically delete)
        container.stop()
        
        # Convert to a dataframe with the result
        temp_df = MetricRangeDataFrame(data)

        # Add the run_id and trial_id to the dataframe
        temp_df["run_id"] = run_id  
        temp_df["trial_id"] = trial_id  

        # Get the run details
        run_details = get_run_details(s3_bucket, run_id, trial_id)

        # Get the autoscaler
        autoscaler = get_autoscaler(run_details)

        # Add the autoscaler to the dataframe
        temp_df["autoscaler"] = autoscaler


        # --- Normalize timescale to 0 ---
        
        min_ts = temp_df.index.min()
        temp_df["elapsed_seconds"] = (temp_df.index - min_ts).total_seconds()
    
        # Concatenate the dataframe with the result
        if df is None:
            df = temp_df
        else:
            df = pd.concat([df, temp_df])
    
    return df




def plot_trial_timeseries_binned(
    df,
    *,
    node=None,
    bin_seconds=30,
    confidence=0.95,
    figsize=(10, 5.5),
    xlabel="Elapsed Time (minutes)",
    ylabel="Value",
    title=None,
    trial_alpha=0.18,
    trial_linewidth=0.8,
    mean_linewidth=2.0,
    ci_alpha=0.20,
    show_trials=True,
):
    """
    Plot binned trial-level time series, across-trial mean, and
    Student's t confidence interval.

    Each (run_id, trial_id) combination is treated as an independent
    experimental replicate.

    Raw observations are first assigned to fixed-width elapsed-time
    bins. Multiple observations within the same trial/bin are averaged
    before calculating across-trial statistics. Thus, each trial
    contributes at most one observation to each time bin.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
            node, value, run_id, trial_id, elapsed_seconds

    node : str, optional
        Restrict the plot to a single node.

    bin_seconds : int or float
        Width of elapsed-time bins in seconds.

    confidence : float
        Confidence level for the Student's t interval.

    figsize : tuple
        Matplotlib figure size.

    xlabel, ylabel, title : str
        Plot labels.

    trial_alpha : float
        Transparency of individual trial trajectories.

    trial_linewidth : float
        Width of individual trial lines.

    mean_linewidth : float
        Width of the across-trial mean line.

    ci_alpha : float
        Transparency of the confidence interval.

    show_trials : bool
        Whether to show individual trial trajectories.

    Returns
    -------
    fig, ax, summary, trial_data
        Matplotlib figure and axes, across-trial summary statistics,
        and the binned trial-level data.
    """

    required = {
        "node",
        "value",
        "run_id",
        "trial_id",
        "elapsed_seconds",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataframe is missing required columns: {sorted(missing)}"
        )

    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be greater than zero.")

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    # --------------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------------

    data = df.copy()

    if node is not None:
        data = data[data["node"] == node].copy()

    data = data.dropna(
        subset=["value", "elapsed_seconds", "run_id", "trial_id"]
    )

    if data.empty:
        raise ValueError("No observations remain after filtering.")

    # Unique experimental replicate.
    #
    # Using both fields prevents trial_id values from separate runs
    # from accidentally being treated as the same trial.
    data["trial_key"] = (
        data["run_id"].astype(str)
        + "-"
        + data["trial_id"].astype(str)
    )

    # --------------------------------------------------------------
    # Assign observations to fixed-width elapsed-time bins
    # --------------------------------------------------------------
    #
    # Example for bin_seconds=30:
    #
    #   0 <= t < 30   -> bin 0
    #  30 <= t < 60   -> bin 1
    #  60 <= t < 90   -> bin 2
    #
    # We plot each bin at its midpoint:
    #
    # 15, 45, 75, ...
    #

    data["time_bin"] = np.floor(
        data["elapsed_seconds"] / bin_seconds
    ).astype(int)

    data["bin_start_seconds"] = (
        data["time_bin"] * bin_seconds
    )

    data["bin_midpoint_seconds"] = (
        data["bin_start_seconds"] + bin_seconds / 2
    )

    # --------------------------------------------------------------
    # Stage 1:
    # Aggregate observations WITHIN each trial/bin
    # --------------------------------------------------------------
    #
    # This is important statistically. A trial containing more raw
    # samples within a bin should not receive more weight than another
    # trial.
    #

    trial_data = (
        data
        .groupby(
            [
                "trial_key",
                "time_bin",
                "bin_start_seconds",
                "bin_midpoint_seconds",
            ],
            as_index=False,
        )
        .agg(
            value=("value", "mean"),
            observations=("value", "size"),
        )
        .sort_values(
            ["trial_key", "time_bin"]
        )
    )

    trial_data["elapsed_minutes"] = (
        trial_data["bin_midpoint_seconds"] / 60.0
    )

    # --------------------------------------------------------------
    # Stage 2:
    # Calculate statistics ACROSS trials for each bin
    # --------------------------------------------------------------

    summary = (
        trial_data
        .groupby(
            [
                "time_bin",
                "bin_start_seconds",
                "bin_midpoint_seconds",
            ],
            as_index=False,
        )
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            n=("value", "count"),
        )
        .sort_values("time_bin")
    )

    summary["elapsed_minutes"] = (
        summary["bin_midpoint_seconds"] / 60.0
    )

    # Standard error across experimental replicates
    summary["se"] = (
        summary["std"] / np.sqrt(summary["n"])
    )

    # Student's t critical value.
    #
    # n can differ between bins if a trial has missing data.
    summary["t_critical"] = stats.t.ppf(
        (1 + confidence) / 2,
        df=summary["n"] - 1,
    )

    # Confidence interval half-width
    summary["ci_half_width"] = (
        summary["t_critical"] * summary["se"]
    )

    summary["ci_lower"] = (
        summary["mean"] - summary["ci_half_width"]
    )

    summary["ci_upper"] = (
        summary["mean"] + summary["ci_half_width"]
    )

    # A confidence interval cannot be estimated from a single trial.
    summary.loc[
        summary["n"] < 2,
        ["ci_lower", "ci_upper"]
    ] = np.nan

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)

    # Individual trial trajectories
    if show_trials:
        for _, trial in trial_data.groupby("trial_key"):

            ax.plot(
                trial["elapsed_minutes"],
                trial["value"],
                color="0.65",
                alpha=trial_alpha,
                linewidth=trial_linewidth,
                zorder=1,
            )

    # Student's t confidence interval
    ax.fill_between(
        summary["elapsed_minutes"],
        summary["ci_lower"],
        summary["ci_upper"],
        color="0.25",
        alpha=ci_alpha,
        linewidth=0,
        label=f"{confidence:.0%} CI",
        zorder=2,
    )

    # Across-trial mean
    ax.plot(
        summary["elapsed_minutes"],
        summary["mean"],
        color="0.10",
        linewidth=mean_linewidth,
        label="Mean",
        zorder=3,
    )

    # --------------------------------------------------------------
    # Dissertation-style formatting
    # --------------------------------------------------------------

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    # Horizontal reference grid only
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.35,
    )

    ax.grid(axis="x", visible=False)

    # Remove unnecessary borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        loc="best",
    )

    ax.margins(x=0)

    fig.tight_layout()

    return fig, ax, summary, trial_data


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy import stats


def plot_cpu_allocation(
    df,
    bin_seconds=60,
    saturation_threshold=0.95,
    autoscaler_order=("none", "hpa", "vpa", "keda"),
    figsize=None,
):
    """
    Plot node CPU allocation pressure over elapsed benchmark time.

    For each trial:
      1. Bin observations by elapsed time.
      2. Compute the mean allocation ratio for each node within each bin.
      3. Take the maximum across nodes for each trial/bin.

    Across trials:
      - plot each trial as a light line
      - plot the mean as a dark line
      - plot a 95% Student's-t confidence interval around the mean

    Parameters
    ----------
    df : pandas.DataFrame
        Required columns:
            node
            value
            run_id
            trial_id
            autoscaler
            elapsed_seconds

        value is expected to be:
            requested CPU / allocatable CPU

        expressed as a ratio (e.g., 0.95 = 95%).

    bin_seconds : int
        Width of elapsed-time bins.

    saturation_threshold : float
        Ratio used to indicate severe CPU allocation pressure.

    autoscaler_order : iterable
        Desired ordering of autoscaler panels.

    figsize : tuple or None
        Figure size. If None, determined automatically.

    Returns
    -------
    fig, axes, trial_binned, summary
    """

    data = df.copy()

    # ------------------------------------------------------------------
    # 1. Basic cleanup
    # ------------------------------------------------------------------

    required = {
        "node",
        "value",
        "run_id",
        "trial_id",
        "autoscaler",
        "elapsed_seconds",
    }

    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data["elapsed_seconds"] = pd.to_numeric(
        data["elapsed_seconds"], errors="coerce"
    )

    data = data.dropna(
        subset=["value", "elapsed_seconds", "autoscaler", "trial_id", "node"]
    )

    # Unique trial identifier in case trial_id is reused across runs.
    data["trial"] = (
        data["run_id"].astype(str)
        + "/"
        + data["trial_id"].astype(str)
    )

    # ------------------------------------------------------------------
    # 2. Bin elapsed time
    # ------------------------------------------------------------------

    # Use the bin midpoint for plotting.
    data["time_bin"] = (
        np.floor(data["elapsed_seconds"] / bin_seconds) * bin_seconds
        + bin_seconds / 2
    )

    # Mean within node × trial × time bin.
    node_binned = (
        data.groupby(
            ["autoscaler", "trial", "node", "time_bin"],
            observed=True,
            as_index=False,
        )
        .agg(allocation_ratio=("value", "mean"))
    )

    # ------------------------------------------------------------------
    # 3. Maximum node allocation within each trial/time bin
    # ------------------------------------------------------------------

    trial_binned = (
        node_binned.groupby(
            ["autoscaler", "trial", "time_bin"],
            observed=True,
            as_index=False,
        )
        .agg(allocation_ratio=("allocation_ratio", "max"))
    )

    # ------------------------------------------------------------------
    # 4. Across-trial mean and Student's-t 95% CI
    # ------------------------------------------------------------------

    summary = (
        trial_binned.groupby(
            ["autoscaler", "time_bin"],
            observed=True,
        )["allocation_ratio"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["se"] = summary["std"] / np.sqrt(summary["count"])

    summary["t_crit"] = stats.t.ppf(
        0.975,
        df=summary["count"] - 1,
    )

    # CI is undefined for n < 2.
    summary.loc[summary["count"] < 2, "t_crit"] = np.nan

    summary["ci"] = summary["t_crit"] * summary["se"]
    summary["ci_low"] = summary["mean"] - summary["ci"]
    summary["ci_high"] = summary["mean"] + summary["ci"]

    # ------------------------------------------------------------------
    # 5. Determine autoscaler panels
    # ------------------------------------------------------------------

    present = set(trial_binned["autoscaler"].unique())

    autoscalers = [
        a for a in autoscaler_order
        if a in present
    ]

    # Include unexpected autoscaler names at the end.
    autoscalers += sorted(present - set(autoscalers))

    n = len(autoscalers)

    if figsize is None:
        figsize = (7.0, 2.3 * n)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    if n == 1:
        axes = np.array([axes])

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------

    for ax, autoscaler in zip(axes, autoscalers):

        trials = trial_binned[
            trial_binned["autoscaler"] == autoscaler
        ]

        agg = summary[
            summary["autoscaler"] == autoscaler
        ]

        # Individual trials
        for _, trial in trials.groupby("trial", observed=True):
            trial = trial.sort_values("time_bin")

            ax.plot(
                trial["time_bin"] / 60,
                trial["allocation_ratio"] * 100,
                linewidth=0.7,
                alpha=0.20,
            )

        # 95% Student's-t CI
        ax.fill_between(
            agg["time_bin"] / 60,
            agg["ci_low"] * 100,
            agg["ci_high"] * 100,
            alpha=0.18,
            linewidth=0,
        )

        # Across-trial mean
        ax.plot(
            agg["time_bin"] / 60,
            agg["mean"] * 100,
            linewidth=2.0,
            label="Mean",
        )

        # Saturation threshold
        ax.axhline(
            saturation_threshold * 100,
            linestyle="--",
            linewidth=1.0,
        )

        # 100% allocatable capacity
        ax.axhline(
            100,
            linestyle=":",
            linewidth=1.0,
        )

        ax.set_title(
            autoscaler.upper() if autoscaler != "none" else "No autoscaler",
            loc="left",
            fontweight="bold",
        )

        ax.set_ylabel("CPU allocation (%)")

        ax.grid(
            axis="y",
            alpha=0.20,
            linewidth=0.6,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Elapsed time (minutes)")

    fig.suptitle(
        "Maximum Node CPU Request Saturation by Autoscaler",
        fontweight="bold",
    )

    return fig, axes, trial_binned, summary

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy import stats


def plot_cpu_saturation_time_share(
    df,
    threshold=0.95,
    autoscaler_order=("none", "hpa", "vpa", "keda"),
    figsize=(7, 5),
    jitter=0.08,
):
    """
    Plot trial-level percentage of time under severe CPU request saturation.

    CPU request saturation at time t is defined as:

        max_n(
            requested_cpu_n(t) / allocatable_cpu_n(t)
        )

    A trial is considered saturated at time t when this value is >= threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Required columns:
            node
            value
            run_id
            trial_id
            autoscaler
            elapsed_seconds

        `value` is expected to be:
            requested CPU / allocatable CPU

    threshold : float
        CPU request saturation threshold.
        Default = 0.95 (95%).

    autoscaler_order : iterable
        Desired ordering of autoscaler categories.

    figsize : tuple
        Figure dimensions.

    jitter : float
        Horizontal jitter applied to trial observations.

    Returns
    -------
    fig, ax, trial_summary, autoscaler_summary
    """

    data = df.copy()

    required = {
        "node",
        "value",
        "run_id",
        "trial_id",
        "autoscaler",
        "elapsed_seconds",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data["value"] = pd.to_numeric(
        data["value"],
        errors="coerce",
    )

    data["elapsed_seconds"] = pd.to_numeric(
        data["elapsed_seconds"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "node",
            "value",
            "run_id",
            "trial_id",
            "autoscaler",
            "elapsed_seconds",
        ]
    )

    # --------------------------------------------------------------
    # 1. Maximum node CPU request saturation at each observation
    # --------------------------------------------------------------

    instantaneous = (
        data.groupby(
            [
                "run_id",
                "trial_id",
                "autoscaler",
                "elapsed_seconds",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            max_cpu_request_ratio=("value", "max")
        )
    )

    instantaneous["saturated"] = (
        instantaneous["max_cpu_request_ratio"] >= threshold
    )

    # --------------------------------------------------------------
    # 2. Trial-level saturation time share
    #
    # Because observations are regularly spaced, the proportion of
    # observations satisfying the condition equals the time share.
    # --------------------------------------------------------------

    trial_summary = (
        instantaneous.groupby(
            ["run_id", "trial_id", "autoscaler"],
            observed=True,
            as_index=False,
        )
        .agg(
            cpu_saturation_time_share=("saturated", "mean"),
            max_cpu_request_ratio=("max_cpu_request_ratio", "max"),
            mean_max_cpu_request_ratio=("max_cpu_request_ratio", "mean"),
            observations=("saturated", "size"),
        )
    )

    trial_summary["cpu_saturation_time_share_pct"] = (
        trial_summary["cpu_saturation_time_share"] * 100
    )

    # --------------------------------------------------------------
    # 3. Autoscaler-level mean and Student's-t 95% CI
    # --------------------------------------------------------------

    autoscaler_summary = (
        trial_summary.groupby(
            "autoscaler",
            observed=True,
        )["cpu_saturation_time_share_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    autoscaler_summary["se"] = (
        autoscaler_summary["std"]
        / np.sqrt(autoscaler_summary["count"])
    )

    autoscaler_summary["t_crit"] = stats.t.ppf(
        0.975,
        df=autoscaler_summary["count"] - 1,
    )

    autoscaler_summary.loc[
        autoscaler_summary["count"] < 2,
        "t_crit",
    ] = np.nan

    autoscaler_summary["ci"] = (
        autoscaler_summary["t_crit"]
        * autoscaler_summary["se"]
    )

    autoscaler_summary["ci_low"] = (
        autoscaler_summary["mean"]
        - autoscaler_summary["ci"]
    )

    autoscaler_summary["ci_high"] = (
        autoscaler_summary["mean"]
        + autoscaler_summary["ci"]
    )

    # Percentages cannot extend outside [0, 100].
    autoscaler_summary["ci_low"] = (
        autoscaler_summary["ci_low"].clip(lower=0)
    )

    autoscaler_summary["ci_high"] = (
        autoscaler_summary["ci_high"].clip(upper=100)
    )

    # --------------------------------------------------------------
    # 4. Determine plotting order
    # --------------------------------------------------------------

    present = set(trial_summary["autoscaler"].unique())

    autoscalers = [
        a for a in autoscaler_order
        if a in present
    ]

    autoscalers += sorted(
        present - set(autoscalers)
    )

    x_positions = np.arange(len(autoscalers))

    # --------------------------------------------------------------
    # 5. Plot
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=figsize,
        constrained_layout=True,
    )

    # Deterministic jitter so the figure is reproducible.
    rng = np.random.default_rng(42)

    for x, autoscaler in zip(x_positions, autoscalers):

        trials = trial_summary[
            trial_summary["autoscaler"] == autoscaler
        ]

        summary = autoscaler_summary[
            autoscaler_summary["autoscaler"] == autoscaler
        ].iloc[0]

        # Individual trials
        x_jittered = (
            x
            + rng.uniform(
                -jitter,
                jitter,
                size=len(trials),
            )
        )

        ax.scatter(
            x_jittered,
            trials["cpu_saturation_time_share_pct"],
            s=35,
            alpha=0.45,
            zorder=2,
            label=None,
        )

        # Mean
        ax.scatter(
            x,
            summary["mean"],
            s=90,
            marker="D",
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )

        # 95% Student's-t CI
        if np.isfinite(summary["ci"]):
            ax.errorbar(
                x,
                summary["mean"],
                yerr=[
                    [summary["mean"] - summary["ci_low"]],
                    [summary["ci_high"] - summary["mean"]],
                ],
                fmt="none",
                capsize=5,
                linewidth=1.5,
                zorder=3,
            )

    # --------------------------------------------------------------
    # 6. Formatting
    # --------------------------------------------------------------

    labels = [
        "No autoscaler" if a == "none" else a.upper()
        for a in autoscalers
    ]

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)

    ax.set_ylabel(
        f"Time at ≥{threshold * 100:.0f}% CPU request saturation (%)"
    )

    ax.set_xlabel("Autoscaler")

    ax.set_title(
        f"Trial-Level Time at ≥{threshold * 100:.0f}% "
        "Maximum Node CPU Request Saturation",
        fontweight="bold",
    )

    ax.set_ylim(-2, 102)

    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.6,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        fig,
        ax,
        trial_summary,
        autoscaler_summary,
    )



def prepare_cpu_requested_vs_usage(
    cluster_cpu_requested_df,
    cluster_cpu_usage_df,
    bin_seconds=60,
):
    """
    Merge requested and actual CPU ratios and identify the most
    CPU-request-saturated node at each trial/time.

    Both input `value` columns are assumed to be ratios relative to
    allocatable CPU:
        1.0 = 100% of allocatable CPU.

    Returns
    -------
    matched : DataFrame
        Node-level matched requested/usage observations.

    pressure : DataFrame
        At each trial/time, the node with the greatest requested CPU ratio.

    binned : DataFrame
        60-second (by default) means of pressure-node requested/usage ratios.
    """

    requested = cluster_cpu_requested_df.copy()
    usage = cluster_cpu_usage_df.copy()

    keys = [
        "run_id",
        "trial_id",
        "autoscaler",
        "node",
        "elapsed_seconds",
    ]

    requested["value"] = pd.to_numeric(
        requested["value"], errors="coerce"
    )
    usage["value"] = pd.to_numeric(
        usage["value"], errors="coerce"
    )

    requested["elapsed_seconds"] = pd.to_numeric(
        requested["elapsed_seconds"], errors="coerce"
    )
    usage["elapsed_seconds"] = pd.to_numeric(
        usage["elapsed_seconds"], errors="coerce"
    )

    requested = requested.dropna(
        subset=keys + ["value"]
    )
    usage = usage.dropna(
        subset=keys + ["value"]
    )

    # Rename before merge.
    requested = requested[keys + ["value"]].rename(
        columns={"value": "requested_ratio"}
    )

    usage = usage[keys + ["value"]].rename(
        columns={"value": "usage_ratio"}
    )

    # ----------------------------------------------------------
    # Match node-for-node and time-for-time
    # ----------------------------------------------------------

    matched = requested.merge(
        usage,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    matched["reservation_gap"] = (
        matched["requested_ratio"]
        - matched["usage_ratio"]
    )

    # ----------------------------------------------------------
    # At each timestamp, select the node under greatest
    # scheduling pressure: max requested / allocatable.
    # ----------------------------------------------------------

    group_keys = [
        "run_id",
        "trial_id",
        "autoscaler",
        "elapsed_seconds",
    ]

    idx = (
        matched.groupby(
            group_keys,
            observed=True,
        )["requested_ratio"]
        .idxmax()
    )

    pressure = (
        matched.loc[idx]
        .sort_values(group_keys)
        .reset_index(drop=True)
    )

    # Unique trial identifier in case trial_id is reused.
    pressure["trial"] = (
        pressure["run_id"].astype(str)
        + "/"
        + pressure["trial_id"].astype(str)
    )

    # ----------------------------------------------------------
    # Bin for visualization
    # ----------------------------------------------------------

    pressure["time_bin"] = (
        np.floor(
            pressure["elapsed_seconds"] / bin_seconds
        ) * bin_seconds
        + bin_seconds / 2
    )

    binned = (
        pressure.groupby(
            [
                "run_id",
                "trial_id",
                "trial",
                "autoscaler",
                "time_bin",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            requested_ratio=("requested_ratio", "mean"),
            usage_ratio=("usage_ratio", "mean"),
        )
    )

    binned["reservation_gap"] = (
        binned["requested_ratio"]
        - binned["usage_ratio"]
    )

    return matched, pressure, binned


def summarize_cpu_time_series(cpu_binned):
    """
    Calculate across-trial mean and Student's-t 95% CI for
    requested and actual CPU ratios.
    """

    long = cpu_binned.melt(
        id_vars=[
            "run_id",
            "trial_id",
            "trial",
            "autoscaler",
            "time_bin",
        ],
        value_vars=[
            "requested_ratio",
            "usage_ratio",
        ],
        var_name="metric",
        value_name="ratio",
    )

    summary = (
        long.groupby(
            ["autoscaler", "time_bin", "metric"],
            observed=True,
        )["ratio"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["se"] = (
        summary["std"]
        / np.sqrt(summary["count"])
    )

    summary["t_crit"] = stats.t.ppf(
        0.975,
        df=summary["count"] - 1,
    )

    summary.loc[
        summary["count"] < 2, "t_crit"
    ] = np.nan

    summary["ci"] = (
        summary["t_crit"] * summary["se"]
    )

    summary["ci_low"] = (
        summary["mean"] - summary["ci"]
    )

    summary["ci_high"] = (
        summary["mean"] + summary["ci"]
    )

    return summary


def plot_cpu_requested_vs_usage(
    cpu_binned,
    autoscaler_order=("none", "hpa", "vpa", "keda"),
    figsize=None,
):
    summary = summarize_cpu_time_series(cpu_binned)

    present = set(summary["autoscaler"].unique())

    autoscalers = [
        a for a in autoscaler_order
        if a in present
    ]
    autoscalers += sorted(
        present - set(autoscalers)
    )

    n = len(autoscalers)

    if figsize is None:
        figsize = (7.2, 2.35 * n)

    fig, axes = plt.subplots(
        n,
        1,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    if n == 1:
        axes = np.array([axes])

    # Use matplotlib defaults rather than hard-coded colors.
    default_colors = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    metric_styles = {
        "requested_ratio": {
            "label": "Requested",
            "color": default_colors[0],
        },
        "usage_ratio": {
            "label": "Actual usage",
            "color": default_colors[1],
        },
    }

    for ax, autoscaler in zip(
        axes, autoscalers
    ):

        subset = summary[
            summary["autoscaler"] == autoscaler
        ]

        for metric, style in metric_styles.items():

            s = subset[
                subset["metric"] == metric
            ].sort_values("time_bin")

            x = s["time_bin"] / 60

            # Confidence interval
            ax.fill_between(
                x,
                s["ci_low"] * 100,
                s["ci_high"] * 100,
                color=style["color"],
                alpha=0.12,
                linewidth=0,
            )

            # Mean
            ax.plot(
                x,
                s["mean"] * 100,
                color=style["color"],
                linewidth=2.0,
                label=style["label"],
            )

        # 95% scheduling-pressure reference
        ax.axhline(
            95,
            linestyle="--",
            linewidth=1.0,
        )

        # 100% allocatable CPU
        ax.axhline(
            100,
            linestyle=":",
            linewidth=1.0,
        )

        title = (
            "No autoscaler"
            if autoscaler == "none"
            else autoscaler.upper()
        )

        ax.set_title(
            title,
            loc="left",
            fontweight="bold",
        )

        ax.set_ylabel(
            "Allocatable CPU (%)"
        )

        ax.grid(
            axis="y",
            alpha=0.20,
            linewidth=0.6,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(
        frameon=False,
        ncol=2,
        loc="upper left",
    )

    axes[-1].set_xlabel(
        "Elapsed time (minutes)"
    )

    fig.suptitle(
        "Requested and Actual CPU on the Most "
        "Request-Saturated Node",
        fontweight="bold",
    )

    return fig, axes, summary


def plot_cpu_reservation_gap(
    cpu_pressure,
    autoscaler_order=("none", "hpa", "vpa", "keda"),
    figsize=(7, 5),
    jitter=0.08,
):
    """
    Trial-level mean difference between requested CPU and
    actual CPU usage on the most request-saturated node.
    """

    trial_summary = (
        cpu_pressure.groupby(
            ["run_id", "trial_id", "autoscaler"],
            observed=True,
            as_index=False,
        )
        .agg(
            mean_requested_ratio=(
                "requested_ratio", "mean"
            ),
            mean_usage_ratio=(
                "usage_ratio", "mean"
            ),
            mean_reservation_gap=(
                "reservation_gap", "mean"
            ),
        )
    )

    trial_summary["reservation_gap_pct"] = (
        trial_summary["mean_reservation_gap"]
        * 100
    )

    # ----------------------------------------------------------
    # Autoscaler summary
    # ----------------------------------------------------------

    summary = (
        trial_summary.groupby(
            "autoscaler",
            observed=True,
        )["reservation_gap_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["se"] = (
        summary["std"]
        / np.sqrt(summary["count"])
    )

    summary["t_crit"] = stats.t.ppf(
        0.975,
        df=summary["count"] - 1,
    )

    summary.loc[
        summary["count"] < 2,
        "t_crit",
    ] = np.nan

    summary["ci"] = (
        summary["t_crit"]
        * summary["se"]
    )

    # ----------------------------------------------------------
    # Plotting order
    # ----------------------------------------------------------

    present = set(
        trial_summary["autoscaler"].unique()
    )

    autoscalers = [
        a for a in autoscaler_order
        if a in present
    ]

    autoscalers += sorted(
        present - set(autoscalers)
    )

    x_positions = np.arange(
        len(autoscalers)
    )

    fig, ax = plt.subplots(
        figsize=figsize,
        constrained_layout=True,
    )

    rng = np.random.default_rng(42)

    # ----------------------------------------------------------
    # Plot individual trials + mean/CI
    # ----------------------------------------------------------

    for x, autoscaler in zip(
        x_positions,
        autoscalers,
    ):

        trials = trial_summary[
            trial_summary["autoscaler"]
            == autoscaler
        ]

        s = summary[
            summary["autoscaler"]
            == autoscaler
        ].iloc[0]

        x_jittered = (
            x
            + rng.uniform(
                -jitter,
                jitter,
                len(trials),
            )
        )

        # Individual trials
        ax.scatter(
            x_jittered,
            trials["reservation_gap_pct"],
            s=35,
            alpha=0.45,
            zorder=2,
        )

        # Mean
        ax.scatter(
            x,
            s["mean"],
            s=90,
            marker="D",
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )

        # Student's-t 95% CI
        if np.isfinite(s["ci"]):
            ax.errorbar(
                x,
                s["mean"],
                yerr=s["ci"],
                fmt="none",
                capsize=5,
                linewidth=1.5,
                zorder=3,
            )

    labels = [
        (
            "No autoscaler"
            if a == "none"
            else a.upper()
        )
        for a in autoscalers
    ]

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)

    ax.axhline(
        0,
        linewidth=0.8,
        linestyle=":",
    )

    ax.set_xlabel("Autoscaler")

    ax.set_ylabel(
        "Requested − actual CPU\n"
        "(percentage points of allocatable CPU)"
    )

    ax.set_title(
        "Trial-Level CPU Reservation Gap",
        fontweight="bold",
    )

    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.6,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig, ax, trial_summary, summary


def download_locust_dbs(s3_bucket, run_id, data_directory):
    local_db_paths = []

    # Get the file listing
    files = get_s3_file_listing(s3_bucket, s3_prefix=run_id)

    # Iterate through the files and download the locust databases
    for file_info in files:
        # skip non-database files
        if not is_locust_db(file_info):
            continue
        
        # parse trial, role, and filename from the S3 file info
        trial, role, filename = parse_locust_db_file_info(file_info)

        # skip if it-operations.  It operations transactions are not tracked for response time/failure rate
        if role == "it-operations":
            continue
        
        # download the database if it doesn't already exist
        local_db_path = download_locust_db(data_directory, s3_bucket, file_info["Key"], trial, role, filename)
        
        local_db_paths.append(local_db_path)

    return local_db_paths


def create_run_db(run_id, run_db_path, s3_bucket, local_db_paths):
    # Delete the log database if it already exists
    if os.path.exists(run_db_path):
        os.remove(run_db_path)

    # Create the log database
    create_log_database(run_db_path)

    # Populate the log database from the locust database files
    for local_db_path in local_db_paths:
        
        # User regex to parse trial and role from '../data/calibration-15-percentile-a/trial0001/db/back-office/back-office.db'
        match = re.match(r'.*/([^/]+)/db/(.*)/(.*\.db)', local_db_path)
        trial_id = match.group(1)
        role = match.group(2)   
        # print(f"{local_db_path} - {trial_id} - {role}")

        # skip if it-operations.  It operations transactions are not tracked for response time/failure rate
        if role == "it-operations":
            continue

        # get the autoscaler from the run details
        run_details = get_run_details(s3_bucket, run_id, trial_id)
        autoscaler = get_autoscaler(run_details)
        
        # load as a dataframe, augmented with run id, trial id, and role
        df = get_raw_log_as_dataframe(local_db_path)
        df["run_id" ] = run_id
        df["trial_id" ] = trial_id
        df["role" ] = role
        df["autoscaler" ] = autoscaler
        df["success"] = (
            pd.to_numeric(df["status_code"], errors="coerce")
            .between(200, 299)
            )
        df["failure"] = ~df["success"]

        # append to the sqlite database
        df.to_sql('logs', sqlite3.connect(run_db_path), if_exists='append', index=False)
        

def get_images(s3_bucket, run_id, trial_id, verbose=False):
    """Get the run details from S3"""
    prefixes = ["pre", "post"]
    images = []
    for prefix in prefixes:
        key = f"{run_id}/{trial_id}/images/{prefix}-images.json"
        obj = s3.get_object(Bucket=s3_bucket, Key=key)
        image = json.loads(obj['Body'].read())
        images.append(image)
    return images


def get_start_end_time(s3_bucket, run_id, verbose=False):
    """Get the run details from S3"""
    key = f"{run_id}/start_time.txt"
    obj = s3.get_object(Bucket=s3_bucket, Key=key)
    start_time = obj['Body'].read()
    start_time = start_time.decode('utf-8')
    start_time = start_time.replace("T", " ")
    start_time = start_time.replace("Z", "")
    
    key = f"{run_id}/end_time.txt"
    obj = s3.get_object(Bucket=s3_bucket, Key=key)
    end_time = obj['Body'].read()
    end_time = end_time.decode('utf-8')
    end_time = end_time.replace("T", " ")
    end_time = end_time.replace("Z", "")
    
    return start_time, end_time


def get_benchmark_images_df(s3_bucket, run_id, trial_id, verbose=False):
    """Get listing of helm charts used in the benchmark"""
    
    # Get the environment from S3 and extract the controller image
    key = f"{run_id}/environment.json"
    obj = s3.get_object(Bucket=s3_bucket, Key=key)
    environment = json.loads(obj['Body'].read())
    controller_image = f"{environment['KASBENCH_IMAGE_NAME']}:{environment['KASBENCH_IMAGE_TAGS']}"
    controller_image_id = environment["KASBENCH_IMAGE_ID"]
    controller = {"Image": controller_image, "ImageId": controller_image_id}
    
    
    # Load {run_id}/{trial_id}/kasbench-runner.json from S3
    key = f"{run_id}/{trial_id}/images/kasbench-runner.json"
    obj = s3.get_object(Bucket=s3_bucket, Key=key)
    kasbench_runner_details = json.loads(obj['Body'].read())
    kasbench_runner_image = kasbench_runner_details["imageName"]
    kasbench_runner_image_id = kasbench_runner_details["imageId"]
    kasbench_runner = {"Image": kasbench_runner_image, "ImageId": kasbench_runner_image_id}

    # Load {run_id}/{trial_id}/load_runner_image.json from S3
    key = f"{run_id}/{trial_id}/images/load_runner_image.json"
    obj = s3.get_object(Bucket=s3_bucket, Key=key)
    load_generator_details = json.loads(obj['Body'].read())
    load_generator_image = load_generator_details["image"]
    load_generator_image_id = load_generator_details["imageId"]
    load_generator = {"Image": load_generator_image, "ImageId": load_generator_image_id}

    benchmark_images = {
        "controller": controller,
        "runner": kasbench_runner,
        "load_generator": load_generator,
    }
    benchmark_images_df = pd.DataFrame(benchmark_images)
    benchmark_images_df.columns = ["KASBench Controller", "KASBench Runner", "KASBench Load Generator"]
    benchmark_images_df = benchmark_images_df.T
    benchmark_images_df.columns = ["Image_original", "ImageID"]
    
    benchmark_images_df = benchmark_images_df.reset_index()
    benchmark_images_df = benchmark_images_df.rename(columns={"index": "Component"})

    if verbose:
        display(benchmark_images_df)

    return benchmark_images_df


def get_globeco_helm_version():
    """Get the latest version of the globeco-helm chart"""

    helm_repo_name: str = "globeco-repo"
    helm_repo_url: str = "https://kasbench.github.io/globeco-helm"
    # Query github to get the latest version of the globeco-helm chart
    cmd = f"helm repo add {helm_repo_name} {helm_repo_url}"
    subprocess.run(cmd.split(), check=True, capture_output=True)
    cmd = f"helm search repo {helm_repo_name}/globeco --versions"
    result = subprocess.run(cmd.split(), check=True, capture_output=True, text=True)
    # The result is a table.  The second line is the latest version
    lines = result.stdout.split('\n')
    version = lines[1].split()[1]
    return version


def get_helm_versions(s3_bucket, run_id, verbose=False):
    """ Get the helm chart versions from the S3 bucket """
    
    helm_df = None
    helm_charts = {}

    # Get all the files for the run
    files = get_s3_file_listing(s3_bucket, run_id)
    for file in files:
        # select the roundtrip json results
        if file["Key"].endswith("helm_versions.json"):
            
            # Fetch object content directly from S3 using boto3
            obj = s3.get_object(Bucket=s3_bucket, Key=file["Key"])
            helm_details = json.loads(obj['Body'].read())
            
            for chart in helm_details["charts"]:
                helm_charts[chart["name"]] = {
                    "Name": chart["name"],
                    "Version": chart["version"],
                    "URL": chart["url"]
                }

    helm_df = pd.DataFrame(helm_charts).T
    return helm_df


def download_and_load_locust_dbs(s3_bucket, run_id, run_db, data_dir):
    files = get_s3_file_listing(s3_bucket, s3_prefix=run_id)

    # Delete the log database if it already exists
    if os.path.exists(run_db):
        os.remove(run_db)

    # Create the log database
    create_log_database(run_db)

    # Populate the log database from the locust database files
    for file_info in files:
        # skip non-database files
        if not is_locust_db(file_info):
            continue
        
        # parse trial, role, and filename from the S3 file info
        trial, role, filename = parse_locust_db_file_info(file_info)

        # get the autoscaler from the run details
        run_details = get_run_details(s3_bucket, run_id, trial)
        autoscaler = get_autoscaler(run_details)

        # skip if it-operations.  It operations transactions are not tracked for response time/failure rate
        if role == "it-operations":
            continue
        
        # download the database if it doesn't already exist
        local_path = download_locust_db(data_dir, s3_bucket, file_info["Key"], trial, role, filename)
        
        # load as a dataframe, augmented with run id, trial id, and role
        df = get_raw_log_as_dataframe(local_path)
        df["run_id" ] = run_id
        df["trial_id" ] = trial
        df["role" ] = role
        df["autoscaler" ] = autoscaler
        df["success"] = (
            pd.to_numeric(df["status_code"], errors="coerce")
            .between(200, 299)
            )
        df["failure"] = ~df["success"]

        # append to the sqlite database
        df.to_sql('logs', sqlite3.connect(run_db), if_exists='append', index=False)
        


def plot_mean_response_time_by_autoscaler(
    trial_summary: pd.DataFrame,
    *,
    autoscaler_order: list[str] | None = None,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    filename: str = "../figures/response_time_by_autoscaler.png",
    show: bool = False,
    include_title: bool = False,
) -> tuple[plt.Figure, plt.Axes, str]:
    """
    Plot trial-level mean response times by autoscaler.

    Each small point represents one trial. The larger point represents the
    unweighted mean of the trial response times. Error bars show a two-sided
    Student-t confidence interval for the mean.

    Parameters
    ----------
    trial_summary:
        DataFrame containing:
          - autoscaler
          - trial_id
          - mean_response_time

    autoscaler_order:
        Optional display order. Autoscalers not listed here are appended
        alphabetically.

    confidence_level:
        Confidence level for the error bars. Default is 0.95.

    random_seed:
        Seed used to make horizontal point jitter reproducible.

    filename:
        Destination path for the saved figure.

    show:
        Whether to display the plot interactively.

    include_title:
        Whether to render the axis title.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, str]
        The generated figure, axes, and saved file path.
    """
    if autoscaler_order is None:
        autoscaler_order = ["none", "hpa", "vpa", "keda"]

    required_columns = {"autoscaler", "trial_id", "mean_response_time"}
    missing_columns = required_columns - set(trial_summary.columns)

    if missing_columns:
        raise ValueError(
            f"trial_summary is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    plot_df = trial_summary[
        ["autoscaler", "trial_id", "mean_response_time"]
    ].copy()

    plot_df["mean_response_time"] = pd.to_numeric(
        plot_df["mean_response_time"],
        errors="coerce",
    )

    if plot_df["mean_response_time"].isna().any():
        bad_rows = plot_df.loc[plot_df["mean_response_time"].isna()]
        raise ValueError(
            "mean_response_time contains missing or nonnumeric values. "
            f"Invalid rows:\n{bad_rows}"
        )

    present_autoscalers = sorted(plot_df["autoscaler"].dropna().unique())

    if autoscaler_order is None:
        autoscalers = present_autoscalers
    else:
        autoscalers = [
            autoscaler
            for autoscaler in autoscaler_order
            if autoscaler in present_autoscalers
        ]
        autoscalers.extend(
            autoscaler
            for autoscaler in present_autoscalers
            if autoscaler not in autoscalers
        )

    rng = np.random.default_rng(random_seed)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    alpha = 1 - confidence_level
    summary_rows = []

    for position, autoscaler in enumerate(autoscalers):
        autoscaler_df = plot_df.loc[plot_df["autoscaler"] == autoscaler]
        response_times = autoscaler_df["mean_response_time"].to_numpy()

        # Reproducible horizontal jitter so overlapping trial points remain visible
        jittered_x = rng.normal(
            loc=position,
            scale=0.045,
            size=len(response_times),
        )

        ax.scatter(
            jittered_x,
            response_times,
            s=45,
            alpha=0.65,
            label="Individual trial" if position == 0 else None,
            zorder=2,
        )

        number_of_trials = len(response_times)
        mean_rt = response_times.mean()

        if number_of_trials >= 2:
            standard_deviation = response_times.std(ddof=1)
            standard_error = standard_deviation / np.sqrt(number_of_trials)
            critical_value = t.ppf(
                1 - alpha / 2,
                df=number_of_trials - 1,
            )
            confidence_interval_half_width = critical_value * standard_error
        else:
            standard_deviation = np.nan
            standard_error = np.nan
            confidence_interval_half_width = np.nan

        ax.errorbar(
            position,
            mean_rt,
            yerr=confidence_interval_half_width,
            fmt="o",
            markersize=9,
            capsize=7,
            capthick=1.5,
            linewidth=2,
            label=(
                f"Mean and {confidence_level:.0%} CI"
                if position == 0
                else None
            ),
            zorder=3,
        )

        summary_rows.append(
            {
                "autoscaler": autoscaler,
                "trials": number_of_trials,
                "mean_response_time": mean_rt,
                "sd_response_time": standard_deviation,
                "se_response_time": standard_error,
                "ci_lower": (
                    max(0, mean_rt - confidence_interval_half_width)
                    if number_of_trials >= 2
                    else np.nan
                ),
                "ci_upper": (
                    mean_rt + confidence_interval_half_width
                    if number_of_trials >= 2
                    else np.nan
                ),
            }
        )

    ax.set_xticks(range(len(autoscalers)))
    ax.set_xticklabels([autoscaler.upper() for autoscaler in autoscalers])

    ax.set_xlabel("Autoscaler")
    ax.set_ylabel("Mean response time (ms)")
    if include_title:
        ax.set_title(
            "Mean Response Time by Autoscaler\n"
            "Trial-level means with overall mean and "
            f"{confidence_level:.0%} confidence interval"
        )

    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()

    ax.response_time_summary = pd.DataFrame(summary_rows)

    if show:
        plt.show()

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    return fig, ax, filename



def plot_failure_rate_by_autoscaler(
    trial_summary: pd.DataFrame,
    *,
    autoscaler_order: list[str] | None = None,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    filename: str = "../figures/failure_rate_by_autoscaler.png",
    show: bool = False,
    include_title: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot trial-level failure rates by autoscaler.

    Each small point represents one trial. The larger point represents the
    unweighted mean of the trial failure rates. Error bars show a two-sided
    Student-t confidence interval for the mean.

    Parameters
    ----------
    trial_summary:
        DataFrame containing:
          - autoscaler
          - trial_id
          - failure_rate

        failure_rate must be expressed as a proportion, e.g. 0.025 for 2.5%.

    autoscaler_order:
        Optional display order. Autoscalers not listed here are appended
        alphabetically.

    confidence_level:
        Confidence level for the error bars. Default is 0.95.

    random_seed:
        Seed used to make horizontal point jitter reproducible.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The generated figure and axes.
    """

    if autoscaler_order is None:
        autoscaler_order=["none", "hpa", "vpa", "keda"]

    required_columns = {"autoscaler", "trial_id", "failure_rate"}
    missing_columns = required_columns - set(trial_summary.columns)

    if missing_columns:
        raise ValueError(
            f"trial_summary is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    plot_df = trial_summary[
        ["autoscaler", "trial_id", "failure_rate"]
    ].copy()

    plot_df["failure_rate"] = pd.to_numeric(
        plot_df["failure_rate"],
        errors="coerce",
    )

    if plot_df["failure_rate"].isna().any():
        bad_rows = plot_df.loc[plot_df["failure_rate"].isna()]
        raise ValueError(
            "failure_rate contains missing or nonnumeric values. "
            f"Invalid rows:\n{bad_rows}"
        )

    if not plot_df["failure_rate"].between(0, 1).all():
        bad_rows = plot_df.loc[
            ~plot_df["failure_rate"].between(0, 1)
        ]
        raise ValueError(
            "failure_rate must be between 0 and 1. "
            f"Invalid rows:\n{bad_rows}"
        )

    present_autoscalers = sorted(
        plot_df["autoscaler"].dropna().unique()
    )

    if autoscaler_order is None:
        autoscalers = present_autoscalers
    else:
        autoscalers = [
            autoscaler
            for autoscaler in autoscaler_order
            if autoscaler in present_autoscalers
        ]

        autoscalers.extend(
            autoscaler
            for autoscaler in present_autoscalers
            if autoscaler not in autoscalers
        )

    rng = np.random.default_rng(random_seed)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    alpha = 1 - confidence_level
    summary_rows = []

    for position, autoscaler in enumerate(autoscalers):
        autoscaler_df = plot_df.loc[
            plot_df["autoscaler"] == autoscaler
        ]

        rates_percent = (
            autoscaler_df["failure_rate"].to_numpy() * 100
        )

        # Add slight horizontal jitter so overlapping trial points remain visible.
        jittered_x = rng.normal(
            loc=position,
            scale=0.045,
            size=len(rates_percent),
        )

        ax.scatter(
            jittered_x,
            rates_percent,
            s=45,
            alpha=0.65,
            label="Individual trial" if position == 0 else None,
            zorder=2,
        )

        number_of_trials = len(rates_percent)
        mean_rate = rates_percent.mean()

        if number_of_trials >= 2:
            standard_deviation = rates_percent.std(ddof=1)
            standard_error = (
                standard_deviation / np.sqrt(number_of_trials)
            )

            critical_value = t.ppf(
                1 - alpha / 2,
                df=number_of_trials - 1,
            )

            confidence_interval_half_width = (
                critical_value * standard_error
            )
        else:
            standard_deviation = np.nan
            standard_error = np.nan
            confidence_interval_half_width = np.nan

        ax.errorbar(
            position,
            mean_rate,
            yerr=confidence_interval_half_width,
            fmt="o",
            markersize=9,
            capsize=7,
            capthick=1.5,
            linewidth=2,
            label=(
                f"Mean and {confidence_level:.0%} CI"
                if position == 0
                else None
            ),
            zorder=3,
        )

        summary_rows.append(
            {
                "autoscaler": autoscaler,
                "trials": number_of_trials,
                "mean_failure_rate": mean_rate / 100,
                "sd_failure_rate": standard_deviation / 100,
                "se_failure_rate": standard_error / 100,
                "ci_lower": (
                    max(
                        0,
                        mean_rate
                        - confidence_interval_half_width,
                    )
                    / 100
                    if number_of_trials >= 2
                    else np.nan
                ),
                "ci_upper": (
                    min(
                        100,
                        mean_rate
                        + confidence_interval_half_width,
                    )
                    / 100
                    if number_of_trials >= 2
                    else np.nan
                ),
            }
        )

    ax.set_xticks(range(len(autoscalers)))
    ax.set_xticklabels(
        [autoscaler.upper() for autoscaler in autoscalers]
    )

    ax.set_xlabel("Autoscaler")
    ax.set_ylabel("Failure rate (%)")
    if include_title:
        ax.set_title(
            "Failure Rate by Autoscaler\n"
            "Trial-level rates with mean and "
            f"{confidence_level:.0%} confidence interval"
        )

    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()

    # The summary can be retrieved from the axes for later use if desired.
    ax.failure_rate_summary = pd.DataFrame(summary_rows)

    if show:
        plt.show()
    
    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    return fig, ax, filename

def plot_roundtrip_completion_by_autoscaler(
    trial_summary: pd.DataFrame,
    *,
    autoscaler_order: list[str] | None = None,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    filename: str = "../figures/roundtrip_completion_by_autoscaler.png",
    show: bool = False,
    include_title: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot trial-level round-trip completion percentages by autoscaler.

    Each small point represents one trial. The larger point represents the
    unweighted mean of the trial percentages. Error bars show a two-sided
    Student-t confidence interval for the mean.

    Parameters
    ----------
    trial_summary:
        DataFrame containing:
          - autoscaler
          - trial_id
          - roundtrip_completion_percentage

        roundtrip_completion_percentage must be expressed in percentage
        points, such as 9.3 for 9.3%, rather than 0.093.

    autoscaler_order:
        Optional display order. Autoscalers present in the dataframe but not
        listed here are appended alphabetically.

    confidence_level:
        Confidence level for the error bars. Defaults to 0.95.

    random_seed:
        Seed used to make the horizontal jitter reproducible.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The generated figure and axes.
    """

    if autoscaler_order is None:
        autoscaler_order = ["none", "hpa", "vpa", "keda"]

    required_columns = {
        "autoscaler",
        "trial_id",
        "roundtrip_completion_percentage",
    }
    missing_columns = required_columns - set(trial_summary.columns)

    if missing_columns:
        raise ValueError(
            "trial_summary is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    plot_df = trial_summary[
        [
            "autoscaler",
            "trial_id",
            "roundtrip_completion_percentage",
        ]
    ].copy()

    if plot_df["autoscaler"].isna().any():
        raise ValueError("autoscaler contains missing values.")

    if plot_df["trial_id"].isna().any():
        raise ValueError("trial_id contains missing values.")

    plot_df["roundtrip_completion_percentage"] = pd.to_numeric(
        plot_df["roundtrip_completion_percentage"],
        errors="coerce",
    )

    if plot_df["roundtrip_completion_percentage"].isna().any():
        bad_rows = plot_df.loc[
            plot_df["roundtrip_completion_percentage"].isna()
        ]
        raise ValueError(
            "roundtrip_completion_percentage contains missing or "
            f"nonnumeric values:\n{bad_rows}"
        )

    if not plot_df["roundtrip_completion_percentage"].between(
        0, 100
    ).all():
        bad_rows = plot_df.loc[
            ~plot_df["roundtrip_completion_percentage"].between(0, 100)
        ]
        raise ValueError(
            "roundtrip_completion_percentage must be between 0 and 100. "
            f"Invalid rows:\n{bad_rows}"
        )

    # Prevent a trial from being counted more than once for an autoscaler.
    duplicate_mask = plot_df.duplicated(
        subset=["autoscaler", "trial_id"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_rows = plot_df.loc[
            duplicate_mask,
            ["autoscaler", "trial_id"],
        ].sort_values(["autoscaler", "trial_id"])

        raise ValueError(
            "Each autoscaler/trial_id combination must appear exactly once. "
            f"Duplicate rows:\n{duplicate_rows}"
        )

    present_autoscalers = sorted(
        plot_df["autoscaler"].astype(str).unique()
    )

    if autoscaler_order is None:
        autoscalers = present_autoscalers
    else:
        autoscalers = [
            autoscaler
            for autoscaler in autoscaler_order
            if autoscaler in present_autoscalers
        ]

        autoscalers.extend(
            autoscaler
            for autoscaler in present_autoscalers
            if autoscaler not in autoscalers
        )

    if not autoscalers:
        raise ValueError("No autoscaler data is available to plot.")

    rng = np.random.default_rng(random_seed)
    alpha = 1 - confidence_level

    fig, ax = plt.subplots(figsize=(8, 5.5))

    summary_rows: list[dict[str, float | int | str]] = []

    for position, autoscaler in enumerate(autoscalers):
        autoscaler_df = plot_df.loc[
            plot_df["autoscaler"] == autoscaler
        ].sort_values("trial_id")

        percentages = autoscaler_df[
            "roundtrip_completion_percentage"
        ].to_numpy(dtype=float)

        number_of_trials = len(percentages)

        # Slight horizontal jitter keeps overlapping trial points visible.
        jittered_x = rng.normal(
            loc=position,
            scale=0.045,
            size=number_of_trials,
        )

        ax.scatter(
            jittered_x,
            percentages,
            s=45,
            alpha=0.65,
            label="Individual trial" if position == 0 else None,
            zorder=2,
        )

        mean_percentage = percentages.mean()

        if number_of_trials >= 2:
            standard_deviation = percentages.std(ddof=1)
            standard_error = (
                standard_deviation / np.sqrt(number_of_trials)
            )

            critical_value = t.ppf(
                1 - alpha / 2,
                df=number_of_trials - 1,
            )

            confidence_interval_half_width = (
                critical_value * standard_error
            )

            confidence_interval_lower = max(
                0.0,
                mean_percentage - confidence_interval_half_width,
            )
            confidence_interval_upper = min(
                100.0,
                mean_percentage + confidence_interval_half_width,
            )

            # Asymmetric errors allow confidence bounds to be clipped to
            # the logically valid 0%-100% interval.
            lower_error = mean_percentage - confidence_interval_lower
            upper_error = confidence_interval_upper - mean_percentage
            yerr = np.array([[lower_error], [upper_error]])
        else:
            standard_deviation = np.nan
            standard_error = np.nan
            critical_value = np.nan
            confidence_interval_half_width = np.nan
            confidence_interval_lower = np.nan
            confidence_interval_upper = np.nan
            yerr = None

        ax.errorbar(
            position,
            mean_percentage,
            yerr=yerr,
            fmt="o",
            markersize=9,
            capsize=7,
            capthick=1.5,
            linewidth=2,
            label=(
                f"Mean and {confidence_level:.0%} CI"
                if position == 0
                else None
            ),
            zorder=3,
        )

        summary_rows.append(
            {
                "autoscaler": autoscaler,
                "trials": number_of_trials,
                "mean_roundtrip_completion_percentage": mean_percentage,
                "sd_roundtrip_completion_percentage": standard_deviation,
                "se_roundtrip_completion_percentage": standard_error,
                "t_critical": critical_value,
                "ci_half_width": confidence_interval_half_width,
                "ci_lower": confidence_interval_lower,
                "ci_upper": confidence_interval_upper,
            }
        )

    ax.set_xticks(range(len(autoscalers)))
    ax.set_xticklabels(
        [autoscaler.upper() for autoscaler in autoscalers]
    )

    ax.set_xlabel("Autoscaler")
    ax.set_ylabel("Round-trip completion percentage (%)")

    if include_title:
        ax.set_title(
            "Round-Trip Completion by Autoscaler\n"
            "Trial-level percentages with mean and "
            f"{confidence_level:.0%} confidence interval"
        )

    # Start at zero because the metric has a meaningful zero.
    # Let Matplotlib choose the upper bound unless the values approach 100%.
    ax.set_ylim(bottom=0)

    ax.grid(
        axis="y",
        alpha=0.25,
    )
    ax.legend()

    fig.tight_layout()

    # Attach the statistics used in the chart for convenient retrieval.
    ax.roundtrip_completion_summary = pd.DataFrame(summary_rows)

    if show:
        plt.show()

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    return fig, ax, filename


def images_to_html(images_df):
    df = images_df.copy()

    

    # Build the combined Image / ImageID cell.
    # Escape values so characters such as &, <, and > remain valid HTML.
    df["Image"] = df.apply(
        lambda row: (
            f"{escape(str(row['Image_original']))}"
            f"<br>"
            f"<span class=\"image-id\">{escape(str(row['ImageID']))}</span>"
        ),
        axis=1
    )

    # Generate the HTML table.
    html = df[["Component", "Image"]].to_html(
        index=False,
        escape=False,
        header=True,
        classes="data-table compact",
        border=0
    )

    # html = html.replace("Deployment_display", "Deployment")
    # html = html.replace("Image_display", "Image")

    return html



def merged_trial_summary_df_to_html(merged_trial_summary_df):
    """Returns the merge summary dataframe as a HTML table."""
    df = merged_trial_summary_df.copy()

    df = df[
        [
            "trial_id",
            "autoscaler",
            "requests",
            "mean_response_time",
            "failures",
            "failure_rate",
            "roundtrip_completion_percentage",
        ]
    ]

    df.columns = [
        "Trial ID",
        "Autoscaler",
        "Requests",
        "Mean Response Time (ms)",
        "Failures",
        "Failure Rate",
        "Round-trip Completion (%)",
    ]

    # Clean the column axis name BEFORE converting to Styler
    df.columns.name = None

    # Sort by Trial ID
    df = df.sort_values(by=["Trial ID"])

    # Format columns (DataFrame level)
    df["Failure Rate"] = df["Failure Rate"].apply(lambda x: f"{x:.3%}")
    df["Round-trip Completion (%)"] = df["Round-trip Completion (%)"].apply(
        lambda x: f"{x:.1f}%"
    )

    # Initialize Styler
    style = df.style.format({"Mean Response Time (ms)": "{:.0f}"})

    # Apply properties and styles
    style = style.set_properties(**{"text-align": "center"})
    style = style.set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    )

    # Capitalize the autoscaler column (all caps)
    style = style.set_properties(
        subset=["Autoscaler"], **{"text-transform": "uppercase"}
    )

    # Hide the index on the Styler object
    style = style.hide(axis="index")

    # Generate the HTML table using Styler.to_html options
    html = style.to_html(
        escape=False,
        encoding="utf-8",  # replacing structural parameters not supported by styler
    )

    # If you need to inject custom classes or borders into the <table> tag,
    # it is safest to do it on the final string or via set_table_attributes
    style = style.set_table_attributes('class="data-table compact" border="0"')
    html = style.to_html(escape=False)


    return html


def autoscaler_summary_df_to_html(autoscaler_summary_df):
    """Returns the merge summary dataframe as a HTML table."""
    df = autoscaler_summary_df.copy()

    df = df[
        [
            "autoscaler",
            "mean_rt",
            "mean_failure_rate",
            "mean_roundtrip_completion_percentage",
        ]
    ]

    df.columns = [
        "Autoscaler",
        "Mean Response Time (ms)",
        "Failure Rate",
        "Round-trip Completion (%)",
    ]

    # Clean the column axis name BEFORE converting to Styler
    df.columns.name = None

    # # Sort by Trial ID
    # df = df.sort_values(by=["Trial ID"])

    # Format columns (DataFrame level)
    df["Failure Rate"] = df["Failure Rate"].apply(lambda x: f"{x:.3%}")
    df["Round-trip Completion (%)"] = df["Round-trip Completion (%)"].apply(
        lambda x: f"{x:.1f}%"
    )

    # Initialize Styler
    style = df.style.format({"Mean Response Time (ms)": "{:.0f}"})

    # Apply properties and styles
    style = style.set_properties(**{"text-align": "center"})
    style = style.set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    )

    # Capitalize the autoscaler column (all caps)
    style = style.set_properties(
        subset=["Autoscaler"], **{"text-transform": "uppercase"}
    )

    # Hide the index on the Styler object
    style = style.hide(axis="index")

    # Generate the HTML table using Styler.to_html options
    html = style.to_html(
        escape=False,
        encoding="utf-8",  # replacing structural parameters not supported by styler
    )

    # If you need to inject custom classes or borders into the <table> tag,
    # it is safest to do it on the final string or via set_table_attributes
    style = style.set_table_attributes('class="data-table compact" border="0"')
    html = style.to_html(escape=False)


    return html

