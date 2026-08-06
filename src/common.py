from __future__ import annotations

import io
import os
import sqlite3
import json
from dataclasses import dataclass


import boto3
import pandas as pd
import numpy as np
from scipy.stats import t



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


def is_locust_db(file_info):
    """Check if the file info refers to a locust database file"""
    s3_key = file_info["Key"]
    is_db_path = s3_key.split("/")[-2] == "db" 
    return is_db_path and s3_key.endswith(".db")

def parse_locust_db_file_info(file_info):
    """Parse a locust DB key to extract the experiment name"""
    s3_key = file_info["Key"]
    parts = s3_key.split("/")
    trial, filename = parts[-3], parts[-1]
    role = filename.split(".")[0]
    return trial, role, filename

def parse_roundtrip_file_info(file_info):
    """Parse a locust DB key to extract the experiment name"""
    s3_key = file_info["Key"]
    parts = s3_key.split("/")
    trial, filename = parts[-3], parts[-1]
    return trial, filename

    

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
    obj = boto3.client('s3').get_object(Bucket=s3_bucket, Key=key)
    return json.loads(obj['Body'].read())


def get_autoscaler(run_details):
    """ Get the autoscaler from the run details"""
    autoscaler = run_details["initialization"]["autoscaler"]
    return autoscaler    


def get_trial_summary_df(df):
    """ Computes aggregates for each trial """
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


def merge_roundtrip_completion_percentage(files, s3_bucket, trial_summary_df):
        
    roundtrip_df = None

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


def append_time_slice(df, time_slice_seconds=30):
    """Append a time_slice column to the dataframe."""
    
    # Make a copy of the dataframe
    df = df.copy()

    # Convert start_time to datetime and subtract the minimum start_time to get a timedelta
    df["start_time_dt"] = pd.to_datetime(df.start_time)
    min_start = df.start_time_dt.min()
    df.start_time_dt = df.start_time_dt - min_start
    min_start = min_start - min_start
    print(f"min_start = {min_start}")

    # Calculate the maximum start time
    max_start = df.start_time_dt.max()

    # Increment max_start by one time slice
    max_start = max_start + pd.Timedelta(seconds=time_slice_seconds)
    print(f"max_start = {max_start}")

    # Calculate interval_range from the lowest start time to the highest start time of the dataframe
    time_slice_index = pd.interval_range(
        start=min_start,
        end=max_start,
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
        time_slices["mean_response_time_ms"] > slo_rt_ms
    ).astype(np.int8)

    # Equivalent to max(sgn(SLO_RT - RT_t), 0).
    time_slices["indicator_o_rt"] = (
        time_slices["mean_response_time_ms"] < slo_rt_ms
    ).astype(np.int8)

    # Equivalent to max(sgn(FR_t - SLO_FR), 0).
    time_slices["indicator_u_fr"] = (
        time_slices["failure_rate"] > slo_fr
    ).astype(np.int8)

    # Equivalent to max(sgn(SLO_FR - FR_t), 0).
    time_slices["indicator_o_fr"] = (
        time_slices["failure_rate"] < slo_fr
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
