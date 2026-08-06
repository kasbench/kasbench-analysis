import io
import os
import sqlite3
import json

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
