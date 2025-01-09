import pandas as pd
import psutil
import os


def ram_resources(rig):
    process = psutil.Process(os.getpid())  # Get current process
    mem_info = process.memory_info()  # Memory usage details

    RSS = f"RSS (Resident Set Size): {mem_info.rss / 1024 ** 2:.2f} MB"  # Actual memory in RAM
    VMS = f"VMS (Virtual Memory Size): {mem_info.vms / 1024 ** 2:.2f} MB"  # Total virtual memory

    return dict(RSS=RSS, VMS=VMS)


def averages(rig):
    # Convert 'total_infer_time' to numeric (if necessary)
    a = pd.to_numeric(rig.db_unknown.df["total_infer_time"], errors='coerce')

    # Calculate the total inference time sum and mean
    avg_infer_time = a.sum() / len(rig.db_unknown.df)  # Average time

    # Calculate the average errors
    avg_errors = rig.db_unknown.df["is_error"].map({"True": 1, "False": 0}).mean()

    return dict(avg_infer_time=avg_infer_time, avg_errors=avg_errors)


def metadata(rig):
    ram_r = ram_resources(rig)
    agents_data = str(rig.agents_manager)
    average_r = averages(rig)
    return dict(ram_usage=ram_r, agents_data=agents_data, averages=average_r)
