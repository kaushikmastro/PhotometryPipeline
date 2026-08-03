import re
import subprocess
import time

from prefect import flow, get_run_logger, task

TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
    "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED",
}


@task
def submit_and_monitor_slurm_job(script: str, slice_file: str, dependency: str = "", timeout: int = 20 * 3600, poll_interval: int = 60) -> dict:
    logger = get_run_logger()

    sbatch_cmd = ["sbatch"]
    if dependency:
        sbatch_cmd.append(f"--dependency={dependency}")
    sbatch_cmd += [script, slice_file]

    result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    job_id = match.group(1)
    logger.info("submitted job %s for slice=%s (script=%s)", job_id, slice_file, script)

    waited = 0
    state, exit_code = "", ""
    while True:
        out = subprocess.run(
            ["sacct", "-j", job_id, "-X", "-n", "-P", "--format=State,ExitCode"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        state, exit_code = out.split("|")
        state = state.strip()
        if state in TERMINAL_STATES:
            break
        if waited >= timeout:
            raise TimeoutError(f"job {job_id} ({slice_file}) still {state} after {timeout}s")
        time.sleep(poll_interval)
        waited += poll_interval

    record = {"job_id": job_id, "slice_file": slice_file, "state": state, "exit_code": exit_code, "waited_s": waited}
    logger.info("job %s (%s) finished: %s", job_id, slice_file, record)

    if state != "COMPLETED" or not exit_code.startswith("0:"):
        raise RuntimeError(f"job {job_id} ({slice_file}) ended {state}, exit={exit_code}")
    return record


@flow(name="grind-phase")
def grind_phase(phase: str, slice_files: list[str], script: str, dependency: str = "") -> dict:
    logger = get_run_logger()

    futures = [submit_and_monitor_slurm_job.submit(script, sf, dependency) for sf in slice_files]

    results, failures = [], []
    for f in futures:
        try:
            results.append(f.result())
        except Exception as e:
            failures.append(str(e))

    logger.info("phase=%s: %d/%d succeeded", phase, len(results), len(slice_files))

    if failures:
        raise RuntimeError(f"{phase}: {len(failures)} slice job(s) failed: {failures}")
    return {"phase": phase, "n_slices": len(slice_files), "results": results}
