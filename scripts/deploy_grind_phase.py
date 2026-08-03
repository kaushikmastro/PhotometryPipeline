from prefect import flow

if __name__ == "__main__":
    grind_phase = flow.from_source(
        source="/home/kaushim07/photometry_mcmc_env/scripts",
        entrypoint="prefect_grind_phase_flow.py:grind_phase",
    )
    grind_phase.deploy(
        name="grind-phase",
        work_pool_name="geometry-grind",
    )
