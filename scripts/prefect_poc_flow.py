from pathlib import Path

from prefect import flow, task

HAMO_DIR = Path("/scratch/kaushim07/vesta_data/04_geometry_tables_dsk256_110825/hamo")


@task
def count_hamo_parquets() -> int:
    return len(list(HAMO_DIR.glob("*.parquet")))


@flow(name="hamo-parquet-count-poc")
def poc_flow() -> int:
    count = count_hamo_parquets()
    print(f"Current hamo/ parquet count: {count}")
    return count


if __name__ == "__main__":
    poc_flow()
