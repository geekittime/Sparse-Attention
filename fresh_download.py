import subprocess

import modal


DATASET_REPO = "https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest"
TRACE_SET_PATH = "/data"
TMP_DATASET_PATH = "/tmp/mlsys26-contest"

app = modal.App("refresh-flashinfer-trace")
volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .apt_install("git", "git-lfs", "ca-certificates")
    .env({"GIT_LFS_SKIP_SMUDGE": "0"})
)


def run_cmd(cmd, cwd=None):
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


@app.function(volumes={TRACE_SET_PATH: volume}, image=image, timeout=7200)
def refresh_dataset():
    print("Refreshing Modal volume 'flashinfer-trace' with latest dataset...", flush=True)

    run_cmd(["git", "lfs", "install", "--skip-repo"])
    run_cmd(["rm", "-rf", TMP_DATASET_PATH])
    run_cmd(["git", "clone", DATASET_REPO, TMP_DATASET_PATH])
    run_cmd(["git", "lfs", "pull"], cwd=TMP_DATASET_PATH)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=TMP_DATASET_PATH,
        text=True,
    ).strip()
    print(f"Dataset commit: {commit}", flush=True)

    # Remove both normal and hidden entries from the mounted volume root.
    subprocess.run(
        f"find {TRACE_SET_PATH} -mindepth 1 -maxdepth 1 -exec rm -rf {{}} +",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"cp -a {TMP_DATASET_PATH}/. {TRACE_SET_PATH}/",
        shell=True,
        check=True,
    )
    with open(f"{TRACE_SET_PATH}/DATASET_COMMIT.txt", "w", encoding="utf-8") as f:
        f.write(commit + "\n")

    print("Top-level dataset layout:", flush=True)
    subprocess.run(["find", TRACE_SET_PATH, "-maxdepth", "2", "-type", "d"], check=True)
    print("DSA baseline files, if present:", flush=True)
    subprocess.run(
        f"find {TRACE_SET_PATH}/solutions/baseline -maxdepth 2 -type f 2>/dev/null | "
        "grep -E 'flashinfer_(deepgemm_wrapper_2ba145|wrapper_5af199)' || true",
        shell=True,
        check=True,
    )

    volume.commit()
    print("Dataset refresh complete and committed to Modal volume.", flush=True)


@app.local_entrypoint()
def main():
    refresh_dataset.remote()
