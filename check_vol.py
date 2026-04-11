import modal
import subprocess

app = modal.App("debug-dataset")
vol = modal.Volume.from_name("flashinfer-trace")

@app.function(volumes={"/data": vol})
def check_path():
    print("--- 正在扫描云端硬盘目录结构 ---")
    # 打印前两层的所有文件夹
    subprocess.run("find /data -maxdepth 2 -type d", shell=True)
    print("--------------------------------")

@app.local_entrypoint()
def main():
    check_path.remote()