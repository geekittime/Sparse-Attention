import modal
import subprocess

app = modal.App("final-fix-dataset")
vol = modal.Volume.from_name("flashinfer-trace")

@app.function(volumes={"/data": vol})
def fix_structure():
    print("把 workloads 里的赛道数据提取到根目录...")
    # 把 workloads 里的所有子文件夹（gated_delta_net 等）移到外层
    subprocess.run("mv /data/workloads/* /data/", shell=True, check=False)
    vol.commit()
    print("修复完成！现在的云盘根目录有：")
    subprocess.run("ls -la /data", shell=True)

@app.local_entrypoint()
def main():
    fix_structure.remote()