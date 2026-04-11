import modal
import subprocess

app = modal.App("download-dataset-robust")
vol = modal.Volume.from_name("flashinfer-trace")

# 配置带有 git-lfs 的基础镜像
image = modal.Image.debian_slim().apt_install("git", "git-lfs")

# 关键：加上 timeout=3600，防止下载中途被平台“谋杀”
@app.function(volumes={"/data": vol}, image=image, timeout=3600)
def download_to_volume():
    print("1. 清理云盘旧文件...")
    subprocess.run("rm -rf /data/*", shell=True)
    
    print("2. 开始克隆数据集 (大概需要几分钟，请耐心等待它跑完)...")
    subprocess.run(["git", "lfs", "install"], check=True)
    # 克隆到临时文件夹
    subprocess.run(["git", "clone", "https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest", "/data/tmp_dataset"], check=True)
    
    print("3. 把数据提取到正确的根目录...")
    # 把 workloads 文件夹提出来
    subprocess.run("mv /data/tmp_dataset/* /data/", shell=True)
    subprocess.run("rm -rf /data/tmp_dataset", shell=True)
    
    print("4. 检查现在的云盘内容 (如果你看到 workloads 文件夹说明成功了)：")
    subprocess.run("ls -la /data", shell=True)
    
    # 将更改持久化保存到 Volume
    vol.commit()
    print("5. 完美保存！数据集已就绪。")

@app.local_entrypoint()
def main():
    download_to_volume.remote()