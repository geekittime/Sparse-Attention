import modal
import subprocess

app = modal.App("download-dataset")
# 绑定云端硬盘
vol = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

@app.function(volumes={"/data": vol})
def download_to_volume():
    print("开始在 Modal 云端下载数据集...")
    # 安装云端环境所需的 git 和 lfs
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "git", "git-lfs"], check=True)
    subprocess.run(["git", "lfs", "install"], check=True)
    
    # 直接拉取到云端硬盘的挂载目录中
    subprocess.run(["git", "clone", "https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest", "/data/mlsys26-contest"], check=True)
    
    # 将更改持久化保存到 Volume
    vol.commit()
    print("下载并保存完成！数据集已经在你的 Modal 云端准备就绪。")

@app.local_entrypoint()
def main():
    download_to_volume.remote()