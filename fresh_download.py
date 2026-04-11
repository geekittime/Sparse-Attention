import modal
import subprocess

app = modal.App("fresh-download-dataset")
vol = modal.Volume.from_name("flashinfer-trace")

# 准备包含 git-lfs 的环境
image = modal.Image.debian_slim().apt_install("git", "git-lfs")

@app.function(volumes={"/data": vol}, image=image, timeout=3600)
def download_and_organize():
    print("1. 确保云盘是空的...")
    subprocess.run("rm -rf /data/*", shell=True)
    
    print("2. 开始极速下载官方数据集 (大约需要2-3分钟)...")
    subprocess.run(["git", "lfs", "install"], check=True)
    # 先克隆到一个临时文件夹
    subprocess.run(["git", "clone", "https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest", "/data/temp_repo"], check=True)
    
    print("3. 正在将数据提取到根目录...")
    # 把临时文件夹里的所有东西移到外层根目录
    subprocess.run("mv /data/temp_repo/* /data/", shell=True)
    # 删除没用的临时文件夹
    subprocess.run("rm -rf /data/temp_repo", shell=True)
    
    print("4. 下载和整理完成！现在的云盘根目录结构如下：")
    subprocess.run("ls -la /data", shell=True)
    
    # 提交保存到云盘
    vol.commit()
    print("5. 完美搞定！可以开始测速了！")

@app.local_entrypoint()
def main():
    download_and_organize.remote()