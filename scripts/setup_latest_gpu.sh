set -ex

# python -m pip install --upgrade --no-cache-dir pip

uv pip install --no-cache-dir packaging ninja numpy pandas ipython ipykernel gdown wheel setuptools
# This has to be pinned for VLLM to work.
uv pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-cache-dir flash-attn --no-build-isolation
uv pip install --no-cache-dir vllm

git clone https://github.com/volcengine/verl
cd verl
uv pip install --no-cache-dir -e .
cd ..

uv pip install --no-cache-dir -e .[dev,agent]
# Upgrade agentops to the latest version
uv pip install --no-cache-dir -U agentops
