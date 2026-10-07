#!/usr/bin/env bash
# VESSL workspace 첫 설정 (저장소 루트에서 실행: bash scripts/setup.sh)
set -e
cd "$(dirname "$0")/.."
pip install -q -r requirements.txt
# 공식 평가 코드 (mF1 은 이 코드로만 계산)
if [ ! -d third_party/ocelot23algo ]; then
  git clone --depth 1 https://github.com/lunit-io/ocelot23algo third_party/ocelot23algo
fi
mkdir -p data checkpoints results
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
df -h /root 2>/dev/null || df -h .
echo "OK. 데이터는 data/ 아래에 압축 해제 후: python -m src.dataset --data_root data/<폴더>"
