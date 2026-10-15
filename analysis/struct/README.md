# allin1 原始輸出（P1）

`song.json` 為 allin1 原始輸出（path, bpm, beats, downbeats, beat_positions, segments），未含 activations/embeddings。

## 環境（`~/creative/env-allin1`，CPU only，Python 3.11）
實際可行的流程（與計畫書不同處：NATTEN 0.17.5 沒有舊 API，改用 PR #39）：
```bash
mkdir -p ~/creative/env-allin1 && cd ~/creative/env-allin1
uv venv -p 3.11 && source .venv/bin/activate
uv pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install "numpy<2" setuptools wheel cython
uv pip install git+https://github.com/CPJKU/madmom        # @27f032e8947204902c675e5e341a3faf5dc86dae
uv pip install allin1 librosa "numpy<2"
# PR #39（純 PyTorch NATTEN 替代；jdf/all-in-one 分支 torch-natten-fallback）
uv pip install "git+https://github.com/jdf/all-in-one@3646cafab37c8a3002463a015fb7b231d30944dd"
```
- NATTEN 0.17.5 CPU wheel（`https://shi-labs.com/natten/wheels/cpu/torch2.6.0/natten-0.17.5%2Btorch260cpu-cp311-cp311-linux_x86_64.whl`；該站憑證已過期，需 `curl -k`）可安裝但 (1) 匯入時 `get_device_cc()` 在無 CUDA 下丟錯，(2) 0.17.5 已無 allin1 需要的 `natten1dav` 等舊 API。故最終未使用 NATTEN，已移除。
- 實際執行是 PR #39 的純 PyTorch 實作（未對原始碼額外打 patch）。

關鍵版本：torch 2.6.0+cpu, torchaudio 2.6.0+cpu, allin1 1.1.0 (PR #39 @3646caf), madmom 0.17.dev0 (@27f032e), numpy 1.26.4, demucs 4.1.0, librosa 0.11.0, scipy 1.17.1。natten：未安裝。

## 執行
```bash
cd ~/creative/env-allin1/work && source ../.venv/bin/activate
allin1 ~/creative/Video_Generation/audio/song.wav -o ~/creative/Video_Generation/analysis/struct -d cpu --demix-dir work/demix --spec-dir work/spec
```
耗時約 58 秒（wall，16 核 CPU，含 demucs 分離）。demix/spec 副產物在 `~/creative/env-allin1/work/work/`，不在 repo。

`audio/song.wav` sha256: `bb720091527a3d01f6a52165a21690924162ef1e02877f5ed8e50149797a6108`

## 結果摘要
bpm 107；232 beats、58 downbeats；首拍 2.22 s、首 downbeat 2.79 s；downbeat 每 4 拍一次，無例外。
注意：約 85.97–87.31 s 與 111.93–112.72 s 附近拍距異常（0.44/0.8 s，標準約 0.56 s），需人工聽檢。
