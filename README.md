## Book
- [Pythonで学ぶ画像認識](https://book.impress.co.jp/books/1122101074)
- [GitHub](https://github.com/py-img-recog/python_image_recognition)
- Purchase Date: 230927 (DRMfree)

<img src="https://img.ips.co.jp/ij/22/1122101074/1122101074-520x.jpg" height="150px">

## CheckList
- [ ] **第1章　画像認識とは？**  
  - [x] 第1節　画像認識の概要  
  - [ ] 第2節　コンピュータに画像認識の仕組みを理解しよう  
  - [ ] 第3節　実社会で使われている画像認識アプリケーション  
  - [x] 第4節　画像認識のための開発環境構築
- [ ] **第2章　画像処理の基礎知識**  
  - [x] 第1節　画像データを読み込んで表示してみよう  
  - [ ] 第2節　画像に平滑化フィルタをかけてみよう  
  - [ ] 第3節　畳み込み演算を使った特徴抽出  
  - [ ] 第4節　アテンションを使った特徴抽出  
- [ ] **第3章　深層学習を使う準備**  
  - [ ] 第1節　学習と評価の基礎  
  - [ ] 第2節　深層ニューラルネットワーク
- [ ] **第4章　画像分類**  
  - [ ] 第1節　順伝播型ニューラルネットワークによる手法  
  - [ ] 第2節　畳み込みニューラルネットワークによる手法 —ResNet18を実装してみよう—  
  - [ ] 第3節　Transformerによる手法 —Vision Transformerを実装してみよう—  
  - [ ] 第4節　精度向上の工夫と評価  
- [ ] **第5章　物体検出**  
  - [ ] 第1節　物体検出の基礎  
  - [ ] 第2節　データセットの準備  
  - [ ] 第3節　CNNによる手法 —RetinaNetを実装してみよう—  
  - [ ] 第4節　Transformerによる手法 —DETRを実装してみよう—
- [ ] **第6章　画像キャプショニング**  
  - [ ] 第1節　画像キャプショニングの基礎  
  - [ ] 第2節　データセットの準備  
  - [ ] 第3節　CNN-LSTMによる手法 —Show, attend and tellを実装してみよう—  
  - [ ] 第4節　アテンション機構による手法 —Show, attend and tellを実装してみよう—  
  - [ ] 第5節　Transformerによる画像キャプショニングを実装してみよう

## Docker + WSL2 + CUDA or MPS の環境構築
- WSL2上でないと、DockerからCUDAを使えない
- Docker Desktop において幾つか設定をする（必須かどうかは要検証）
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) をインストール
- 各種コマンドで環境確認
  ```
  % lsb_release -a
  No LSB modules are available.
  Distributor ID: Ubuntu
  Description:    Ubuntu 24.04.3 LTS
  Release:        24.04
  Codename:       noble

  % nvidia-smi
  Mon Sep 22 14:24:58 2025
  +-----------------------------------------------------------------------------------------+
  | NVIDIA-SMI 575.64.01              Driver Version: 576.80         CUDA Version: 12.9     |
  |-----------------------------------------+------------------------+----------------------+
  | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
  |                                         |                        |               MIG M. |
  |=========================================+========================+======================|
  |   0  NVIDIA GeForce RTX 4070        On  |   00000000:01:00.0  On |                  N/A |
  |  0%   52C    P0             36W /  200W |    3081MiB /  12282MiB |     10%      Default |
  |                                         |                        |                  N/A |
  +-----------------------------------------+------------------------+----------------------+

  +-----------------------------------------------------------------------------------------+
  | Processes:                                                                              |
  |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
  |        ID   ID                                                               Usage      |
  |=========================================================================================|
  |    0   N/A  N/A              31      G   /Xwayland                             N/A      |
  |    0   N/A  N/A              41      G   /Xwayland                             N/A      |
  +-----------------------------------------------------------------------------------------+

  % docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
  Unable to find image 'nvidia/cuda:12.2.0-base-ubuntu22.04' locally
  12.2.0-base-ubuntu22.04: Pulling from nvidia/cuda
  aece8493d397: Pull complete
  9fe5ccccae45: Pull complete
  8054e9d6e8d6: Pull complete
  bdddd5cb92f6: Pull complete
  5324914b4472: Pull complete
  Digest: sha256:ecdf8549dd5f12609e365217a64dedde26ecda26da8f3ff3f82def6749f53051
  Status: Downloaded newer image for nvidia/cuda:12.2.0-base-ubuntu22.04
  Mon Sep 22 05:29:48 2025
  +-----------------------------------------------------------------------------------------+
  | NVIDIA-SMI 575.64.01              Driver Version: 576.80         CUDA Version: 12.9     |
  |-----------------------------------------+------------------------+----------------------+
  | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
  |                                         |                        |               MIG M. |
  |=========================================+========================+======================|
  |   0  NVIDIA GeForce RTX 4070        On  |   00000000:01:00.0  On |                  N/A |
  |  0%   39C    P0             34W /  200W |    3259MiB /  12282MiB |     13%      Default |
  |                                         |                        |                  N/A |
  +-----------------------------------------+------------------------+----------------------+

  +-----------------------------------------------------------------------------------------+
  | Processes:                                                                              |
  |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
  |        ID   ID                                                               Usage      |
  |=========================================================================================|
  |    0   N/A  N/A              31      G   /Xwayland                             N/A      |
  |    0   N/A  N/A              41      G   /Xwayland                             N/A      |
  +-----------------------------------------------------------------------------------------+

  ```
  - VSCode なら右下に「CRLF」と表示されている箇所をクリック → LF に変更して保存
  - Compose V2 でも WSL2 では run --gpus がサポートされない場合がある？
  - Debian/Ubuntu 系の Python 3.12 では システム Python に対して pip で直接パッケージをインストールすることを制限している？
  - Ubuntu 24.04 + Python3 系では、python コマンドは存在せず python3 が正しい？
  - WSL2 側の NVIDIA ドライバは CUDA 13.0 以上に対応していない？