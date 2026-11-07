# lewm-resolution-collapse 交接

这个仓库做的是规划诊断，不是从头训练 LeWM。世界模型用的是官方已发布权重。代码已经在 GitHub，本文件说明新机器还要补的数据。

## 代码

模型结构在 `jepa.py`、`module.py`。`spt_compat.py` 用来加载 checkpoint，不需要安装完整的 `stable_pretraining`。环境和 CEM 来自 `stable_worldmodel`。

`pretrained.py` 从 `$STABLEWM_HOME/hf_<env>/` 读权重。未设置时 `$STABLEWM_HOME` 是 `~/.stable_worldmodel`。每个目录里要有 `config.json` 和 `weights.pt`。

## 权重和数据都不在 git 里

官方权重（四个环境都要）：

- https://huggingface.co/quentinll/lewm-tworooms → `~/.stable_worldmodel/hf_tworooms/`
- https://huggingface.co/quentinll/lewm-pusht → `~/.stable_worldmodel/hf_pusht/`
- https://huggingface.co/quentinll/lewm-reacher → `~/.stable_worldmodel/hf_reacher/`
- https://huggingface.co/quentinll/lewm-cube → `~/.stable_worldmodel/hf_cube/`

目录名是 `hf_tworooms`（带 s），Hugging Face 仓库名是 `lewm-tworooms`。下载后把 `config.json` 和 `weights.pt` 放进上面的目录。

专家数据是 HDF5，来自同一作者的 [LeWM Hugging Face collection](https://huggingface.co/collections/quentinll/lewm)。解压后按下面的路径放。`stable_worldmodel` 的读取规则是 `$STABLEWM_HOME/datasets/<名字>.h5`，名字可以带子目录：

| 环境 | 文件 |
|---|---|
| PushT | `~/.stable_worldmodel/datasets/pusht_expert_train.h5` |
| TwoRoom | `~/.stable_worldmodel/datasets/tworoom.h5` |
| Reacher | `~/.stable_worldmodel/datasets/dmc/reacher_random.h5` |
| Cube | `~/.stable_worldmodel/datasets/ogbench/cube_single_expert.h5` |

这四个文件合计约 244GB。旧服务器上的副本在 `/cephfs/zhangheting/persistent/datasets/stable_worldmodel/`，只在那台机器的 cephfs 还挂着时才拷得到。

本仓库附录里的 RND 网络和线性探针在 https://huggingface.co/TingheOliver/lewm-resolution-collapse ，体积很小。

## 环境

和另外两个项目共用。见 `OliverZ-dot/wm-lab`。关键包：`torch==2.6.0+cu124`、`transformers==5.15.0`（`pretrained.py` 里的 ViT key 重映射是按这个版本写的）、`stable_worldmodel` 用本地的 `swm_src`，不要装一个对不上的 PyPI 版本盖掉它。`wm-lab` 的安装脚本会把 `swm_src` 放到 `sys.path`。

## 接着做

数值和曲线已经在 `outputs/runs/`，不重跑也能画图：

```bash
python analyze_rank_fidelity.py
python plot_paper_resolution.py
```

要重跑某个环境：

```bash
python exp_resolution_curve.py --env tworoom
python explore_train.py exploration.name=disagreement
```

先确认对应的 `hf_<env>` 和 HDF5 都在。`explore_train.py` 会自己采数据再训练，仍然要能 import `stable_worldmodel` 并创建环境。
