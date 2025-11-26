### 代码库概览

该代码库的主要目的是模拟和优化在毯子被掀开后重新覆盖人体的任务。该过程使用经过训练的图神经网络（GNN）来预测不同动作的结果，并使用优化器来寻找最佳动作。

以下是工作流程的细分：
1.  **入口点**：运行模拟的关键脚本是 `code/run_robe_sim_new_opt.py`。
2.  **模拟环境**：物理和交互由位于 `assistive-gym-fem/assistive_gym/envs/robe_bm_reversible.py` 中的 `RobeReversibleEnv` 环境处理。
3.  **核心逻辑**：模拟是一个两阶段过程：
    *   **揭开 (Uncover)**：首先，模拟一个“揭开”动作。脚本从一个 `.pkl` 文件中加载之前揭开模拟的结果，以设置初始状态。
    *   **重新覆盖 (Re-cover)**：然后，脚本使用随机搜索优化器（`gradient_free_optimizers.RandomSearchOptimizer`）来寻找最佳的重新覆盖动作。这个动作由预训练的 GNN 模型的预测指导。
4.  **奖励计算**：重新覆盖动作的成功通过 `assistive-gym-fem/assistive_gym/envs/bu_gnn_util_re.py` 中复杂的奖励系统进行衡量。它创建一个人体密集点云，并根据身体的哪些部位（目标与非目标）被正确覆盖来计算分数。

### 理解关键脚本

-   `code/run_robe_sim_new_opt.py`：**这是您应该使用的脚本。** 它是执行重新覆盖任务的功能性、正确入口点。
-   `code/run_robe_sim_joint_opt.py`：**应忽略此脚本。** 它是一个不完整或旧版本，并且在缺少加载初始“揭开”状态的逻辑时会引发 `NameError`。

### 如何重现工作

要运行重新覆盖模拟，您将使用 `code/run_robe_sim_new_opt.py`。该脚本需要预训练的模型和“揭开”结果数据集作为起始点。

以下是运行脚本的典型命令，并附带参数说明：

```bash
python3 code/run_robe_sim_new_opt.py \
    --model-path <path_to_your_model> \
    --graph-config 2D \
    --env-var standard \
    --num-rollouts 10
```

**参数说明：**

*   `--model-path`：（必需）包含您要评估的已训练 GNN 模型的目录路径。
*   `--graph-config`：（必需）GNN 中布料的表示方式。可以是 `2D` 或 `3D`。`2D` 是一个很好的默认值。
*   `--env-var`：（必需）指定要使用的环境变体。`standard` 是默认值，没有变体。其他选项包括 `body_shape_var`、`pose_var`、`blanket_var` 和 `combo_var`。
*   `--num-rollouts`：运行模拟和评估的次数。

运行此命令后，脚本将：
1.  加载 GNN 模型。
2.  从数据集中随机选择一个“揭开”状态。
3.  运行优化器以找到最佳的重新覆盖动作。
4.  在模拟中执行该动作。
5.  打印结果并将详细输出保存到模型目录中 `cma_evaluations` 下的一个新的 `.pkl` 文件中。

我已准备好提供进一步的帮助。如果您想继续运行脚本或有更多问题，请告诉我。
