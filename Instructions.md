# 项目代码拆分说明

我将原 Notebook 代码完整没变的按功能分到了不同的文件里。以下是原本 10 个 Section 的具体去向：

### 配置与数据
* **Section 3 (TrainConfig & LoRAConfig)**  `configs/config.py`
  *(超参数、混合比例等，都在这里改)*

* **Section 4 (所有的数据加载器)**  
`data/dataloader.py`

### 模型与分析 (核心逻辑)
* **Section 1 (TIESUnlearnLoRALinear 类)** 
`models/ties_lora.py`
* **Section 2 (注入 LoRA 与 Phase 2.5 层级分析)** 被拆成了两个文件：

  * `models/surgery.py` *(负责替换网络层和参数卸载)*

  * `models/analyzer.py` *(负责 Phase 2.5 的 KL 散度计算和 kNN 特征打分)*

### 训练与评估
* **Section 7 (train_ties_unlearn 主训练循环)**
`training/trainer.py`
* **Section 8 (单路 LoRA 跑基线)** 
`training/baseline.py`
* **Section 5 (eval_mnli, eval_hans)**
`training/evaluate.py`

### 工具与入口
* **Section 6 (参数分组、混合精度、保存检查点)**
`utils/optim_utils.py`
* **Section 9 (print_comparison 结果比对)**
`utils/logger.py`
* **Section 10 (程序启动与传参)**
`main.py`

### 配置文件

* **Optional Colab dependency installer**

    `requirements.txt`

    终端执行 `pip install -r requirements.txt`安装全部依赖

### **代码运行**
直接在终端执行 `python main.py` 即可，这等价于原来点击 Notebook 的“全部运行”。