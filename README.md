# TIES-Unlearning Diff-LoRA (Layer-wise kNN)
This repository implements a multi-stage fine-tuning algorithm that combines TIES-Merging with Low-Rank Unlearning to eliminate shortcut/bias features learned by Large Language Models on specific tasks.

## Core Mechanism

Traditional **$W_{high} - W_{low}$** parameter subtraction often degrades a model's general reasoning capabilities. To address this, our approach introduces **Phase 2.5 (Layer-wise kNN Analysis)** to dynamically probe and locate specific network layers where shortcuts are heavily concentrated. This is followed by selective TIES merging in Phase 3 to achieve precise, "targeted debiasing".
* Phase 1 (Learn Task): Freeze the N-path; train the classification head and high-rank P-LoRA.
* Phase 2 (Capture Shortcut): Freeze the P-path and classification head; train low-rank N-LoRA on biased mixed data.
* Phase 2.5 (Layer-wise Analysis): Freeze the entire model. Automatically identify the layers with the most severe shortcuts using symmetric KL divergence and kNN feature clustering scores.
* Phase 3 (TIES Fine-tuning): Execute TIES merging (sign alignment and magnitude trimming) strictly on the selected layers to complete the debiasing fine-tuning process.

## Directory Structure

* `configs/`: Global hyperparameters and LoRA configurations (`TrainConfig`, `LoRAConfig`).

* `data/`: Scripts for downloading MNLI and HANS datasets, data mixing, and constructing kNN analysis samples.

* `models/`: Custom dual-path LoRA layers (`ties_lora.py`), model surgery scripts (`surgery.py`), and the Phase 2.5 analysis engine (`analyzer.py`).

* `training/`: The three-phase training main loop, metrics evaluation, and baseline comparisons.

* `utils/:` Optimizer parameter grouping and logging utilities.

## Getting Started

### Prerequisites

    ```bash
    Python >= 3.9

    PyTorch >= 2.0.0
    ```

### Installation
1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/TIES-Unlearning.git
    cd TIES-Unlearning
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```


## Quick Start

1. Run the complete three-phase debiasing training and baseline experiments:

    ```bash
    python main.py
    ```

2. After running, the evaluation metrics will be saved in `ties_unlearn_results/final_summary.json` 

## Configuration Highlights
You can easily customize the experiment by modifying the parameters in `configs/`. Here are the most critical toggles:

* **Baseline Comparison (JTT vs. TIES)**: Set `run_jtt = True` to run the Just Train Twice (JTT) baseline.

    * Adjust `jtt_upweight_factor` (default: 4) to control the penalty for misclassified samples.

* Layer-wise Analysis (Phase 2.5): `enable_layerwise_analysis`: Toggle the automatic layer selection.

    * `layer_selection_topk` (default: 4): Number of layers to select for targeted debiasing.

* TIES Merging:

    * `trim_ratio` (default: 0.2): Controls the proportion of low-magnitude weights dropped during Phase 3.

* Data Mixture:

    * `phase2_mnli_mix_ratio` (default: 0.10): The proportion of MNLI data mixed with HANS entailment data during Phase 2 shortcut capture.

## Contributing & Contact
We welcome contributions! If you encounter any bugs, have feature requests, or want to discuss the code:

Issues: Please open an issue on GitHub. We actively monitor the issue tracker.

Pull Requests: Feel free to submit PRs for code improvements or bug fixes.

Contact: For academic collaborations or specific technical questions, please reach out via [ye_bohou@outlook.com].