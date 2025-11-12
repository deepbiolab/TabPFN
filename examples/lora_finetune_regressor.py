"""
Provides a detailed example of fine-tuning a TabPFNRegressor model using LoRA.

This script demonstrates the complete workflow, including data loading, model
configuration with LoRA injection, a parameter-efficient fine-tuning loop,
and performance evaluation.

Note: We recommend running the fine-tuning scripts on a CUDA-enabled GPU.
"""

from functools import partial

import numpy as np
import sklearn.datasets
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- [LoRA] 1. 导入 loralib ---
try:
    import lora
except ImportError:
    raise ImportError(
        "loralib is not installed. Please install it with 'pip install loralib'"
    )

from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator


# --- [LoRA] 2. 辅助函数，用于统计可训练参数 ---
def count_trainable_parameters(model):
    """Returns the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prepare_data(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads, subsets, and splits the Bike Sharing Demand dataset."""
    print("--- 1. Data Preparation ---")
    bike_sharing = sklearn.datasets.fetch_openml(
        name="Bike_Sharing_Demand", version=2, as_frame=True, parser="auto"
    )

    X_df = bike_sharing.data
    y_df = bike_sharing.target
    X_numeric = X_df.select_dtypes(include=np.number)
    X_all, y_all = X_numeric.values, y_df.values

    rng = np.random.default_rng(config["random_seed"])
    num_samples_to_use = min(config["num_samples_to_use"], len(y_all))
    indices = rng.choice(np.arange(len(y_all)), size=num_samples_to_use, replace=False)
    X, y = X_all[indices], y_all[indices]

    splitter = partial(
        train_test_split,
        test_size=config["valid_set_ratio"],
        random_state=config["random_seed"],
    )
    X_train, X_test, y_train, y_test = splitter(X, y)

    print(
        f"Loaded and split data: {X_train.shape[0]} train, {X_test.shape[0]} test samples."
    )
    print("---------------------------\n")
    return X_train, X_test, y_train, y_test


def setup_regressor(config: dict) -> tuple[TabPFNRegressor, dict]:
    """Initializes the TabPFN regressor and its configuration."""
    print("--- 2. Model Setup ---")
    regressor_config = {
        "ignore_pretraining_limits": True,
        "device": config["device"],
        "n_estimators": 1,  # 使用单个模型以简化微调
        "random_state": config["random_seed"],
        "inference_precision": torch.float32,
    }
    # 注意: differentiable_input=True 在微调时可能是必要的，因为它会影响模型内部的图构建
    regressor = TabPFNRegressor(
        **regressor_config, fit_mode="batched", differentiable_input=True
    )

    print(f"Using device: {config['device']}")
    print("----------------------\n")
    return regressor, regressor_config


def evaluate_regressor(
    regressor: TabPFNRegressor,
    eval_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float]:
    """Evaluates the regressor's performance on the test set."""
    # 在评估前，需要将LoRA权重合并，或者在评估模式下运行
    # clone_model_for_evaluation 会创建一个干净的副本，所以我们需要对副本也进行操作
    # 这里我们简化一下，直接在微调后的模型上评估
    
    # 切换到评估模式，loralib会自动处理
    regressor.model_.eval()
    
    # 我们可以选择合并权重进行评估，这样速度更快，且不需要 loralib
    # 如果不合并，lora层会在前向传播时计算，效果一样但稍慢
    # lora.merge_lora_weights(regressor.model_) # 可选步骤

    try:
        # 注意：这里我们不再克隆模型，而是直接使用微调后的regressor
        # 为了保证评估的公平性，理想情况下应该在一个干净的、合并了权重的模型上评估
        # 但为了简化示例，我们直接在当前模型上评估
        
        # 为了更准确的评估，我们需要一个干净的上下文。我们重新 fit 一下
        # 这确保了评估时的内部状态是基于完整的训练数据的
        eval_regressor = clone_model_for_evaluation(regressor, eval_config, TabPFNRegressor)
        eval_regressor.fit(X_train, y_train)
        
        predictions = eval_regressor.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        mse, mae, r2 = np.nan, np.nan, np.nan

    return mse, mae, r2


def main() -> None:
    """Main function to configure and run the LoRA finetuning workflow."""
    # --- Master Configuration ---
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_samples_to_use": 10_000, # 使用一个较小的值以加速示例
        "random_seed": 42,
        "valid_set_ratio": 0.3,
        "n_inference_context_samples": 1024, # 上下文大小
    }
    
    # --- [LoRA] 3. 添加LoRA相关的超参数 ---
    config["finetuning"] = {
        "epochs": 5, # 训练更多轮次以观察LoRA效果
        # LoRA通常可以使用比全量微调更高的学习率
        "learning_rate": 1e-4, 
        "meta_batch_size": 1,
        "batch_size": 1024, # 保持与上下文大小一致
        
        # LoRA-specific hyperparameters
        "lora_r": 8,           # LoRA的秩
        "lora_alpha": 16,      # LoRA的缩放因子
        "lora_dropout": 0.1,   # LoRA层的Dropout
    }
    # 确保batch_size不超过实际训练样本数
    config["finetuning"]["batch_size"] = int(
        min(
            config["n_inference_context_samples"],
            config["num_samples_to_use"] * (1 - config["valid_set_ratio"]),
        )
    )

    # --- Setup Data, Model, and Dataloader ---
    X_train, X_test, y_train, y_test = prepare_data(config)
    regressor, regressor_config = setup_regressor(config)

    # 官方脚本中，模型在 get_preprocessed_datasets 时才真正被初始化
    # 我们需要在它被初始化之后再注入LoRA
    splitter = partial(train_test_split, test_size=config["valid_set_ratio"])
    training_datasets = regressor.get_preprocessed_datasets(
        X_train, y_train, splitter, max_data_size=config["finetuning"]["batch_size"]
    )
    
    # 从regressor实例中获取底层模型
    # 在TabPFN 2.0+版本中，模型实例是 .model_
    model = regressor.model_

    # --- [LoRA] 4. 对模型进行LoRA改造 ---
    print("\n--- Injecting LoRA layers ---")
    original_params = count_trainable_parameters(model)

    # 遍历所有模块，将符合条件的 nn.Linear 替换为 lora.Linear
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 排除解码头，通常我们不希望对最后的输出层应用LoRA
            if 'decoder' in name:
                print(f"  Skipping LoRA for decoder layer: {name}")
                continue

            parent_name = '.'.join(name.split('.')[:-1])
            layer_name = name.split('.')[-1]
            parent_module = model.get_submodule(parent_name)

            lora_layer = lora.Linear(
                module.in_features,
                module.out_features,
                r=config["finetuning"]["lora_r"],
                lora_alpha=config["finetuning"]["lora_alpha"],
                lora_dropout=config["finetuning"]["lora_dropout"],
                bias=module.bias is not None,
            )
            # 复制原始权重
            lora_layer.weight = module.weight
            if module.bias is not None:
                lora_layer.bias = module.bias
            
            # 替换
            setattr(parent_module, layer_name, lora_layer)
            print(f"  Replaced '{name}' with LoRA layer.")

    # 冻结所有非LoRA参数
    lora.mark_only_lora_as_trainable(model)

    lora_params = count_trainable_parameters(model)
    print(f"\nOriginal trainable parameters: {original_params}")
    print(f"LoRA trainable parameters: {lora_params} ({(lora_params/original_params)*100:.2f}%)")
    print("---------------------------------\n")

    # 创建 DataLoader 和 Optimizer
    finetuning_dataloader = DataLoader(
        training_datasets,
        batch_size=config["finetuning"]["meta_batch_size"],
        collate_fn=meta_dataset_collator,
    )

    # 优化器现在只会看到可训练的LoRA参数
    optimizer = Adam(model.parameters(), lr=config["finetuning"]["learning_rate"])
    print(
        f"--- Optimizer Initialized for LoRA: Adam, LR: {config['finetuning']['learning_rate']} ---\n"
    )

    eval_config = {
        **regressor_config,
        "inference_config": {
            "SUBSAMPLE_SAMPLES": config["n_inference_context_samples"]
        },
    }

    # --- Finetuning and Evaluation Loop ---
    print("--- 3. Starting Finetuning & Evaluation ---")
    for epoch in range(config["finetuning"]["epochs"] + 1):
        # 初始评估（epoch 0）
        if epoch == 0:
            status = "Initial (Before Finetuning)"
            mse, mae, r2 = evaluate_regressor(
                regressor, eval_config, X_train, y_train, X_test, y_test
            )
            print(
                f"📊 {status} Evaluation | Test MSE: {mse:.4f}, Test MAE: {mae:.4f}, Test R2: {r2:.4f}\n"
            )
            continue
        
        # 微调
        model.train()  # 确保模型处于训练模式
        progress_bar = tqdm(finetuning_dataloader, desc=f"Finetuning Epoch {epoch}")
        for data_batch in progress_bar:
            optimizer.zero_grad()
            (
                X_trains_preprocessed,
                X_tests_preprocessed,
                y_trains_znorm,
                y_test_znorm,
                cat_ixs,
                confs,
                raw_space_bardist_,
                znorm_space_bardist_,
                _,
                y_test_raw,
            ) = data_batch

            regressor.raw_space_bardist_ = raw_space_bardist_[0]
            regressor.bardist_ = znorm_space_bardist_[0]
            regressor.fit_from_preprocessed(
                X_trains_preprocessed, y_trains_znorm, cat_ixs, confs
            )
            logits, _, _ = regressor.forward(X_tests_preprocessed)

            loss_fn = znorm_space_bardist_[0]
            y_target = y_test_znorm

            loss = loss_fn(logits, y_target.to(config["device"])).mean()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # 每个epoch后评估
        status = f"After Epoch {epoch}"
        mse, mae, r2 = evaluate_regressor(
            regressor, eval_config, X_train, y_train, X_test, y_test
        )
        print(
            f"📊 {status} Evaluation | Test MSE: {mse:.4f}, Test MAE: {mae:.4f}, Test R2: {r2:.4f}\n"
        )

    print("--- ✅ LoRA Finetuning Finished ---")


if __name__ == "__main__":
    main()
