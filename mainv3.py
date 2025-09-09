# ==============================================================================
# 코드 버전 및 기능 요약
# ==============================================================================
# 버전: 3.1
#
# 주요 기능:
# 1. 2-Way 동시 시뮬레이션: 단일 실행으로 아래 두 개의 독립적인 연합학습 시스템을
#    동시에 진행하고 그 결과를 직접 비교합니다.
#    - 모델 A (Baseline): 전통적인 '단순 평균(FedAvg)' 방식
#    - 모델 B (mFL): '기여도 기반(FedContrib)' 방어 시스템이 적용된 방식
# 2. 악의적 노드 시뮬레이션:
#    - 랜덤 노이즈 공격 (--ns): 무작위 파라미터를 제출하여 학습을 방해합니다.
#    - 데이터 포이즈닝 공격 (--df): 데이터 라벨을 조작하여 특정 클래스의 성능을 저하시킵니다.
# 3. 상세 분석 및 시각화:
#    - 훈련 데이터와 위원회 테스트 데이터의 분할 현황을 모두 출력합니다.
#    - 두 모델의 성능을 비교하는 다양한 그래프와 상세한 로그 파일을 생성합니다.
#    - 라운드 수에 따라 그래프의 X축 눈금 간격을 자동으로 조절합니다.
# 4. 유연한 실행 옵션:
#    --ns [N]:      랜덤 노이즈 공격자 수를 지정합니다. (기본값: 0, 옵션만 줄 경우: 1)
#    --df [N] [L]:  데이터 포이즈닝 공격자 수(N)와 오염 레벨(L)을 지정합니다.
#    --A:           모델 A (단순 평균) 시뮬레이션만 실행합니다.
#    --B:           모델 B (기여도 기반) 시뮬레이션만 실행합니다.
#    --csf [F]:     기여도 점수 차이를 증폭시키는 배수를 설정합니다. (기본값: 1.0)
# ==============================================================================


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision.models as models

import flwr as fl
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Parameters,
    FitRes,
)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

import numpy as np
from collections import OrderedDict, defaultdict
from typing import List, Tuple, Dict, Optional, Union, Set
import matplotlib.pyplot as plt
import pandas as pd
import time
import argparse
import os
from datetime import datetime

# ==============================================================================
# 1. 하이퍼파라미터 및 설정
# ==============================================================================
# --- 코드 버전 ---
CODE_VERSION = "3.1"

# --- 기본 학습 파라미터 ---
NUM_CLIENTS = 10
BATCH_SIZE = 32
NUM_ROUNDS = 30
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.01

# --- 데이터 분할 파라미터 ---
DIRICHLET_ALPHA = 0.5
MIN_SAMPLES_PER_CLASS = 30
RANDOM_SEED = 42

# --- 기여도 측정 파라미터 ---
NUM_COMMITTEE_MEMBERS = 3
EWMA_BETA = 0.9
CONTRIBUTION_SCALING_FACTOR = 1.0

# --- 시스템 설정 ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"학습 장치: {DEVICE}")

# --- 결과 기록용 전역 변수 ---
history = defaultdict(list)
aggregation_weights_history = defaultdict(list)
time_history = defaultdict(list)

# ==============================================================================
# 2. 데이터셋 준비
# ==============================================================================
def prepare_datasets() -> Tuple[List[Dataset], List[DataLoader], pd.DataFrame, pd.DataFrame]:
    """훈련 데이터셋과 위원회용 테스트 데이터셋을 준비합니다."""
    np.random.seed(RANDOM_SEED)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    data_path = './data'
    trainset = CIFAR10(root=data_path, train=True, download=True, transform=transform)
    testset = CIFAR10(root=data_path, train=False, download=True, transform=transform)
    
    # 훈련 데이터셋 Non-IID 분할
    class_indices_train = [np.where(np.array(trainset.targets) == i)[0].tolist() for i in range(10)]
    for indices in class_indices_train: np.random.shuffle(indices)
    client_data_indices = [[] for _ in range(NUM_CLIENTS)]
    for cid in range(NUM_CLIENTS):
        for clsid in range(10):
            num_to_take = MIN_SAMPLES_PER_CLASS
            taken = class_indices_train[clsid][:num_to_take]
            class_indices_train[clsid] = class_indices_train[clsid][num_to_take:]
            client_data_indices[cid].extend(taken)
    for clsid in range(10):
        if not class_indices_train[clsid]: continue
        rem_indices = class_indices_train[clsid]
        props = np.random.dirichlet(np.repeat(DIRICHLET_ALPHA, NUM_CLIENTS))
        samples = (props * len(rem_indices)).astype(int)
        rem = len(rem_indices) - samples.sum()
        if rem > 0: samples[np.argmax(samples)] += rem
        start = 0
        for cid in range(NUM_CLIENTS):
            num_give = samples[cid]
            end = start + num_give
            client_data_indices[cid].extend(rem_indices[start:end])
            start = end
    train_subsets = [Subset(trainset, indices) for indices in client_data_indices]
    partition_report_df = create_partition_report(client_data_indices, trainset.targets, "Client")

    # 위원회용 테스트 데이터셋 IID 분할
    print(f"\n[데이터 분할] 위원회({NUM_COMMITTEE_MEMBERS}명)를 위해 테스트셋을 IID로 분할합니다.")
    committee_test_loaders = []
    class_indices_test = [np.where(np.array(testset.targets) == i)[0] for i in range(10)]
    committee_indices = [[] for _ in range(NUM_COMMITTEE_MEMBERS)]
    for indices_per_class in class_indices_test:
        split_indices = np.array_split(indices_per_class, NUM_COMMITTEE_MEMBERS)
        for i in range(NUM_COMMITTEE_MEMBERS):
            committee_indices[i].extend(split_indices[i])
    
    for i, indices in enumerate(committee_indices):
        dataset = Subset(testset, indices)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
        committee_test_loaders.append(loader)
    
    committee_report_df = create_partition_report(committee_indices, testset.targets, "Committee")
    
    return train_subsets, committee_test_loaders, partition_report_df, committee_report_df

def create_partition_report(indices_list, all_labels, entity_name="Entity"):
    report = []
    all_labels_np = np.array(all_labels)
    for i, indices in enumerate(indices_list):
        labels = all_labels_np[indices]
        counts = {cls: count for cls, count in zip(*np.unique(labels, return_counts=True))}
        row = {entity_name: i, "Num Samples": len(indices)}
        for c in range(10): row[f"Class {c}"] = counts.get(c, 0)
        report.append(row)
    return pd.DataFrame(report)

# ==============================================================================
# 3. 모델 및 클라이언트 구현
# ==============================================================================
def get_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class DualSystemClient(fl.client.NumPyClient):
    """모델 A와 B에 대해 각각 학습을 수행하는 클라이언트"""
    def __init__(self, cid: str, train_subset: Subset, client_type: str, poison_map: Dict[int, int] = None):
        self.cid = cid
        self.client_type = client_type
        
        if client_type == "poison":
            poison_info = ", ".join([f"{k}->{v}" for k, v in poison_map.items()])
            print(f"  - [경고] 데이터 포이즈닝 노드 {cid}가 라벨 ({poison_info}) 공격을 수행합니다.")
            poisoned_dataset = PoisonedDataset(train_subset, poison_map)
            self.trainloader = DataLoader(poisoned_dataset, batch_size=BATCH_SIZE, shuffle=True)
        else:
            self.trainloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

        self.model = get_model(10)

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        self.model.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in params_dict}), strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        params_a = parameters_to_ndarrays(config["params_a"])
        params_b = parameters_to_ndarrays(config["params_b"])
        metrics = {"cid": self.cid}
        
        # --- A 모델 학습 ---
        if config.get("run_a", True):
            if self.client_type == "noise":
                print(f"  - [경고] 노이즈 공격 노드 {self.cid}가 A 모델에 랜덤 파라미터를 제출합니다.")
                updated_params_a = [val.cpu().numpy() for _, val in get_model(10).state_dict().items()]
                metrics["fit_duration_a"] = 0.1
            else:
                start_a = time.time()
                self.set_parameters(params_a)
                self.model.to(DEVICE).train()
                optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=0.9)
                for _ in range(LOCAL_EPOCHS):
                    for images, labels in self.trainloader:
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        optimizer.zero_grad()
                        loss = nn.CrossEntropyLoss()(self.model(images), labels)
                        loss.backward()
                        optimizer.step()
                updated_params_a = self.get_parameters({})
                metrics["fit_duration_a"] = time.time() - start_a
            metrics["params_a_ndarrays"] = updated_params_a

        # --- B 모델 학습 ---
        if config.get("run_b", True):
            if self.client_type == "noise":
                print(f"  - [경고] 노이즈 공격 노드 {self.cid}가 B 모델에 랜덤 파라미터를 제출합니다.")
                updated_params_b = [val.cpu().numpy() for _, val in get_model(10).state_dict().items()]
                metrics["fit_duration_b"] = 0.1
            else:
                start_b = time.time()
                self.set_parameters(params_b)
                self.model.to(DEVICE).train()
                optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=0.9)
                for _ in range(LOCAL_EPOCHS):
                    for images, labels in self.trainloader:
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        optimizer.zero_grad()
                        loss = nn.CrossEntropyLoss()(self.model(images), labels)
                        loss.backward()
                        optimizer.step()
                updated_params_b = self.get_parameters({})
                metrics["fit_duration_b"] = time.time() - start_b
            metrics["params_b_ndarrays"] = updated_params_b
            
        return self.get_parameters({}), len(self.trainloader.dataset), metrics

class PoisonedDataset(Dataset):
    def __init__(self, original_dataset, poison_map: Dict[int, int]):
        self.original_dataset = original_dataset
        self.poison_map = poison_map
    def __len__(self): return len(self.original_dataset)
    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        return (image, self.poison_map[label]) if label in self.poison_map else (image, label)

# ==============================================================================
# 4. 2-Way 통합 전략
# ==============================================================================
def evaluate_model(model: nn.Module, testloader: DataLoader, target_classes: Optional[List[int]] = None) -> Dict[str, float]:
    model.to(DEVICE).eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            if target_classes:
                mask = torch.tensor([label in target_classes for label in labels])
                images, labels = images[mask], labels[mask]
                if images.size(0) == 0: continue
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return {"accuracy": 0.0} if total == 0 else {"accuracy": correct / total}

class DualSystemStrategy(FedAvg):
    def __init__(
        self,
        committee_test_loaders: List[DataLoader],
        beta: float,
        scaling_factor: float,
        poisoned_classes: Optional[Set[int]] = None,
        run_a: bool = True,
        run_b: bool = True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.committee_test_loaders = committee_test_loaders
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.poisoned_classes = poisoned_classes if poisoned_classes else set()
        self.run_a = run_a
        self.run_b = run_b
        
        self.params_a: Optional[Parameters] = None
        self.params_b: Optional[Parameters] = None
        self.model_for_eval = get_model(10)
        self.ewma_scores = defaultdict(float)

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        initial_params = super().initialize_parameters(client_manager)
        if initial_params:
            self.params_a = initial_params
            self.params_b = initial_params
        return initial_params

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        config = {"params_a": self.params_a, "params_b": self.params_b, "run_a": self.run_a, "run_b": self.run_b}
        fit_ins = fl.common.FitIns(parameters, config)
        clients = client_manager.sample(num_clients=self.min_fit_clients, min_num_clients=self.min_fit_clients)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        
        if not results: return None, {}
        
        for _, fit_res in results:
            cid = fit_res.metrics.get("cid", "unknown")
            if self.run_a: time_history[f"cli_{cid}_fit_duration_a"].append(fit_res.metrics.get("fit_duration_a", 0))
            if self.run_b: time_history[f"cli_{cid}_fit_duration_b"].append(fit_res.metrics.get("fit_duration_b", 0))

        if self.run_a:
            print("\n" + "="*30 + f" 라운드 {server_round}: 모델 A (단순 평균) " + "="*30)
            agg_start_a = time.time()
            weights_results_a = [(fit_res.metrics["params_a_ndarrays"], fit_res.num_examples) for _, fit_res in results]
            aggregated_ndarrays_a = self.aggregate(weights_results_a)
            self.params_a = ndarrays_to_parameters(aggregated_ndarrays_a)
            time_history["base_agg_time"].append(time.time() - agg_start_a)

            eval_start_a = time.time()
            global_state_dict_a = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model_for_eval.state_dict().keys(), aggregated_ndarrays_a)})
            self.model_for_eval.load_state_dict(global_state_dict_a)
            full_test_loader = DataLoader(ConcatDataset([loader.dataset for loader in self.committee_test_loaders]), batch_size=BATCH_SIZE*2)
            metrics_a = evaluate_model(self.model_for_eval, full_test_loader)
            history["base_acc"].append(metrics_a["accuracy"])
            if self.poisoned_classes:
                poisoned_metrics_a = evaluate_model(self.model_for_eval, full_test_loader, target_classes=list(self.poisoned_classes))
                history["base_poi_acc"].append(poisoned_metrics_a["accuracy"])
            time_history["base_eval_time"].append(time.time() - eval_start_a)
            print(f"  - 전체 정확도: {metrics_a['accuracy']:.4f}")
            if self.poisoned_classes: print(f"  - 오염 클래스 정확도: {history['base_poi_acc'][-1]:.4f}")

        if self.run_b:
            print("\n" + "="*30 + f" 라운드 {server_round}: 모델 B (기여도 기반) " + "="*30)
            committee_eval_start = time.time()
            client_contributions = {}
            for _, fit_res in results:
                client_id = fit_res.metrics.get("cid", "unknown")
                params_numpy = fit_res.metrics["params_b_ndarrays"]
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model_for_eval.state_dict().keys(), params_numpy)})
                self.model_for_eval.load_state_dict(state_dict)
                accuracies = [evaluate_model(self.model_for_eval, loader)["accuracy"] for loader in self.committee_test_loaders]
                current_accuracy = np.mean(accuracies)
                previous_ewma = self.ewma_scores[client_id]
                new_ewma = self.beta * current_accuracy + (1 - self.beta) * previous_ewma
                self.ewma_scores[client_id] = new_ewma
                client_contributions[client_id] = new_ewma
            time_history["mfl_eval_time"].append(time.time() - committee_eval_start)

            scaled_contributions = {cid: contrib ** self.scaling_factor for cid, contrib in client_contributions.items()}
            total_contribution = sum(scaled_contributions.values())
            if total_contribution == 0: total_contribution = 1e-9 
            aggregation_weights = {cid: contrib / total_contribution for cid, contrib in scaled_contributions.items()}
            
            print("  - 모델 통합 가중치")
            for cid, weight in sorted(aggregation_weights.items(), key=lambda item: int(item[0])):
                print(f"    - 클라이언트 {cid}: {weight:.4f}")
                aggregation_weights_history[f"client_{cid}_weight"].append(weight)

            agg_start_b = time.time()
            params_list_b = [fit_res.metrics["params_b_ndarrays"] for _, fit_res in results]
            client_ids = [fit_res.metrics.get("cid", "unknown") for _, fit_res in results]
            aggregated_ndarrays_b = []
            for i in range(len(params_list_b[0])):
                weighted_layer_sum = sum(params_list_b[j][i] * aggregation_weights.get(client_ids[j], 0) for j in range(len(client_ids)))
                aggregated_ndarrays_b.append(weighted_layer_sum)
            self.params_b = ndarrays_to_parameters(aggregated_ndarrays_b)
            time_history["mfl_agg_time"].append(time.time() - agg_start_b)

            eval_start_b = time.time()
            global_state_dict_b = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model_for_eval.state_dict().keys(), aggregated_ndarrays_b)})
            self.model_for_eval.load_state_dict(global_state_dict_b)
            full_test_loader = DataLoader(ConcatDataset([loader.dataset for loader in self.committee_test_loaders]), batch_size=BATCH_SIZE*2)
            metrics_b = evaluate_model(self.model_for_eval, full_test_loader)
            history["mfl_acc"].append(metrics_b["accuracy"])
            if self.poisoned_classes:
                poisoned_metrics_b = evaluate_model(self.model_for_eval, full_test_loader, target_classes=list(self.poisoned_classes))
                history["mfl_poi_acc"].append(poisoned_metrics_b["accuracy"])
            time_history["mfl_global_eval_time"].append(time.time() - eval_start_b)
            print(f"  - 전체 정확도: {metrics_b['accuracy']:.4f}")
            if self.poisoned_classes: print(f"  - 오염 클래스 정확도: {history['mfl_poi_acc'][-1]:.4f}")
        
        return self.params_b if self.run_b else self.params_a, {}

# ==============================================================================
# 5. 결과 저장 및 시각화
# ==============================================================================
def save_results(output_dir, partition_report_df, committee_report_df, args, poisoned_classes: Set[int]):
    print(f"\n[결과 저장] 학습 결과를 '{output_dir}' 폴더에 저장하는 중...")
    
    results_filepath = os.path.join(output_dir, "federated_learning_results_v3.txt")
    with open(results_filepath, "w", encoding="utf-8") as f:
        f.write(f"버전: {CODE_VERSION}\n")
        f.write("="*80 + "\n 훈련 데이터 분할 현황\n" + "="*80 + "\n")
        f.write(partition_report_df.to_string() + "\n\n")
        
        f.write("="*80 + "\n 위원회 테스트 데이터 분할 현황\n" + "="*80 + "\n")
        f.write(committee_report_df.to_string() + "\n\n")
        
        f.write("="*80 + "\n 라운드별 글로벌 모델 정확도\n" + "="*80 + "\n")
        accuracy_data = {}
        if history.get("base_acc"): accuracy_data['base_acc'] = history["base_acc"]
        if history.get("base_poi_acc"):
            clean_list = sorted([int(c) for c in poisoned_classes])
            accuracy_data[f'base_poi_acc({str(clean_list).replace(" ", "")})'] = history["base_poi_acc"]
        if history.get("mfl_acc"): accuracy_data['mFL_acc'] = history["mfl_acc"]
        if history.get("mfl_poi_acc"):
            clean_list = sorted([int(c) for c in poisoned_classes])
            accuracy_data[f'mFL_poi_acc({str(clean_list).replace(" ", "")})'] = history["mfl_poi_acc"]
        
        accuracy_df = pd.DataFrame(accuracy_data)
        accuracy_df.index = pd.RangeIndex(1, len(accuracy_df) + 1, name='Round')
        f.write(accuracy_df.to_string(float_format="{:.4f}".format) + "\n\n")

        if aggregation_weights_history:
            f.write("="*80 + "\n 라운드별 모델 합성 가중치 (모델 B)\n" + "="*80 + "\n")
            weights_df = pd.DataFrame(aggregation_weights_history)
            weights_df.index = [f"Round {i+1}" for i in range(len(weights_df))]
            f.write(weights_df.to_string() + "\n\n")

        f.write("="*80 + "\n 라운드별 시간 측정 결과 (초)\n" + "="*80 + "\n")
        time_df = pd.DataFrame(time_history)
        time_df.index = [f"Round {i+1}" for i in range(len(time_df))]
        f.write(time_df.to_string() + "\n\n")
        
        total_simulation_time = time.time() - simulation_start_time
        f.write(f"총 시뮬레이션 시간: {total_simulation_time:.2f}초\n")
        
        avg_times = {col: time_df[col].mean() for col in time_df.columns}
        total_avg_fit = sum(v for k, v in avg_times.items() if 'fit_duration' in k)
        total_avg_agg = avg_times.get('base_agg_time', 0) + avg_times.get('mfl_agg_time', 0)
        total_avg_eval = avg_times.get('base_eval_time', 0) + avg_times.get('mfl_eval_time', 0) + avg_times.get('mfl_global_eval_time', 0)
        f.write(f"(평균 클라이언트 학습: {total_avg_fit:.2f}초, 평균 서버 합산: {total_avg_agg:.2f}초, 평균 서버 평가: {total_avg_eval:.2f}초)\n")

        if history.get("base_acc") and history.get("mfl_acc"):
            avg_base_acc = np.mean(history["base_acc"])
            avg_mfl_acc = np.mean(history["mfl_acc"])
            acc_improvement = ((avg_mfl_acc - avg_base_acc) / avg_base_acc) * 100 if avg_base_acc > 0 else float('inf')
            f.write("\n" + "="*80 + "\n 방어 시스템 효과 요약 (모델 B vs 모델 A)\n" + "="*80 + "\n")
            f.write(f"- 전체 정확도 평균 향상률: {acc_improvement:+.2f}%\n")
            if history.get("base_poi_acc") and history.get("mfl_poi_acc"):
                avg_base_poi_acc = np.mean(history["base_poi_acc"])
                avg_mfl_poi_acc = np.mean(history["mfl_poi_acc"])
                poi_acc_improvement = ((avg_mfl_poi_acc - avg_base_poi_acc) / avg_base_poi_acc) * 100 if avg_base_poi_acc > 0 else float('inf')
                f.write(f"- 오염 클래스 정확도 평균 향상률: {poi_acc_improvement:+.2f}%\n")
        f.write("\n")

        f.write("="*80 + "\n 시뮬레이션 파라미터\n" + "="*80 + "\n")
        f.write(f"CODE_VERSION: {CODE_VERSION}\n")
        f.write(f"NUM_CLIENTS: {NUM_CLIENTS}\n")
        f.write(f"NUM_ROUNDS: {NUM_ROUNDS}\n")
        f.write(f"LOCAL_EPOCHS: {LOCAL_EPOCHS}\n")
        f.write(f"DIRICHLET_ALPHA: {DIRICHLET_ALPHA}\n")
        f.write(f"MIN_SAMPLES_PER_CLASS: {MIN_SAMPLES_PER_CLASS}\n")
        f.write(f"RANDOM_SEED: {RANDOM_SEED}\n")
        f.write(f"NUM_COMMITTEE_MEMBERS: {NUM_COMMITTEE_MEMBERS}\n")
        f.write(f"EWMA_BETA: {EWMA_BETA}\n")
        f.write(f"CONTRIBUTION_SCALING_FACTOR: {args.csf}\n")
        f.write(f"NUM_NOISE_ATTACKERS: {args.ns}\n")
        num_df_attackers = args.df[0] if args.df else 0
        poison_level = args.df[1] if args.df and len(args.df) > 1 else (1 if num_df_attackers > 0 else 0)
        f.write(f"NUM_DATAF_ATTACKERS: {num_df_attackers}\n")
        f.write(f"DATAF_POISON_LEVEL: {poison_level}\n")

    print(f"'{results_filepath}' 파일이 저장되었습니다.")
    
    # --- 그래프 생성 ---
    def get_xticks(total_rounds):
        if total_rounds > 100:
            step = 10
        elif total_rounds >= 50:
            step = 5
        else:
            return list(range(1, total_rounds + 1))
        ticks = list(range(step, total_rounds + 1, step))
        if 1 not in ticks: ticks.insert(0, 1)
        return ticks

    figsize = (20, 6) if NUM_ROUNDS > 100 else (12, 6)
    
    # ... (그래프 생성 로직 전체) ...
    
# ==============================================================================
# 6. 연합학습 시뮬레이션 실행
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description=f"버전 {CODE_VERSION}: 2-Way 통합 시뮬레이션")
    parser.add_argument("--A", action="store_true", help="모델 A(단순 평균)만 실행")
    parser.add_argument("--B", action="store_true", help="모델 B(기여도 기반)만 실행")
    parser.add_argument("--ns", type=int, nargs='?', const=1, default=0, metavar='N', help="랜덤 노이즈 공격자 수")
    parser.add_argument("--df", type=int, nargs='+', metavar=('N', 'L'), help="데이터 포이즈닝 공격자 수(N) 및 오염 레벨(L)")
    parser.add_argument("--csf", type=float, default=CONTRIBUTION_SCALING_FACTOR, metavar='F', help="기여도 가중치 배수")
    args = parser.parse_args()
    
    run_a, run_b = (True, False) if args.A else (False, True) if args.B else (True, True)

    # --- 결과 저장 폴더 생성 ---
    now = datetime.now()
    timestamp = now.strftime("%m%d_%H%M")
    options_list = []
    if args.ns > 0: options_list.append(f"ns{args.ns}")
    if args.df: 
        num_df = args.df[0]
        poison_level = args.df[1] if len(args.df) > 1 else 1
        options_list.append(f"df{num_df}L{poison_level}")
    if args.csf != 1.0: options_list.append(f"csf{args.csf}")
    options_list.append(f"ew{EWMA_BETA}")
    options_str = f"_[{','.join(options_list)}]" if options_list else ""
    output_dir = f"{timestamp}_v{CODE_VERSION}_R{NUM_ROUNDS}{options_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80, f"\n버전 {CODE_VERSION}: 2-Way 통합 시뮬레이션을 시작합니다.\n", "=" * 80)
    
    train_subsets, committee_test_loaders, partition_report_df, committee_report_df = prepare_datasets()
    
    print("\n 최종 훈련 데이터 분할 현황")
    print("="*80)
    print(partition_report_df.to_string())
    print("\n 최종 위원회 테스트 데이터 분할 현황")
    print("="*80)
    print(committee_report_df.to_string())
    print("="*80 + "\n")

    # --- 악의적인 노드 설정 ---
    num_ns = args.ns
    num_df = args.df[0] if args.df else 0
    poison_level = args.df[1] if args.df and len(args.df) > 1 else 1
    
    total_malicious = num_ns + num_df
    client_ids_str = [str(i) for i in range(NUM_CLIENTS)]
    
    df_ids = set(client_ids_str[NUM_CLIENTS - num_df:])
    ns_ids = set(client_ids_str[NUM_CLIENTS - total_malicious : NUM_CLIENTS - num_df])
    
    shared_poison_map: Dict[int, int] = {}
    if df_ids:
        np.random.seed(RANDOM_SEED)
        all_classes = list(range(10))
        source_classes = np.random.choice(all_classes, poison_level, replace=False)
        for source in source_classes:
            target_options = [c for c in all_classes if c != source]
            shared_poison_map[source] = np.random.choice(target_options)

    poisoned_classes_to_track = set(shared_poison_map.keys())

    if ns_ids: print(f"[설정] 노이즈 공격 노드 수: {len(ns_ids)}개, ID: {sorted(list(ns_ids))}")
    if df_ids: print(f"[설정] 데이터 포이즈닝 노드 수: {len(df_ids)}개, ID: {sorted(list(df_ids))}, 오염 레벨: {poison_level}")
    if poisoned_classes_to_track: print(f"[설정] 추적할 오염 클래스: {sorted([int(c) for c in poisoned_classes_to_track])}")

    def client_fn(cid: str) -> fl.client.Client:
        client_type = "normal"
        poison_map = None
        if cid in ns_ids: client_type = "noise"
        elif cid in df_ids:
            client_type = "poison"
            poison_map = shared_poison_map
        
        return DualSystemClient(cid, train_subsets[int(cid)], client_type, poison_map).to_client()

    strategy = DualSystemStrategy(
        committee_test_loaders=committee_test_loaders,
        beta=EWMA_BETA,
        scaling_factor=args.csf,
        poisoned_classes=poisoned_classes_to_track,
        run_a=run_a,
        run_b=run_b,
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=None,
    )
    
    global simulation_start_time
    simulation_start_time = time.time()
    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
            client_resources={"num_cpus": 2, "num_gpus": 0.5} if DEVICE.type == "cuda" else {"num_cpus": 2}
        )
    except Exception as e:
        print(f"\n[오류] 시뮬레이션 중 예외가 발생했습니다: {e}")
    finally:
        total_time = time.time() - simulation_start_time
        print("\n" + "=" * 80, f"\n연합학습 시뮬레이션 종료. (총 소요 시간: {total_time:.2f}초)\n", "=" * 80)
        if history:
            save_results(output_dir, partition_report_df, committee_report_df, args, poisoned_classes_to_track)
        else:
            print("[알림] 학습이 완료되기 전에 종료되어 저장할 결과가 없습니다.")

if __name__ == "__main__":
    main()
