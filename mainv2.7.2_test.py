# ==============================================================================
# 코드 버전 및 기능 요약
# ==============================================================================
# 버전: 2.12.2 (CIFAR-100)
#
# 주요 기능:
# 1. 기여도 기반 연합학습 (FedContrib), 독립 위원회(Committee)
# 2. 악의적 노드 시뮬레이션 (랜덤 노이즈 --ns, 데이터 포이즈닝 --df)
# 3. 공격 강도 및 방식 조절 (집중 공격)
# 4. 상세 분석 및 시각화 (모델 궤적 지도, 가중치 발산 분석, 통합 정확도 그래프 등)
# 5. 그래프 X축 동적 조절
# 6. 실험 재현성 (랜덤 시드 고정)
# 7. 유연한 실행 옵션 (--s, --csf)
# 8. 동적 학습률 스케줄러
# 9. [v2.12] 가중치 그래프 정상 노드 색상 개별 구별
# 10. [v2.12.2] CIFAR-100 데이터셋 완벽 지원
# ==============================================================================


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision.datasets import CIFAR10, CIFAR100 # [수정] CIFAR100 추가
import torchvision.transforms as transforms
import torchvision.models as models

import flwr as fl
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg

import numpy as np
from collections import OrderedDict, defaultdict
from typing import List, Tuple, Dict, Optional, Union, Set
import matplotlib.pyplot as plt
import pandas as pd
import time
import argparse
import os
from datetime import datetime
import pytz
from sklearn.decomposition import PCA # PCA를 위해 scikit-learn 임포트

# ==============================================================================
# 1. 하이퍼파라미터 및 설정
# ==============================================================================
# --- 기본 학습 파라미터 ---
NUM_CLIENTS = 10           # 학습에 참여하는 클라이언트(노드)의 수
BATCH_SIZE = 32            # 학습 시 사용되는 데이터 배치의 크기
NUM_ROUNDS = 50            # 총 연합학습 라운드 수
LOCAL_EPOCHS = 2           # 각 클라이언트가 라운드마다 수행하는 로컬 학습 횟수
LEARNING_RATE = 0.01       # 모델 학습률

# --- 학습률 스케줄러 파라미터 ---
LR_SCHEDULER_STEP = 5     # 5 라운드마다 학습률을 조정합니다.
LR_SCHEDULER_GAMMA = 1.0  # 학습률을 20%씩 감소시킵니다.

# --- 데이터 분할 파라미터 ---
DIRICHLET_ALPHA = 0.5      # 데이터의 Non-IID 정도를 조절하는 디리클레 분포 파라미터
# [수정] CIFAR-100은 클래스당 샘플이 적으므로 30 -> 10으로 조정
MIN_SAMPLES_PER_CLASS = 10 # 각 클라이언트가 클래스별로 보장받는 최소 샘플 수
RANDOM_SEED = 42           # 재현 가능한 실험을 위한 랜덤 시드

# --- 버전 2: 기여도 측정 파라미터 ---
NUM_COMMITTEE_MEMBERS = 3  # 학습 노드의 성능을 평가할 위원회(Committee)의 수
EWMA_BETA = 0.3            # 정확도를 보정할 지수가중이동평균(EWMA)의 반영률 조정 파라미터
CONTRIBUTION_SCALING_FACTOR = 1.9 # 기여도 점수 차이를 증폭시키는 배수 (1.0 이상 권장)

# --- 시스템 설정 ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"학습 장치: {DEVICE}")

# --- 결과 기록용 전역 변수 ---
history = {"global_accuracy": []}
poisoned_class_accuracy_history = []
aggregation_weights_history = defaultdict(list)
time_history = defaultdict(list)
model_params_history = {}
global_params_history = {}

# ==============================================================================
# 2. 데이터셋 준비
# ==============================================================================
def prepare_datasets() -> Tuple[List[Dataset], List[DataLoader], pd.DataFrame, int]:
    """훈련 데이터셋과 위원회용 테스트 데이터셋을 준비합니다."""
    np.random.seed(RANDOM_SEED)
    
    # [수정] CIFAR-100의 평균 및 표준편차로 변경
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    data_path = './data'
    # [수정] CIFAR10을 CIFAR100으로 변경
    trainset = CIFAR100(root=data_path, train=True, download=True, transform=transform)
    testset = CIFAR100(root=data_path, train=False, download=True, transform=transform)
    num_classes = len(trainset.classes) # 100이 됨
    
    # --- 훈련 데이터셋 Non-IID 분할 ---
    # [수정] 10 -> num_classes
    class_indices_train = [np.where(np.array(trainset.targets) == i)[0].tolist() for i in range(num_classes)]
    for indices in class_indices_train: np.random.shuffle(indices)
    client_data_indices = [[] for _ in range(NUM_CLIENTS)]
    for cid in range(NUM_CLIENTS):
        # [수정] 10 -> num_classes
        for clsid in range(num_classes):
            num_to_take = MIN_SAMPLES_PER_CLASS
            taken = class_indices_train[clsid][:num_to_take]
            class_indices_train[clsid] = class_indices_train[clsid][num_to_take:]
            client_data_indices[cid].extend(taken)
    # [수정] 10 -> num_classes
    for clsid in range(num_classes):
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
    partition_report_df = create_partition_report(client_data_indices, trainset.targets, num_classes)

    # --- 위원회용 테스트 데이터셋 IID 분할 ---
    print(f"\n[데이터 분할] 위원회({NUM_COMMITTEE_MEMBERS}명)를 위해 테스트셋을 IID로 분할합니다.")
    committee_test_loaders = []
    # [수정] 10 -> num_classes
    class_indices_test = [np.where(np.array(testset.targets) == i)[0] for i in range(num_classes)]
    committee_indices = [[] for _ in range(NUM_COMMITTEE_MEMBERS)]
    for indices_per_class in class_indices_test:
        split_indices = np.array_split(indices_per_class, NUM_COMMITTEE_MEMBERS)
        for i in range(NUM_COMMITTEE_MEMBERS):
            committee_indices[i].extend(split_indices[i])
    
    for i, indices in enumerate(committee_indices):
        dataset = Subset(testset, indices)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
        committee_test_loaders.append(loader)
        print(f" - 위원회 멤버 {i+1} 할당 데이터 수: {len(dataset)}개")

    # [수정] num_classes 반환
    return train_subsets, committee_test_loaders, partition_report_df, num_classes

def create_partition_report(client_indices, all_labels, num_classes):
    report = []
    all_labels_np = np.array(all_labels)
    for i, indices in enumerate(client_indices):
        labels = all_labels_np[indices]
        counts = {cls: count for cls, count in zip(*np.unique(labels, return_counts=True))}
        row = {"Client": i, "Num Samples": len(indices)}
        # [수정] 10 -> num_classes
        for c in range(num_classes): row[f"Class {c}"] = counts.get(c, 0)
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

class CifarClient(fl.client.NumPyClient):
    """정상적으로 학습을 수행하는 클라이언트"""
    def __init__(self, cid: str, train_subset: Subset, model: nn.Module):
        self.cid = cid
        self.model = model
        self.trainloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        self.model.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in params_dict}), strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        fit_start_time = time.time()
        
        # 서버로부터 동적 학습률을 받습니다 (없을 경우 기본값 사용).
        current_lr = config.get("current_lr", LEARNING_RATE)
        
        self.set_parameters(parameters)
        self.model.to(DEVICE).train()
        
        # 동적 학습률(current_lr)을 옵티마이저에 적용합니다.
        optimizer = optim.SGD(self.model.parameters(), lr=current_lr, momentum=0.9)
        
        for _ in range(LOCAL_EPOCHS):
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(self.model(images), labels)
                loss.backward()
                optimizer.step()
        fit_duration = time.time() - fit_start_time
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"cid": self.cid, "fit_duration": fit_duration}

class NoiseClient(fl.client.NumPyClient):
    """학습을 수행하지 않고 랜덤 파라미터를 제출하는 악의적인 클라이언트 (랜덤 노이즈 공격)"""
    # [수정] num_classes 인자 추가
    def __init__(self, cid: str, num_classes: int):
        self.cid = cid
        self.num_classes = num_classes

    def get_parameters(self, config):
        print(f"  - [경고] 노이즈 공격 노드 {self.cid}가 랜덤 파라미터를 제출합니다.")
        # [수정] get_model(10) -> get_model(self.num_classes)
        return [val.cpu().numpy() for _, val in get_model(self.num_classes).state_dict().items()]

    def set_parameters(self, parameters): pass
    def fit(self, parameters, config):
        return self.get_parameters(config={}), 1, {"cid": self.cid, "fit_duration": 0.1}

class PoisonedDataset(Dataset):
    """데이터셋의 라벨을 조작하는 래퍼 클래스"""
    def __init__(self, original_dataset, poison_map: Dict[int, int]):
        self.original_dataset = original_dataset
        self.poison_map = poison_map

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        if label in self.poison_map:
            return image, self.poison_map[label]
        return image, label

class DataPoisoningClient(CifarClient):
    """데이터 라벨을 조작하여 학습하는 악의적인 클라이언트 (데이터 포이즈닝 공격)"""
    def __init__(self, cid: str, train_subset: Subset, model: nn.Module, poison_map: Dict[int, int]):
        poison_info = ", ".join([f"{k}->{v}" for k, v in poison_map.items()])
        print(f"  - [경고] 데이터 포이즈닝 노드 {cid}가 라벨 ({poison_info}) 공격을 수행합니다.")
        poisoned_dataset = PoisonedDataset(train_subset, poison_map)
        super().__init__(cid, poisoned_dataset, model)

# ==============================================================================
# 4. 기여도 기반 통합 전략 (FedContrib)
# ==============================================================================
def evaluate_model(model: nn.Module, testloader: DataLoader, target_classes: Optional[List[int]] = None) -> Dict[str, float]:
    """모델의 성능을 평가합니다. target_classes가 주어지면 해당 클래스에 대한 정확도만 계산합니다."""
    model.to(DEVICE).eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            if target_classes:
                mask = torch.tensor([label in target_classes for label in labels])
                images, labels = images[mask], labels[mask]
                if images.size(0) == 0:
                    continue
            
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    
    if total == 0:
        return {"accuracy": 0.0}
    return {"accuracy": correct / total}

class FedContrib(FedAvg):
    def __init__(
        self,
        committee_test_loaders: List[DataLoader],
        num_classes: int, # [수정] num_classes 인자 추가
        beta: float,
        scaling_factor: float,
        skip_committee_eval: bool = False,
        poisoned_classes: Optional[Set[int]] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.committee_test_loaders = committee_test_loaders
        self.skip_committee_eval = skip_committee_eval
        self.beta = beta
        self.scaling_factor = scaling_factor
        # [수정] get_model(10) -> get_model(num_classes)
        self.model_for_eval = get_model(num_classes)
        self.ewma_scores = defaultdict(float)
        self.poisoned_classes = poisoned_classes if poisoned_classes else set()
        self.initial_lr = LEARNING_RATE

    def configure_fit(
        self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """클라이언트 학습 설정을 구성하고, 시작 시점의 글로벌 모델을 저장합니다."""
        
        current_lr = self.initial_lr * (LR_SCHEDULER_GAMMA ** (server_round // LR_SCHEDULER_STEP))

        if self.skip_committee_eval:
            global_params_history[server_round] = parameters_to_ndarrays(parameters)
        
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        
        for _, fit_ins in fit_ins_list:
            fit_ins.config["current_lr"] = current_lr
            
        return fit_ins_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        
        if not results: return None, {}
        
        round_start_time = time.time()
        for _, fit_res in results:
            cid = fit_res.metrics.get("cid", "unknown")
            duration = fit_res.metrics.get("fit_duration", 0)
            time_history[f"cli {cid} train"].append(duration)

        if self.skip_committee_eval:
            params_for_round = {}
            for _, fit_res in results:
                cid = fit_res.metrics.get("cid", "unknown")
                params_for_round[cid] = parameters_to_ndarrays(fit_res.parameters)
            model_params_history[server_round] = params_for_round

        print("\n" + "-"*80)
        print(f"** 라운드 {server_round}: 모델 통합 및 평가 **")
        
        client_contributions = {}
        committee_eval_start = time.time()
        if not self.skip_committee_eval:
            print("  - 위원회 평가 및 기여도 산출")
            for _, fit_res in results:
                client_id = fit_res.metrics.get("cid", "unknown")
                params_numpy = parameters_to_ndarrays(fit_res.parameters)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model_for_eval.state_dict().keys(), params_numpy)})
                self.model_for_eval.load_state_dict(state_dict)

                accuracies = [evaluate_model(self.model_for_eval, loader)["accuracy"] for loader in self.committee_test_loaders]
                current_accuracy = np.mean(accuracies)
                
                previous_ewma = self.ewma_scores[client_id]
                new_ewma = self.beta * current_accuracy + (1 - self.beta) * previous_ewma
                self.ewma_scores[client_id] = new_ewma
                client_contributions[client_id] = new_ewma
            time_history["cli eval time"].append(time.time() - committee_eval_start)
        else:
            print("  - 위원회 평가를 건너뛰었습니다. (모든 클라이언트 가중치 동일)")
            for _, fit_res in results:
                client_id = fit_res.metrics.get("cid", "unknown")
                client_contributions[client_id] = 1.0
            time_history["cli eval time"].append(0)
        
        scaled_contributions = {
            cid: contrib ** self.scaling_factor
            for cid, contrib in client_contributions.items()
        }

        total_contribution = sum(scaled_contributions.values())
        if total_contribution == 0: total_contribution = 1e-9 
        
        aggregation_weights = {cid: contrib / total_contribution for cid, contrib in scaled_contributions.items()}
        
        print("\n  - 모델 통합 가중치")
        for cid, weight in sorted(aggregation_weights.items(), key=lambda item: int(item[0])):
            print(f"    - 클라이언트 {cid}: {weight:.4f}")
            aggregation_weights_history[f"client_{cid}_weight"].append(weight)

        agg_start_time = time.time()
        params_list = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        client_ids = [fit_res.metrics.get("cid", "unknown") for _, fit_res in results]
        
        aggregated_ndarrays: List[np.ndarray] = []
        for i in range(len(params_list[0])):
            weighted_layer_sum = sum(params_list[j][i] * aggregation_weights.get(client_ids[j], 0) for j in range(len(client_ids)))
            aggregated_ndarrays.append(weighted_layer_sum)
        time_history["agg time"].append(time.time() - agg_start_time)

        if self.skip_committee_eval and server_round in model_params_history:
            model_params_history[server_round]['global'] = aggregated_ndarrays

        print("\n  - 글로벌 모델 성능 평가")
        global_eval_start = time.time()
        global_state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model_for_eval.state_dict().keys(), aggregated_ndarrays)})
        self.model_for_eval.load_state_dict(global_state_dict)
        full_test_loader = DataLoader(ConcatDataset([loader.dataset for loader in self.committee_test_loaders]), batch_size=BATCH_SIZE*2)
        global_metrics = evaluate_model(self.model_for_eval, full_test_loader)
        time_history["global eval time"].append(time.time() - global_eval_start)
        
        print(f"    - 전체 정확도: {global_metrics['accuracy']:.4f}")
        history["global_accuracy"].append(global_metrics['accuracy'])
        
        if self.poisoned_classes:
            poisoned_metrics = evaluate_model(self.model_for_eval, full_test_loader, target_classes=list(self.poisoned_classes))
            poisoned_accuracy = poisoned_metrics['accuracy']
            poisoned_class_accuracy_history.append(poisoned_accuracy)
            clean_poisoned_list = sorted([int(c) for c in self.poisoned_classes])
            print(f"    - 오염된 클래스({clean_poisoned_list}) 정확도: {poisoned_accuracy:.4f}")

        
        total_round_duration = time.time() - round_start_time
        time_history["total time"].append(total_round_duration)
        
        print("\n  - 시간 측정:")
        print(f"    - 서버 통합 시간 (agg time): {time_history['agg time'][-1]:.2f}초")
        if not self.skip_committee_eval:
            print(f"    - 위원회 평가 시간 (cli eval time): {time_history['cli eval time'][-1]:.2f}초")
        print(f"    - 글로벌 모델 평가 시간 (global eval time): {time_history['global eval time'][-1]:.2f}초")
        print(f"    - 총 라운드 시간 (total time): {total_round_duration:.2f}초")
        print("-" * 80)

        return ndarrays_to_parameters(aggregated_ndarrays), {}

# ==============================================================================
# 5. 결과 저장 및 시각화
# ==============================================================================
def visualize_model_trajectories(output_dir, round_num, title_prefix, model_params_round, ns_ids, df_ids):
    """PCA를 사용하여 특정 라운드의 모델 궤적을 시각화합니다."""
    
    if 'global' not in model_params_round:
        print(f"[경고] {round_num} 라운드의 글로벌 모델 파라미터가 없어 궤적 맵을 생성할 수 없습니다.")
        return

    start_global_params = global_params_history.get(round_num)
    if start_global_params is None:
        print(f"[경고] {round_num} 라운드의 *시작* 글로벌 모델 파라미터가 없어 궤적 맵을 생성할 수 없습니다.")
        return
        
    end_client_params = model_params_round
    
    all_params_map = {"start_global": start_global_params}
    all_params_map.update(end_client_params)
    
    client_ids_in_order = sorted(all_params_map.keys(), key=lambda x: (x != 'start_global', x != 'global', int(x) if x.isdigit() else 99))
    
    all_vectors = []
    labels = []
    colors = []
    
    for cid in client_ids_in_order:
        params = all_params_map[cid]
        flat_vector = np.concatenate([p.flatten() for p in params])
        all_vectors.append(flat_vector)
        
        if cid == 'start_global':
            labels.append("Start Global")
            colors.append('green')
        elif cid == 'global':
            labels.append("End Global")
            colors.append('orange')
        elif cid in ns_ids or cid in df_ids:
            labels.append(f"Attacker (C{cid})")
            colors.append('red')
        else:
            labels.append(f"Normal (C{cid})")
            colors.append('blue')

    if len(all_vectors) < 2: return

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    transformed = pca.fit_transform(all_vectors)
    
    start_point = None
    client_points = {}
    
    for i, cid in enumerate(client_ids_in_order):
        if cid == 'start_global':
            start_point = transformed[i]
        else:
            client_points[cid] = transformed[i]

    if start_point is None: return

    plt.figure(figsize=(12, 9))
    legend_handles = {
        "Start Global Model": plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=12),
        "Normal Node Update": plt.Line2D([0], [0], color='blue', lw=2),
        "Attacker Node Update": plt.Line2D([0], [0], color='red', lw=2),
        "Final Global Model": plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='orange', markersize=15)
    }

    for cid, (x, y) in client_points.items():
        if cid == 'global':
            plt.scatter(x, y, color='orange', s=300, marker='*', zorder=6)
            continue
        
        color = 'red' if cid in ns_ids or cid in df_ids else 'blue'
        plt.arrow(start_point[0], start_point[1], x - start_point[0], y - start_point[1],
                  head_width=0.05, length_includes_head=True, color=color, alpha=0.7)
        plt.text(x, y, f" C{cid}", fontsize=9, color=color)

    plt.scatter(start_point[0], start_point[1], color='green', s=200, marker='o', zorder=5)
    plt.title(f"{title_prefix}\n(PCA of Model Trajectories at Round {round_num})", fontsize=16)
    plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
    plt.grid(True)
    explained_var = pca.explained_variance_ratio_.sum() * 100
    legend_handles[f'Explained Variance: {explained_var:.2f}%'] = plt.Line2D([0], [0], color='w')
    plt.legend(handles=legend_handles.values(), labels=legend_handles.keys())
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f"federated_learning_trajectory_map_R{round_num}.png")
    plt.savefig(filepath)
    plt.close()
    print(f"'{filepath}' 파일이 저장되었습니다.")


def save_results(output_dir, partition_report_df, args, ns_ids: Set[str], df_ids: Set[str], poisoned_classes: Set[int]):
    print(f"\n[결과 저장] 학습 결과를 '{output_dir}' 폴더에 저장하는 중...")
    
    results_filepath = os.path.join(output_dir, "federated_learning_results_v2.txt")
    with open(results_filepath, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n 데이터 분할 현황\n" + "="*80 + "\n")
        f.write(partition_report_df.to_string() + "\n\n")
        
        f.write("="*80 + "\n 라운드별 글로벌 모델 정확도\n" + "="*80 + "\n")
        accuracy_data = {'Overall_Accuracy': history['global_accuracy']}
        if poisoned_class_accuracy_history:
            clean_poisoned_list = sorted([int(c) for c in poisoned_classes])
            col_name = f"Poisoned_Acc({str(clean_poisoned_list).replace(' ', '')})"
            accuracy_data[col_name] = poisoned_class_accuracy_history
        
        accuracy_df = pd.DataFrame(accuracy_data)
        accuracy_df.index = pd.RangeIndex(1, len(accuracy_df) + 1, name='Round')
        f.write(accuracy_df.to_string(float_format="{:.4f}".format) + "\n\n")

        f.write("="*80 + "\n 라운드별 모델 합성 가중치\n" + "="*80 + "\n")
        weights_df = pd.DataFrame(aggregation_weights_history)
        weights_df.index = [f"Round {i+1}" for i in range(len(weights_df))]
        f.write(weights_df.to_string() + "\n\n")

        f.write("="*80 + "\n 라운드별 시간 측정 결과 (초)\n" + "="*80 + "\n")
        time_df = pd.DataFrame(time_history)
        column_order = [f'cli {i} train' for i in range(NUM_CLIENTS)] + ['agg time', 'cli eval time', 'global eval time', 'total time']
        existing_columns = [col for col in column_order if col in time_df.columns]
        time_df.index = [f"Round {i+1}" for i in range(len(time_df))]
        f.write(time_df[existing_columns].to_string() + "\n\n")
        
        total_simulation_time = time_df["total time"].sum()
        f.write(f"총 시뮬레이션 시간: {total_simulation_time:.2f}초\n\n")

        f.write("="*80 + "\n 시뮬레이션 파라미터\n" + "="*80 + "\n")
        f.write(f"NUM_CLIENTS: {NUM_CLIENTS}\n")
        f.write(f"NUM_ROUNDS: {NUM_ROUNDS}\n")
        f.write(f"LOCAL_EPOCHS: {LOCAL_EPOCHS}\n")
        f.write(f"LEARNING_RATE (Initial): {LEARNING_RATE}\n") 
        f.write(f"LR_SCHEDULER_STEP: {LR_SCHEDULER_STEP}\n") 
        f.write(f"LR_SCHEDULER_GAMMA: {LR_SCHEDULER_GAMMA}\n") 
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
        f.write(f"COMMITTEE_EVAL_SKIPPED: {args.s}\n")

    print(f"'{results_filepath}' 파일이 저장되었습니다.")
    
    # --- X축 눈금 간격을 동적으로 조절하는 로직 ---
    def get_xticks(total_rounds):
        if total_rounds > 100:
            step = 10
        elif total_rounds >= 50:
            step = 5
        else:
            return list(range(1, total_rounds + 1))
        
        ticks = list(range(step, total_rounds + 1, step))
        if 1 not in ticks: ticks.insert(0, 1)
        if total_rounds not in ticks: ticks.append(total_rounds)
        return ticks

    if history['global_accuracy']:
        rounds = range(1, len(history['global_accuracy']) + 1)
        xticks = get_xticks(len(rounds))

        acc_path = os.path.join(output_dir, "federated_learning_accuracy_v2.png")
        plt.figure(figsize=(10, 6)); plt.plot(rounds, history['global_accuracy'], marker='o', label='Overall Accuracy')
        plt.title("Federated Learning Performance: Accuracy Trend by Round"); plt.xlabel("Round"); plt.ylabel("Accuracy")
        plt.grid(True); plt.xticks(xticks); plt.yticks(np.arange(0, 1.1, 0.1)); plt.ylim(0, 1)
        plt.legend(); plt.savefig(acc_path); plt.close()
        print(f"'{acc_path}' 파일이 저장되었습니다.")
        
        if poisoned_class_accuracy_history:
            full_graph_path = os.path.join(output_dir, "federated_learning_full_graph_v2.png")
            plt.figure(figsize=(10, 6)); plt.plot(rounds, history['global_accuracy'], marker='o', linestyle='-', label='Overall Accuracy')
            clean_poisoned_list = sorted([int(c) for c in poisoned_classes])
            plt.plot(rounds, poisoned_class_accuracy_history, marker='x', linestyle='--', color='r', label=f'Poisoned Classes Acc') # ({clean_poisoned_list})
            plt.title("Overall vs. Poisoned Class Accuracy"); plt.xlabel("Round"); plt.ylabel("Accuracy")
            plt.grid(True); plt.xticks(xticks); plt.yticks(np.arange(0, 1.1, 0.1)); plt.ylim(0, 1)
            plt.legend(); plt.savefig(full_graph_path); plt.close()
            print(f"'{full_graph_path}' 파일이 저장되었습니다.")

    if aggregation_weights_history:
        weights_path = os.path.join(output_dir, "federated_learning_weights_v2.png")
        plt.figure(figsize=(12, 8)); weights_df = pd.DataFrame(aggregation_weights_history)
        rounds_weights = range(1, len(weights_df) + 1); xticks_weights = get_xticks(len(rounds_weights))
        
        cmap = plt.get_cmap('tab10') 
        
        for col in weights_df.columns:
            client_id_str = col.split('_')[1]
            if client_id_str in ns_ids or client_id_str in df_ids:
                color = 'red'
            else:
                color = cmap(int(client_id_str) % 10) 
            
            plt.plot(weights_df.index + 1, weights_df[col], marker='o', linestyle='--', markersize=4, label=f'Client {client_id_str}', color=color)
        
        plt.title("Changes in Normalized Aggregation Weights per Round"); plt.xlabel("Round"); plt.ylabel("Normalized Weight")
        plt.grid(True); plt.xticks(xticks_weights); plt.legend(title="Clients", loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(); plt.savefig(weights_path); plt.close()
        print(f"'{weights_path}' 파일이 저장되었습니다.")

    if poisoned_class_accuracy_history:
        poison_acc_path = os.path.join(output_dir, "federated_learning_poisoned_accuracy_v2.png")
        plt.figure(figsize=(10, 6)); rounds_poisoned = range(1, len(poisoned_class_accuracy_history) + 1)
        xticks_poisoned = get_xticks(len(rounds_poisoned)); clean_poisoned_list = sorted([int(c) for c in poisoned_classes])
        plt.plot(rounds_poisoned, poisoned_class_accuracy_history, marker='o', color='r', label=f'Poisoned Classes Acc ({clean_poisoned_list})')
        num_df = args.df[0] if args.df else 0; poison_level = args.df[1] if args.df and len(args.df) > 1 else 1
        title = f"Accuracy on Poisoned Classes\n(Attack: {num_df} nodes, {poison_level} labels each)"
        plt.title(title); plt.xlabel("Round"); plt.ylabel("Accuracy")
        plt.grid(True); plt.xticks(xticks_poisoned); plt.yticks(np.arange(0, 1.1, 0.1)); plt.ylim(0, 1)
        plt.legend(); plt.savefig(poison_acc_path); plt.close()
        print(f"'{poison_acc_path}' 파일이 저장되었습니다.")
        
    if not args.s and len(history['global_accuracy']) > 1:
        divergence_path = os.path.join(output_dir, "federated_learning_divergence_weights_v2.png")
        acc_diffs = np.diff(history['global_accuracy'])
        if len(acc_diffs) > 1:
            acc_diffs_after_R1 = acc_diffs[1:]
            drop_round_idx = np.argmin(acc_diffs_after_R1); rise_round_idx = np.argmax(acc_diffs_after_R1)
            sharpest_drop_round = drop_round_idx + 2; sharpest_rise_round = rise_round_idx + 2
            weights_df = pd.DataFrame(aggregation_weights_history)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
            drop_weights = weights_df.iloc[sharpest_drop_round - 1]; clients = [f"C{i}" for i in range(NUM_CLIENTS)]
            colors_drop = ['red' if str(i) in ns_ids or str(i) in df_ids else 'blue' for i in range(NUM_CLIENTS)]
            ax1.bar(clients, drop_weights, color=colors_drop)
            ax1.set_title(f"Weights at Sharpest Accuracy Drop\n(R{sharpest_drop_round-1} -> R{sharpest_drop_round}, Diff: {acc_diffs_after_R1[drop_round_idx]:.3f})")
            ax1.set_ylabel("Normalized Weight")
            rise_weights = weights_df.iloc[sharpest_rise_round - 1]
            colors_rise = ['red' if str(i) in ns_ids or str(i) in df_ids else 'blue' for i in range(NUM_CLIENTS)]
            ax2.bar(clients, rise_weights, color=colors_rise)
            ax2.set_title(f"Weights at Sharpest Accuracy Rise\n(R{sharpest_rise_round-1} -> R{sharpest_rise_round}, Diff: {acc_diffs_after_R1[rise_round_idx]:.3f})")
            fig.suptitle('Contribution Analysis: Weights at Critical Moments', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(divergence_path); plt.close()
            print(f"'{divergence_path}' 파일이 저장되었습니다.")

    if args.s and len(history['global_accuracy']) > 1 and model_params_history:
        acc_diffs = np.diff(history['global_accuracy'])
        if len(acc_diffs) > 1:
            acc_diffs_after_R1 = acc_diffs[1:]
            drop_round_idx = np.argmin(acc_diffs_after_R1); rise_round_idx = np.argmax(acc_diffs_after_R1)
            sharpest_drop_round = drop_round_idx + 2; sharpest_rise_round = rise_round_idx + 2
            
            if sharpest_drop_round in model_params_history and sharpest_rise_round in model_params_history:
                visualize_model_trajectories(
                    output_dir,
                    sharpest_drop_round,
                    sharpest_rise_round,
                    ns_ids, df_ids
                )


# ==============================================================================
# 6. 연합학습 시뮬레이션 실행
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="버전 2.12.2 (CIFAR-100): 기여도 및 악의적 노드 시뮬레이션")
    parser.add_argument("--s", action="store_true", help="위원회 평가 스킵")
    parser.add_argument("--ns", type=int, nargs='?', const=1, default=0, metavar='N', help="랜덤 노이즈 공격자 수")
    parser.add_argument("--df", type=int, nargs='+', metavar=('N_ATTACKERS', 'N_CLASSES'), help="데이터 포이즈닝 공격자 수 및 오염 클래스 수")
    parser.add_argument("--csf", type=float, default=CONTRIBUTION_SCALING_FACTOR, metavar='F', help="기여도 가중치 배수 (Contribution Scaling Factor)")
    args = parser.parse_args()
    
    # --- 결과 저장 폴더 생성 ---
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    timestamp = now.strftime("%m%d_%H%M")
    options_list = []
    if args.ns > 0: options_list.append(f"ns{args.ns}")
    if args.df: 
        num_df = args.df[0]
        poison_level = args.df[1] if len(args.df) > 1 else 1
        options_list.append(f"df{num_df}L{poison_level}")
    if args.s: options_list.append("skip")
    if args.csf != 1.0: options_list.append(f"csf{args.csf}")
    options_list.append(f"ew{EWMA_BETA}")
    options_str = f"_[{','.join(options_list)}]" if options_list else ""
    # [수정] 버전 2.12.2로 변경
    output_dir = f"{timestamp}_v2.12.2_R{NUM_ROUNDS}{options_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80, f"\n버전 2.12.2 (CIFAR-100): 기여도 기반 연합학습 시뮬레이션을 시작합니다.\n{'='*80}")
    
    # [수정] 반환값에 num_classes 추가
    train_subsets, committee_test_loaders, partition_report_df, num_classes = prepare_datasets()
    
    print("\n 최종 훈련 데이터 분할 현황")
    print("="*80); print(partition_report_df.to_string()); print("="*80 + "\n")

    # --- 악의적인 노드 ID 및 오염 클래스 설정 ---
    num_ns = args.ns
    num_df = args.df[0] if args.df else 0
    poison_level = args.df[1] if args.df and len(args.df) > 1 else (1 if num_df > 0 else 0)
    
    total_malicious = num_ns + num_df
    client_ids_str = [str(i) for i in range(NUM_CLIENTS)]
    
    df_ids = set(client_ids_str[NUM_CLIENTS - num_df:])
    ns_ids = set(client_ids_str[NUM_CLIENTS - total_malicious : NUM_CLIENTS - num_df])
    
    shared_poison_map: Dict[int, int] = {}
    if df_ids:
        np.random.seed(RANDOM_SEED)
        # [수정] 10 -> num_classes
        all_classes = list(range(num_classes))
        num_poison_classes_to_select = poison_level
        available_sources = list(range(num_classes))
        np.random.shuffle(available_sources)
        source_classes = available_sources[:num_poison_classes_to_select]
        
        for source in source_classes:
            target_options = [c for c in all_classes if c != source and c not in shared_poison_map.values()]
            if not target_options: target_options = [c for c in all_classes if c != source]
            shared_poison_map[source] = np.random.choice(target_options)

    poisoned_classes_to_track = set(shared_poison_map.keys())

    if ns_ids: print(f"[설정] 노이즈 공격 노드 수: {len(ns_ids)}개, ID: {sorted(list(ns_ids))}")
    if df_ids: print(f"[설정] 데이터 포이즈닝 노드 수: {len(df_ids)}개, ID: {sorted(list(df_ids))}, 오염 레벨: {poison_level}")
    if poisoned_classes_to_track: print(f"[설정] 추적할 오염 클래스: {sorted([int(c) for c in poisoned_classes_to_track])}")

    def client_fn(cid: str) -> fl.client.Client:
        # [수정] get_model(10) -> get_model(num_classes)
        model = get_model(num_classes)
        train_subset = train_subsets[int(cid)]
        if cid in ns_ids: 
            # [수정] num_classes 전달
            return NoiseClient(cid, num_classes).to_client()
        elif cid in df_ids: return DataPoisoningClient(cid, train_subset, model, shared_poison_map).to_client()
        else: return CifarClient(cid, train_subset, model).to_client()

    strategy = FedContrib(
        committee_test_loaders=committee_test_loaders,
        num_classes=num_classes, # [수정] num_classes 전달
        beta=EWMA_BETA, skip_committee_eval=args.s,
        poisoned_classes=poisoned_classes_to_track, scaling_factor=args.csf,
        fraction_fit=1.0, min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS, evaluate_fn=None,
    )
    
    simulation_start_time = time.time()
    try:
        fl.simulation.start_simulation(
            client_fn=client_fn, num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
            client_resources={"num_cpus": 2, "num_gpus": 0.5} if DEVICE.type == "cuda" else {"num_cpus": 2}
        )
    except Exception as e:
        print(f"\n[오류] 시뮬레이션 중 예외가 발생했습니다: {e}")
    finally:
        total_time = time.time() - simulation_start_time
        print("\n" + "=" * 80, f"\n연합학습 시뮬레이션 종료. (총 소요 시간: {total_time:.2f}초)\n", "=" * 80)
        if history["global_accuracy"]:
            save_results(output_dir, partition_report_df, args, ns_ids, df_ids, poisoned_classes_to_track)
        else:
            print("[알림] 학습이 완료되기 전에 종료되어 저장할 결과가 없습니다.")

if __name__ == "__main__":
    main()