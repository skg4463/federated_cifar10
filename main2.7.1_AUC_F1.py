# ==============================================================================
# 코드 버전 및 기능 요약
# ==============================================================================
# 버전: 2.7
#
# 주요 기능:
# 1. 기여도 기반 연합학습: 독립된 '위원회(Committee)'가 각 노드의 성능을 평가하고,
#    이를 바탕으로 모델 통합 가중치를 동적으로 조절합니다. (FedContrib)
# 2. 악의적 노드 시뮬레이션:
#    - 랜덤 노이즈 공격 (--ns): 무작위 파라미터를 제출하여 학습을 방해합니다.
#    - 데이터 포이즈닝 공격 (--df): 데이터 라벨을 조작하여 특정 클래스의 성능을 저하시킵니다.
#    - 공격자 수와 오염시킬 클래스 수를 지정할 수 있습니다.   
# 3. 공격 강도 및 방식 조절:
#    - 데이터 포이즈닝 공격 시, 공격자 수와 오염시킬 클래스 수를 지정할 수 있습니다.
#    - 모든 공격자가 완전히 동일한 클래스들을 공격하는 '집중 공격'을 수행합니다.
# 4. 상세 분석 및 시각화:
#    - 전체 정확도, F1 Score, AUC, 모델 통합 가중치, 오염된 클래스 성능을 각각 그래프로 시각화합니다.
#    - 라운드 수에 따라 그래프의 X축 눈금 간격을 자동으로 조절합니다.
#    - 모든 실험 설정과 시간 측정 결과를 체계적인 폴더 구조와 로그 파일로 저장합니다.
# 5. 실험 재현성: 랜덤 시드를 고정하여 동일한 조건에서 반복 실험 및 결과 비교가 가능합니다.
# 6. 유연한 실행 옵션:
#    - --s: 위원회 평가를 건너뛰어 빠른 테스트를 진행할 수 있습니다.
#    - --csf [F]: 기여도 점수 차이를 증폭시키는 배수를 설정합니다. (기본값: 1.0)
#
# 명령어: python mainv2.7.py --ns 2 --df 2 1 --s --csf 1.5
#  (노이즈 공격자 2명, 데이터포이즈닝 공격자 2명(각각 1개 클래스 오염), 위원회 평가 건너뛰기(일반연합학습), 기여도 증폭 배수 1.5)
# 명령어: python mainv2.7.py --ns 0 --df 3 2 --csf 2.0
#  (노이즈 공격자 없음, 데이터포이즈닝 공격자 3명(각각 2개 클래스 오염), 위원회 평가 수행(블록체인연합학습), 기여도 증폭 배수 2.0)
# ==============================================================================


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision.datasets import CIFAR10
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
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ==============================================================================
# 1. 하이퍼파라미터 및 설정
# ==============================================================================
# --- 기본 학습 파라미터 ---
NUM_CLIENTS = 10           # 학습에 참여하는 클라이언트(노드)의 수
BATCH_SIZE = 32            # 학습 시 사용되는 데이터 배치의 크기
NUM_ROUNDS = 50            # 총 연합학습 라운드 수
LOCAL_EPOCHS = 2           # 각 클라이언트가 라운드마다 수행하는 로컬 학습 횟수
LEARNING_RATE = 0.01       # 모델 학습률

# --- 데이터 분할 파라미터 ---
DIRICHLET_ALPHA = 0.5      # 데이터의 Non-IID 조절 디리클레 분포 파라미터
MIN_SAMPLES_PER_CLASS = 100 # 각 클라이언트가 클래스별로 보장받는 최소 샘플 수
RANDOM_SEED = 42           # 재현 가능 실험을 위한 랜덤 시드

# --- 기여도 측정 파라미터 ---
NUM_COMMITTEE_MEMBERS = 3  # 학습 노드의 성능을 평가할 Committee의 수
EWMA_BETA = 0.3            # 지수가중이동평균(EWMA) 반영률 조정 파라미터
CONTRIBUTION_SCALING_FACTOR = 1.5 # 기여도 증폭 배수

# --- 시스템 설정 ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"학습 장치: {DEVICE}")

# --- 결과 기록용 전역 변수 ---
history = {"global_accuracy": [], "global_f1": [], "global_auc": []}
poisoned_class_accuracy_history = []
poisoned_class_f1_history = []
poisoned_class_auc_history = []
aggregation_weights_history = defaultdict(list)
time_history = defaultdict(list)

# ==============================================================================
# 2. 데이터셋 준비
# ==============================================================================
def prepare_datasets() -> Tuple[List[Dataset], List[DataLoader], pd.DataFrame]:
    """훈련 데이터셋과 위원회용 테스트 데이터셋을 준비합니다."""
    np.random.seed(RANDOM_SEED)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    data_path = './data'
    trainset = CIFAR10(root=data_path, train=True, download=True, transform=transform)
    testset = CIFAR10(root=data_path, train=False, download=True, transform=transform)
    
    # --- 훈련 데이터셋 Non-IID 분할 ---
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
    partition_report_df = create_partition_report(client_data_indices, trainset.targets)

    # --- 위원회용 테스트 데이터셋 IID 분할 ---
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
        print(f" - 위원회 멤버 {i+1} 할당 데이터 수: {len(dataset)}개")

    return train_subsets, committee_test_loaders, partition_report_df

def create_partition_report(client_indices, all_labels):
    report = []
    all_labels_np = np.array(all_labels)
    for i, indices in enumerate(client_indices):
        labels = all_labels_np[indices]
        counts = {cls: count for cls, count in zip(*np.unique(labels, return_counts=True))}
        row = {"Client": i, "Num Samples": len(indices)}
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
        self.set_parameters(parameters)
        self.model.to(DEVICE).train()
        optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=0.9)
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
    def __init__(self, cid: str):
        self.cid = cid

    def get_parameters(self, config):
        print(f"  - [경고] 노이즈 공격 노드 {self.cid}가 랜덤 파라미터를 제출합니다.")
        return [val.cpu().numpy() for _, val in get_model(10).state_dict().items()]

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
    """모델의 성능을 평가합니다. accuracy, F1, AUC를 모두 계산합니다."""
    model.to(DEVICE).eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in testloader:
            if target_classes:
                mask = torch.tensor([label in target_classes for label in labels])
                images, labels = images[mask], labels[mask]
                if images.size(0) == 0:
                    continue
            
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    if len(all_labels) == 0:
        return {"accuracy": 0.0, "f1": 0.0, "auc": 0.0}
    
    # Accuracy 계산
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    
    # F1 Score 계산 (macro average)
    try:
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    except:
        f1 = 0.0
    
    # AUC 계산 (macro average, one-vs-rest)
    try:
        # 각 클래스별로 이진분류 AUC 계산 후 평균
        all_probs_array = np.array(all_probs)
        all_labels_array = np.array(all_labels)
        
        if target_classes:
            # 타겟 클래스만 고려
            unique_classes = sorted(list(set(all_labels_array)))
        else:
            # 전체 클래스 고려
            unique_classes = list(range(10))
        
        if len(unique_classes) <= 1:
            auc = 0.5  # 클래스가 1개 이하면 AUC 의미 없음
        else:
            auc_scores = []
            for class_id in unique_classes:
                if class_id < all_probs_array.shape[1]:  # 확률 배열 범위 체크
                    y_true_binary = (all_labels_array == class_id).astype(int)
                    y_prob_binary = all_probs_array[:, class_id]
                    
                    # 해당 클래스가 실제로 존재하는지 확인
                    if len(np.unique(y_true_binary)) > 1:
                        auc_class = roc_auc_score(y_true_binary, y_prob_binary)
                        auc_scores.append(auc_class)
            
            auc = np.mean(auc_scores) if auc_scores else 0.5
    except:
        auc = 0.5
    
    return {"accuracy": accuracy, "f1": f1, "auc": auc}

class FedContrib(FedAvg):
    def __init__(
        self,
        committee_test_loaders: List[DataLoader],
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
        self.model_for_eval = get_model(10)
        self.ewma_scores = defaultdict(float)
        self.poisoned_classes = poisoned_classes if poisoned_classes else set()

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

        print("\n  - 글로벌 모델 성능 평가")
        global_eval_start = time.time()
        global_state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model_for_eval.state_dict().keys(), aggregated_ndarrays)})
        self.model_for_eval.load_state_dict(global_state_dict)
        full_test_loader = DataLoader(ConcatDataset([loader.dataset for loader in self.committee_test_loaders]), batch_size=BATCH_SIZE*2)
        global_metrics = evaluate_model(self.model_for_eval, full_test_loader)
        time_history["global eval time"].append(time.time() - global_eval_start)
        
        print(f"    - 전체 정확도: {global_metrics['accuracy']:.4f}")
        print(f"    - 전체 F1 Score: {global_metrics['f1']:.4f}")
        print(f"    - 전체 AUC: {global_metrics['auc']:.4f}")
        
        history["global_accuracy"].append(global_metrics['accuracy'])
        history["global_f1"].append(global_metrics['f1'])
        history["global_auc"].append(global_metrics['auc'])
        
        if self.poisoned_classes:
            poisoned_metrics = evaluate_model(self.model_for_eval, full_test_loader, target_classes=list(self.poisoned_classes))
            poisoned_accuracy = poisoned_metrics['accuracy']
            poisoned_f1 = poisoned_metrics['f1']
            poisoned_auc = poisoned_metrics['auc']
            
            poisoned_class_accuracy_history.append(poisoned_accuracy)
            poisoned_class_f1_history.append(poisoned_f1)
            poisoned_class_auc_history.append(poisoned_auc)
            
            clean_poisoned_list = sorted([int(c) for c in self.poisoned_classes])
            print(f"    - 오염된 클래스({clean_poisoned_list}) 정확도: {poisoned_accuracy:.4f}")
            print(f"    - 오염된 클래스({clean_poisoned_list}) F1 Score: {poisoned_f1:.4f}")
            print(f"    - 오염된 클래스({clean_poisoned_list}) AUC: {poisoned_auc:.4f}")

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
def save_results(output_dir, partition_report_df, args, poisoned_classes: Set[int]):
    print(f"\n[결과 저장] 학습 결과를 '{output_dir}' 폴더에 저장하는 중...")
    
    results_filepath = os.path.join(output_dir, "federated_learning_results_v2.txt")
    with open(results_filepath, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n 데이터 분할 현황\n" + "="*80 + "\n")
        f.write(partition_report_df.to_string() + "\n\n")
        
        f.write("="*80 + "\n 라운드별 글로벌 모델 성능\n" + "="*80 + "\n")
        performance_data = {
            'Overall_Accuracy': history['global_accuracy'],
            'Overall_F1': history['global_f1'],
            'Overall_AUC': history['global_auc']
        }
        
        if poisoned_class_accuracy_history:
            clean_poisoned_list = sorted([int(c) for c in poisoned_classes])
            col_prefix = f"Poisoned({str(clean_poisoned_list).replace(' ', '')})"
            performance_data[f'{col_prefix}_Acc'] = poisoned_class_accuracy_history
            performance_data[f'{col_prefix}_F1'] = poisoned_class_f1_history
            performance_data[f'{col_prefix}_AUC'] = poisoned_class_auc_history
        
        performance_df = pd.DataFrame(performance_data)
        performance_df.index = pd.RangeIndex(1, len(performance_df) + 1, name='Round')
        f.write(performance_df.to_string(float_format="{:.4f}".format) + "\n\n")

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
    
    # --- [수정] X축 눈금 간격을 동적으로 조절하는 로직 ---
    def get_xticks(total_rounds):
        if total_rounds > 100:
            step = 10
        elif total_rounds >= 50:
            step = 5
        else:
            return list(range(1, total_rounds + 1))
        
        ticks = list(range(step, total_rounds + 1, step))
        if 1 not in ticks:
            ticks.insert(0, 1)
        return ticks

    if history['global_accuracy']:
        rounds = range(1, len(history['global_accuracy']) + 1)
        xticks = get_xticks(len(rounds))

        # 전체 성능 그래프 (Accuracy, F1, AUC)
        overall_perf_path = os.path.join(output_dir, "federated_learning_overall_performance_v2.png")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Accuracy
        ax1.plot(rounds, history['global_accuracy'], marker='o', color='blue', label='Overall Accuracy')
        ax1.set_title("Overall Accuracy Trend")
        ax1.set_ylabel("Accuracy")
        ax1.grid(True); ax1.set_xticks(xticks); ax1.set_yticks(np.arange(0, 1.1, 0.1)); ax1.set_ylim(0, 1)
        ax1.legend()
        
        # F1 Score
        ax2.plot(rounds, history['global_f1'], marker='s', color='green', label='Overall F1 Score')
        ax2.set_title("Overall F1 Score Trend")
        ax2.set_ylabel("F1 Score")
        ax2.grid(True); ax2.set_xticks(xticks); ax2.set_yticks(np.arange(0, 1.1, 0.1)); ax2.set_ylim(0, 1)
        ax2.legend()
        
        # AUC
        ax3.plot(rounds, history['global_auc'], marker='^', color='red', label='Overall AUC')
        ax3.set_title("Overall AUC Trend")
        ax3.set_xlabel("Round")
        ax3.set_ylabel("AUC")
        ax3.grid(True); ax3.set_xticks(xticks); ax3.set_yticks(np.arange(0, 1.1, 0.1)); ax3.set_ylim(0, 1)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(overall_perf_path)
        print(f"'{overall_perf_path}' 파일이 저장되었습니다.")
        plt.close()

        # 통합 성능 그래프 (모든 지표를 한 그래프에)
        combined_perf_path = os.path.join(output_dir, "federated_learning_combined_performance_v2.png")
        plt.figure(figsize=(12, 7))
        plt.plot(rounds, history['global_accuracy'], marker='o', linestyle='-', label='Overall Accuracy')
        plt.plot(rounds, history['global_f1'], marker='s', linestyle='--', label='Overall F1 Score')
        plt.plot(rounds, history['global_auc'], marker='^', linestyle='-.', label='Overall AUC')
        
        plt.title("Overall Performance Metrics Comparison")
        plt.xlabel("Round")
        plt.ylabel("Score")
        plt.grid(True); plt.xticks(xticks); plt.yticks(np.arange(0, 1.1, 0.1)); plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(combined_perf_path)
        print(f"'{combined_perf_path}' 파일이 저장되었습니다.")
        plt.close()

        # 오염된 클래스 성능 그래프
        if poisoned_class_accuracy_history:
            poisoned_perf_path = os.path.join(output_dir, "federated_learning_poisoned_performance_v2.png")
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            clean_poisoned_list = sorted([int(c) for c in poisoned_classes])
            
            # Poisoned Accuracy
            ax1.plot(rounds, poisoned_class_accuracy_history, marker='o', color='red', label=f'Poisoned Classes Acc ({clean_poisoned_list})')
            ax1.set_title(f"Poisoned Classes Accuracy Trend ({clean_poisoned_list})")
            ax1.set_ylabel("Accuracy")
            ax1.grid(True); ax1.set_xticks(xticks); ax1.set_yticks(np.arange(0, 1.1, 0.1)); ax1.set_ylim(0, 1)
            ax1.legend()
            
            # Poisoned F1
            ax2.plot(rounds, poisoned_class_f1_history, marker='s', color='orange', label=f'Poisoned Classes F1 ({clean_poisoned_list})')
            ax2.set_title(f"Poisoned Classes F1 Score Trend ({clean_poisoned_list})")
            ax2.set_ylabel("F1 Score")
            ax2.grid(True); ax2.set_xticks(xticks); ax2.set_yticks(np.arange(0, 1.1, 0.1)); ax2.set_ylim(0, 1)
            ax2.legend()
            
            # Poisoned AUC
            ax3.plot(rounds, poisoned_class_auc_history, marker='^', color='purple', label=f'Poisoned Classes AUC ({clean_poisoned_list})')
            ax3.set_title(f"Poisoned Classes AUC Trend ({clean_poisoned_list})")
            ax3.set_xlabel("Round")
            ax3.set_ylabel("AUC")
            ax3.grid(True); ax3.set_xticks(xticks); ax3.set_yticks(np.arange(0, 1.1, 0.1)); ax3.set_ylim(0, 1)
            ax3.legend()
            
            plt.tight_layout()
            plt.savefig(poisoned_perf_path)
            print(f"'{poisoned_perf_path}' 파일이 저장되었습니다.")
            plt.close()

            # 전체 vs 오염 비교 그래프
            comparison_path = os.path.join(output_dir, "federated_learning_overall_vs_poisoned_v2.png")
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            
            # Accuracy 비교
            ax1.plot(rounds, history['global_accuracy'], marker='o', linestyle='-', label='Overall Accuracy')
            ax1.plot(rounds, poisoned_class_accuracy_history, marker='x', linestyle='--', color='r', label=f'Poisoned Classes Acc ({clean_poisoned_list})')
            ax1.set_title("Overall vs. Poisoned Classes Accuracy")
            ax1.set_ylabel("Accuracy")
            ax1.grid(True); ax1.set_xticks(xticks); ax1.set_yticks(np.arange(0, 1.1, 0.1)); ax1.set_ylim(0, 1)
            ax1.legend()
            
            # F1 비교
            ax2.plot(rounds, history['global_f1'], marker='s', linestyle='-', label='Overall F1 Score')
            ax2.plot(rounds, poisoned_class_f1_history, marker='x', linestyle='--', color='orange', label=f'Poisoned Classes F1 ({clean_poisoned_list})')
            ax2.set_title("Overall vs. Poisoned Classes F1 Score")
            ax2.set_ylabel("F1 Score")
            ax2.grid(True); ax2.set_xticks(xticks); ax2.set_yticks(np.arange(0, 1.1, 0.1)); ax2.set_ylim(0, 1)
            ax2.legend()
            
            # AUC 비교
            ax3.plot(rounds, history['global_auc'], marker='^', linestyle='-', label='Overall AUC')
            ax3.plot(rounds, poisoned_class_auc_history, marker='x', linestyle='--', color='purple', label=f'Poisoned Classes AUC ({clean_poisoned_list})')
            ax3.set_title("Overall vs. Poisoned Classes AUC")
            ax3.set_xlabel("Round")
            ax3.set_ylabel("AUC")
            ax3.grid(True); ax3.set_xticks(xticks); ax3.set_yticks(np.arange(0, 1.1, 0.1)); ax3.set_ylim(0, 1)
            ax3.legend()
            
            plt.tight_layout()
            plt.savefig(comparison_path)
            print(f"'{comparison_path}' 파일이 저장되었습니다.")
            plt.close()

    if aggregation_weights_history:
        weights_path = os.path.join(output_dir, "federated_learning_weights_v2.png")
        plt.figure(figsize=(12, 8))
        weights_df = pd.DataFrame(aggregation_weights_history)
        rounds_weights = range(1, len(weights_df) + 1)
        xticks_weights = get_xticks(len(rounds_weights))
        for col in weights_df.columns:
            client_id = col.split('_')[1]
            plt.plot(weights_df.index + 1, weights_df[col], marker='o', linestyle='--', markersize=4, label=f'Client {client_id}')
        
        plt.title("Changes in Normalized Aggregation Weights per Round")
        plt.xlabel("Round")
        plt.ylabel("Normalized Weight")
        plt.grid(True)
        plt.xticks(xticks_weights)
        plt.legend(title="Clients", loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(weights_path)
        print(f"'{weights_path}' 파일이 저장되었습니다.")
        plt.close()

# ==============================================================================
# 6. 연합학습 시뮬레이션 실행
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="버전 2.7: 기여도 및 악의적 노드 시뮬레이션")
    parser.add_argument("--s", action="store_true", help="위원회 평가 스킵")
    parser.add_argument("--ns", type=int, nargs='?', const=1, default=0, metavar='N', help="랜덤 노이즈 공격자 수")
    parser.add_argument("--df", type=int, nargs='+', metavar=('N_ATTACKERS', 'N_CLASSES'), help="데이터 포이즈닝 공격자 수 및 오염 클래스 수")
    parser.add_argument("--csf", type=float, default=CONTRIBUTION_SCALING_FACTOR, metavar='F', help="기여도 가중치 배수 (Contribution Scaling Factor)")
    args = parser.parse_args()
    
    # --- 결과 저장 폴더 생성 ---
    now = datetime.now()
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
    output_dir = f"{timestamp}_v2.7_R{NUM_ROUNDS}{options_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80, "\n버전 2.7: 기여도 기반 연합학습 시뮬레이션을 시작합니다.\n", "=" * 80)
    
    train_subsets, committee_test_loaders, partition_report_df = prepare_datasets()
    
    print("\n 최종 훈련 데이터 분할 현황")
    print("="*80)
    print(partition_report_df.to_string())
    print("="*80 + "\n")

    # --- 악의적인 노드 ID 및 오염 클래스 설정 ---
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
        model = get_model(10)
        train_subset = train_subsets[int(cid)]
        
        if cid in ns_ids:
            return NoiseClient(cid).to_client()
        elif cid in df_ids:
            return DataPoisoningClient(cid, train_subset, model, shared_poison_map).to_client()
        else:
            return CifarClient(cid, train_subset, model).to_client()

    strategy = FedContrib(
        committee_test_loaders=committee_test_loaders,
        beta=EWMA_BETA,
        skip_committee_eval=args.s,
        poisoned_classes=poisoned_classes_to_track,
        scaling_factor=args.csf,
        fraction_fit=1.0,
        fraction_evaluate=0.0, # flower 클라이언트 레벨 정확도 평가 스킵
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=None,
    )
    
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
        if history["global_accuracy"]:
            save_results(output_dir, partition_report_df, args, poisoned_classes_to_track)
        else:
            print("[알림] 학습이 완료되기 전에 종료되어 저장할 결과가 없습니다.")

if __name__ == "__main__":
    main()