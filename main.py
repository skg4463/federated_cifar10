import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision.models as models

import flwr as fl
from flwr.common import (
    Metrics,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

import numpy as np
from collections import OrderedDict, defaultdict
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
import time
import argparse # 커맨드 라인 인자 처리를 위한 모듈

# ==============================================================================
# 1. 하이퍼파라미터 및 설정
# ==============================================================================
NUM_CLIENTS = 10
BATCH_SIZE = 32
NUM_ROUNDS = 100
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.01
DIRICHLET_ALPHA = 0.5
MIN_SAMPLES_PER_CLASS = 30
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"학습 장치: {DEVICE}")

# 성능 및 시간 기록을 위한 전역 변수
history = {"global_accuracy": [], "global_auc": [], "global_loss": []}
individual_client_performance = defaultdict(dict)
time_history = defaultdict(list)

# ==============================================================================
# 2. 데이터셋 준비 및 Non-IID 분할
# ==============================================================================
def load_and_partition_data() -> Tuple[List[Dataset], DataLoader, int, pd.DataFrame]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    data_path = './data'
    trainset = CIFAR10(root=data_path, train=True, download=True, transform=transform)
    testset = CIFAR10(root=data_path, train=False, download=True, transform=transform)
    num_classes = len(trainset.classes)

    class_indices = [np.where(np.array(trainset.targets) == i)[0].tolist() for i in range(num_classes)]
    for indices in class_indices:
        np.random.shuffle(indices)

    client_data_indices = [[] for _ in range(NUM_CLIENTS)]

    for client_id in range(NUM_CLIENTS):
        for class_id in range(num_classes):
            num_to_take = MIN_SAMPLES_PER_CLASS
            taken_indices = class_indices[class_id][:num_to_take]
            class_indices[class_id] = class_indices[class_id][num_to_take:]
            client_data_indices[client_id].extend(taken_indices)

    for class_id in range(num_classes):
        if not class_indices[class_id]: continue
        remaining_indices_for_class = class_indices[class_id]
        proportions = np.random.dirichlet(np.repeat(DIRICHLET_ALPHA, NUM_CLIENTS))
        samples_to_assign = (proportions * len(remaining_indices_for_class)).astype(int)
        remainder = len(remaining_indices_for_class) - samples_to_assign.sum()
        if remainder > 0: samples_to_assign[np.argmax(samples_to_assign)] += remainder
        
        start_idx = 0
        for client_id in range(NUM_CLIENTS):
            num_to_give = samples_to_assign[client_id]
            end_idx = start_idx + num_to_give
            client_data_indices[client_id].extend(remaining_indices_for_class[start_idx:end_idx])
            start_idx = end_idx

    partition_report_df = create_partition_report(client_data_indices, trainset.targets)
    train_subsets = [Subset(trainset, indices) for indices in client_data_indices]
    testloader = DataLoader(testset, batch_size=BATCH_SIZE * 2, shuffle=False)
    
    return train_subsets, testloader, num_classes, partition_report_df

def create_partition_report(client_indices, all_labels):
    report = []
    all_labels_np = np.array(all_labels)
    for i, indices in enumerate(client_indices):
        client_labels = all_labels_np[indices]
        class_counts = {cls: count for cls, count in zip(*np.unique(client_labels, return_counts=True))}
        row = {"Client": i, "Num Samples": len(indices)}
        for c in range(10): row[f"Class {c}"] = class_counts.get(c, 0)
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
    def __init__(self, cid: str, train_subset: Subset, model: nn.Module):
        self.cid = cid
        self.model = model
        self.trainloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        # [수정] 클라이언트 학습 시간 측정을 위해 시작 시간 기록
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
        
        # [수정] 학습 완료 후, 실제 소요 시간을 계산하여 결과에 포함
        fit_duration = time.time() - fit_start_time
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"cid": self.cid, "fit_duration": fit_duration}

# ==============================================================================
# 4. 평가 함수 및 커스텀 전략
# ==============================================================================
def evaluate_model(model: nn.Module, testloader: DataLoader) -> Dict[str, float]:
    model.to(DEVICE).eval()
    loss, correct, total = 0, 0, 0
    all_labels, all_probs = [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    return {
        "loss": loss / len(testloader),
        "accuracy": correct / total,
        "auc": roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    }

class CustomFedAvg(FedAvg):
    def __init__(self, testloader: DataLoader, skip_individual_eval: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.testloader = testloader
        self.skip_individual_eval = skip_individual_eval
        self.model_for_eval = get_model(10) 

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        
        round_start_time = time.time()

        for _, fit_res in results:
            client_id = fit_res.metrics.get("cid", "unknown")
            fit_duration = fit_res.metrics.get("fit_duration", 0)
            time_history[f"client_{client_id}_fit_duration"].append(fit_duration)
        
        aggregation_start_time = time.time()
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        time_history["aggregation_duration"].append(time.time() - aggregation_start_time)

        if aggregated_parameters is not None:
            print("\n" + "-"*80)
            print(f"** 라운드 {server_round} 서버 평가 결과 **")
            
            if not self.skip_individual_eval:
                print("  - 개별 클라이언트 모델 성능 (통합 전)")
                eval_individual_start_time = time.time()
                for _, fit_res in results:
                    client_id = fit_res.metrics.get("cid", "unknown")
                    client_params_numpy = parameters_to_ndarrays(fit_res.parameters)
                    client_state_dict = OrderedDict(
                        {k: torch.tensor(v) for k, v in zip(self.model_for_eval.state_dict().keys(), client_params_numpy)}
                    )
                    self.model_for_eval.load_state_dict(client_state_dict)
                    metrics = evaluate_model(self.model_for_eval, self.testloader)
                    print(f"    - 클라이언트 {client_id} 제출 모델: Acc {metrics['accuracy']:.4f}, AUC {metrics['auc']:.4f}")
                    individual_client_performance[server_round][client_id] = metrics
                time_history["individual_eval_duration"].append(time.time() - eval_individual_start_time)
            else:
                print("  - 개별 클라이언트 모델 성능 평가를 건너뛰었습니다.")
                time_history["individual_eval_duration"].append(0)


            print("  - 통합 글로벌 모델 성능")
            eval_global_start_time = time.time()
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            global_state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in zip(self.model_for_eval.state_dict().keys(), aggregated_ndarrays)}
            )
            self.model_for_eval.load_state_dict(global_state_dict)
            global_metrics = evaluate_model(self.model_for_eval, self.testloader)
            time_history["global_eval_duration"].append(time.time() - eval_global_start_time)
            
            print(f"    -   >> 글로벌 모델: Loss {global_metrics['loss']:.4f}, Acc {global_metrics['accuracy']:.4f}, AUC {global_metrics['auc']:.4f}")
            
            total_round_duration = time.time() - round_start_time
            time_history["total_round_duration"].append(total_round_duration)
            
            print("  - 시간 측정:")
            print(f"    - 서버 통합 시간: {time_history['aggregation_duration'][-1]:.2f}초")
            if not self.skip_individual_eval:
                print(f"    - 개별 모델 평가 시간: {time_history['individual_eval_duration'][-1]:.2f}초")
            print(f"    - 글로벌 모델 평가 시간: {time_history['global_eval_duration'][-1]:.2f}초")
            print(f"    - 총 라운드 시간: {total_round_duration:.2f}초")
            print("-" * 80)
            
            history["global_loss"].append(global_metrics['loss'])
            history["global_accuracy"].append(global_metrics['accuracy'])
            history["global_auc"].append(global_metrics['auc'])

        return aggregated_parameters, aggregated_metrics

# ==============================================================================
# 5. 결과 저장 및 시각화
# ==============================================================================
def save_results(partition_report_df):
    print("\n[결과 저장] 학습 결과를 파일로 저장하는 중...")
    
    with open("federated_learning_results.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n 데이터 분할 현황\n" + "="*80 + "\n")
        f.write(partition_report_df.to_string() + "\n\n")
        
        f.write("="*80 + "\n 라운드별 글로벌 모델 평가 결과\n" + "="*80 + "\n")
        global_results_df = pd.DataFrame({
            'Round': range(1, len(history['global_accuracy']) + 1),
            'Loss': history['global_loss'],
            'Accuracy': history['global_accuracy'],
            'AUC': history['global_auc']
        }).set_index('Round')
        f.write(global_results_df.to_string() + "\n\n")

        if individual_client_performance:
            f.write("="*80 + "\n 라운드별 개별 클라이언트 모델 성능 (통합 전)\n" + "="*80 + "\n")
            for round_num, client_metrics in individual_client_performance.items():
                f.write(f"--- Round {round_num} ---\n")
                sorted_clients = sorted(client_metrics.items(), key=lambda item: int(item[0]))
                for client_id, metrics in sorted_clients:
                    f.write(f"  Client {client_id}: Accuracy={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}\n")
                f.write("\n")
        
        # [수정] 시간 측정 결과의 컬럼명을 변경하고 순서를 지정하여 저장
        f.write("="*80 + "\n 라운드별 시간 측정 결과 (초)\n" + "="*80 + "\n")
        time_df = pd.DataFrame(time_history)
        
        # 컬럼명 변경 맵 생성
        rename_map = {
            'aggregation_duration': 'agg time',
            'individual_eval_duration': 'cli eval time',
            'global_eval_duration': 'global eval time',
            'total_round_duration': 'total time'
        }
        # 클라이언트 학습 시간 컬럼명 변경
        for i in range(NUM_CLIENTS):
            rename_map[f'client_{i}_fit_duration'] = f'cli {i} train'
        
        time_df.rename(columns=rename_map, inplace=True)
        
        # 출력할 컬럼 순서 지정
        column_order = [f'cli {i} train' for i in range(NUM_CLIENTS)]
        column_order += ['agg time', 'cli eval time', 'global eval time', 'total time']
        
        # 데이터프레임에 존재하는 컬럼만 필터링하여 순서 적용
        existing_columns_in_order = [col for col in column_order if col in time_df.columns]
        
        time_df.index = [f"Round {i+1}" for i in range(len(time_df))]
        f.write(time_df[existing_columns_in_order].to_string() + "\n\n")
        
        total_simulation_time = time_df["total time"].sum()
        f.write(f"총 시뮬레이션 시간: {total_simulation_time:.2f}초\n")


    print("'federated_learning_results.txt' 파일이 저장되었습니다.")
    
    if history['global_accuracy']:
        plt.figure(figsize=(10, 6))
        rounds = range(1, len(history['global_accuracy']) + 1)
        plt.plot(rounds, history['global_accuracy'], marker='o')
        
        plt.title("Federated Learning Performance: Accuracy Trend by Round")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.xticks(rounds)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.ylim(0, 1)
        
        plt.savefig("federated_learning_accuracy.png")
        print("'federated_learning_accuracy.png' 파일이 저장되었습니다.")
        plt.show()

# ==============================================================================
# 6. 연합학습 시뮬레이션 실행
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Flower Federated Learning Simulation")
    # [수정] 옵션을 '--s'로 변경
    parser.add_argument(
        "--s",
        action="store_true",
        help="개별 클라이언트 모델 평가를 건너뛰어 속도를 향상시킵니다."
    )
    args = parser.parse_args()
    
    print("=" * 80, "\n연합학습 시뮬레이션을 시작합니다.\n", "=" * 80)
    
    train_subsets, testloader, num_classes, partition_report_df = load_and_partition_data()
    
    print(" 최종 데이터 분할 현황 (Final Data Partition Report)")
    print("="*80)
    print(partition_report_df.to_string())
    print("="*80 + "\n")

    def client_fn(cid: str) -> fl.client.Client:
        model = get_model(num_classes)
        train_subset = train_subsets[int(cid)]
        return CifarClient(cid, train_subset, model).to_client()

    # [수정] args.s를 사용하여 전략에 옵션 전달
    strategy = CustomFedAvg(
        testloader=testloader,
        skip_individual_eval=args.s,
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
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
            save_results(partition_report_df)
        else:
            print("[알림] 학습이 완료되기 전에 종료되어 저장할 결과가 없습니다.")

if __name__ == "__main__":
    main()
