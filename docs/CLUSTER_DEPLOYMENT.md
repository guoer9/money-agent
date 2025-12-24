# Qwen-vLLM K8s 多节点集群部署指南

## 架构概览

```
                    ┌─────────────────────────────────────────┐
                    │           K8s Master (10.9.3.131)       │
                    │  ┌─────────────────────────────────┐    │
                    │  │ qwen-vllm Pod (GPU: RTX 3080)   │    │
                    │  │ - 8-bit量化模型                  │    │
                    │  │ - 端口: 8000                     │    │
                    │  └─────────────────────────────────┘    │
                    └─────────────────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
    ┌─────────▼─────────┐    ┌─────────▼─────────┐    ┌─────────▼─────────┐
    │  Worker Node 1     │    │  Worker Node 2     │    │  Worker Node N     │
    │  (可选GPU)         │    │  (可选GPU)         │    │  (可选GPU)         │
    │  qwen-vllm Pod    │    │  qwen-vllm Pod    │    │  qwen-vllm Pod    │
    └───────────────────┘    └───────────────────┘    └───────────────────┘
```

---

## 前置要求

### 每个节点需要

| 组件 | 要求 |
|------|------|
| 操作系统 | Ubuntu 20.04/22.04 |
| GPU | NVIDIA (RTX 3080/4090等) + 驱动 |
| 内存 | ≥16GB |
| 磁盘 | ≥50GB |
| 网络 | 节点间互通 |

---

## 快速开始

### 步骤1: 在主节点获取加入命令

```bash
# 在Master节点 (10.9.3.131) 执行
kubeadm token create --print-join-command
```

输出示例:
```
kubeadm join 10.9.3.131:6443 --token xxx --discovery-token-ca-cert-hash sha256:xxx
```

### 步骤2: 在新节点执行部署脚本

```bash
# 1. 下载脚本
curl -O https://raw.githubusercontent.com/guoer9/money-agent/vllm/scripts/k8s/join-cluster.sh

# 2. 修改配置 (编辑脚本中的MASTER_IP, JOIN_TOKEN, CA_CERT_HASH)
vim join-cluster.sh

# 3. 执行
sudo bash join-cluster.sh
```

### 步骤3: 验证节点加入

```bash
# 在Master节点执行
kubectl get nodes

# 输出:
# NAME      STATUS   ROLES           AGE   VERSION
# master    Ready    control-plane   2h    v1.28.15
# worker1   Ready    <none>          5m    v1.28.15
```

### 步骤4: 扩展部署

```bash
# 扩展到多个副本
kubectl scale deployment qwen-vllm --replicas=2

# 查看Pod分布
kubectl get pods -o wide
```

---

## 管理脚本

### 查看集群状态

```bash
bash scripts/k8s/cluster-status.sh
```

### 扩缩容

```bash
# 扩展到3个副本
bash scripts/k8s/scale-deployment.sh 3

# 缩减到1个副本
bash scripts/k8s/scale-deployment.sh 1
```

---

## 常见问题

### Q: Token过期怎么办?

```bash
# 在Master节点重新生成
kubeadm token create --print-join-command
```

### Q: 节点NotReady?

```bash
# 检查kubelet状态
sudo systemctl status kubelet

# 检查网络插件
kubectl get pods -n kube-flannel
```

### Q: GPU不可用?

```bash
# 检查NVIDIA Device Plugin
kubectl get pods -n kube-system | grep nvidia

# 检查节点GPU资源
kubectl describe node <node-name> | grep nvidia
```

### Q: 如何移除节点?

```bash
# 在Master节点
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
kubectl delete node <node-name>

# 在Worker节点
sudo kubeadm reset -f
```

---

## 监控

### Prometheus集成

所有Pod暴露 `/metrics/prometheus` 端点，可统一采集：

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vllm-cluster'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: qwen-vllm
        action: keep
    metrics_path: '/metrics/prometheus'
```

---

## 相关链接

- **GitHub**: https://github.com/guoer9/money-agent/tree/vllm
- **模型**: https://huggingface.co/guoer9/qwen3-8b-news-classifier
- **API文档**: [docs/API.md](API.md)
- **Metrics文档**: [docs/METRICS.md](METRICS.md)
