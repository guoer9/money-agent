#!/bin/bash
#===============================================================================
# Qwen-vLLM K8s Worker节点一键部署脚本
# 
# 使用方法:
#   1. 将此脚本复制到新机器
#   2. 修改下方配置
#   3. 运行: sudo bash join-cluster.sh
#
# 项目: https://github.com/guoer9/money-agent/tree/vllm
# 模型: https://huggingface.co/guoer9/qwen3-8b-news-classifier
#===============================================================================

set -e

#===============================================================================
# 配置区 - 根据你的环境修改
#===============================================================================

# Master节点信息 (从主节点获取: kubeadm token create --print-join-command)
MASTER_IP="10.9.3.131"
MASTER_PORT="6443"
JOIN_TOKEN="cu6k86.1e8x9qh457it3l6h"
CA_CERT_HASH="sha256:7a405916e64356736866bb4cbd5ccb2db719e7099b5a57514d00fd16abac4454"

# K8s版本
K8S_VERSION="1.28"

# 镜像源 (国内加速)
K8S_IMAGE_REPO="6e4mx6zwaaozht-k8s.xuanyuan.run"

# 代理设置 (如果需要)
USE_PROXY=false
PROXY_URL="http://127.0.0.1:7890"

#===============================================================================
# 脚本开始
#===============================================================================

echo "=============================================="
echo "Qwen-vLLM K8s Worker节点部署"
echo "=============================================="
echo "Master: ${MASTER_IP}:${MASTER_PORT}"
echo "K8s版本: v${K8S_VERSION}"
echo "=============================================="

# 检查root权限
if [ "$EUID" -ne 0 ]; then
    echo "❌ 请使用sudo运行此脚本"
    exit 1
fi

# 设置代理
if [ "$USE_PROXY" = true ]; then
    echo "📡 使用代理: $PROXY_URL"
    export http_proxy="$PROXY_URL"
    export https_proxy="$PROXY_URL"
    export no_proxy="localhost,127.0.0.1,${MASTER_IP},10.244.0.0/16"
fi

#----------------------------------------------
# 1. 系统准备
#----------------------------------------------
echo ""
echo ">>> 步骤1: 系统准备"

# 关闭swap
echo "关闭swap..."
swapoff -a
sed -i '/swap/d' /etc/fstab

# 加载内核模块
echo "加载内核模块..."
cat > /etc/modules-load.d/k8s.conf << EOF
overlay
br_netfilter
EOF
modprobe overlay
modprobe br_netfilter

# 设置内核参数
cat > /etc/sysctl.d/k8s.conf << EOF
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
EOF
sysctl --system > /dev/null 2>&1

echo "✅ 系统准备完成"

#----------------------------------------------
# 2. 安装containerd
#----------------------------------------------
echo ""
echo ">>> 步骤2: 安装containerd"

if command -v containerd &> /dev/null; then
    echo "containerd已安装，跳过"
else
    apt-get update
    apt-get install -y containerd
fi

# 配置containerd
mkdir -p /etc/containerd
containerd config default > /etc/containerd/config.toml
sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml

# 配置镜像加速
sed -i "s|registry.k8s.io|${K8S_IMAGE_REPO}|g" /etc/containerd/config.toml

systemctl restart containerd
systemctl enable containerd

echo "✅ containerd配置完成"

#----------------------------------------------
# 3. 安装NVIDIA Container Toolkit (如果有GPU)
#----------------------------------------------
echo ""
echo ">>> 步骤3: 检查GPU"

if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU，安装Container Toolkit..."
    
    # 添加NVIDIA仓库
    if [ ! -f /etc/apt/sources.list.d/nvidia-container-toolkit.list ]; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        apt-get update
    fi
    
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=containerd
    systemctl restart containerd
    
    echo "✅ NVIDIA Container Toolkit安装完成"
else
    echo "⚠️ 未检测到NVIDIA GPU，跳过"
fi

#----------------------------------------------
# 4. 安装kubeadm/kubelet
#----------------------------------------------
echo ""
echo ">>> 步骤4: 安装Kubernetes组件"

if command -v kubeadm &> /dev/null; then
    echo "kubeadm已安装，跳过"
else
    apt-get install -y apt-transport-https ca-certificates curl gpg
    
    # 添加K8s仓库
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://pkgs.k8s.io/core:/stable:/v${K8S_VERSION}/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
    echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v${K8S_VERSION}/deb/ /" | tee /etc/apt/sources.list.d/kubernetes.list
    
    apt-get update
    apt-get install -y kubelet kubeadm
    apt-mark hold kubelet kubeadm
fi

systemctl enable kubelet

echo "✅ Kubernetes组件安装完成"

#----------------------------------------------
# 5. 加入集群
#----------------------------------------------
echo ""
echo ">>> 步骤5: 加入K8s集群"

# 检查是否已加入集群
if [ -f /etc/kubernetes/kubelet.conf ]; then
    echo "⚠️ 已经加入集群，如需重新加入请先执行: kubeadm reset -f"
    exit 0
fi

echo "加入集群: ${MASTER_IP}:${MASTER_PORT}"
kubeadm join ${MASTER_IP}:${MASTER_PORT} \
    --token ${JOIN_TOKEN} \
    --discovery-token-ca-cert-hash ${CA_CERT_HASH}

echo "✅ 成功加入集群!"

#----------------------------------------------
# 6. 验证
#----------------------------------------------
echo ""
echo "=============================================="
echo "🎉 部署完成!"
echo "=============================================="
echo ""
echo "在主节点(${MASTER_IP})上执行以下命令验证:"
echo ""
echo "  kubectl get nodes"
echo "  kubectl get pods -o wide"
echo ""
echo "扩展部署到此节点:"
echo ""
echo "  kubectl scale deployment qwen-vllm --replicas=2"
echo ""
echo "=============================================="
