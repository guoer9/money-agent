#!/bin/bash
#===============================================================================
# Qwen-vLLM 部署扩缩容脚本
# 
# 使用方法: 
#   bash scale-deployment.sh 3    # 扩展到3个副本
#   bash scale-deployment.sh      # 查看当前状态
#===============================================================================

DEPLOYMENT_NAME="qwen-vllm"
REPLICAS=${1:-""}

if [ -z "$REPLICAS" ]; then
    echo "当前部署状态:"
    kubectl get deployment ${DEPLOYMENT_NAME}
    echo ""
    echo "Pod分布:"
    kubectl get pods -o wide -l app=${DEPLOYMENT_NAME}
    echo ""
    echo "使用方法: $0 <副本数>"
    echo "例如: $0 3"
else
    echo "扩展 ${DEPLOYMENT_NAME} 到 ${REPLICAS} 个副本..."
    kubectl scale deployment ${DEPLOYMENT_NAME} --replicas=${REPLICAS}
    
    echo ""
    echo "等待Pod就绪..."
    kubectl rollout status deployment/${DEPLOYMENT_NAME} --timeout=300s
    
    echo ""
    echo "当前Pod分布:"
    kubectl get pods -o wide -l app=${DEPLOYMENT_NAME}
fi
