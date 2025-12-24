#!/bin/bash
#===============================================================================
# Qwen-vLLM K8s集群状态监控脚本
# 
# 使用方法: bash cluster-status.sh
#===============================================================================

echo "=============================================="
echo "Qwen-vLLM K8s 集群状态"
echo "时间: $(date)"
echo "=============================================="

echo ""
echo ">>> 节点状态"
kubectl get nodes -o wide

echo ""
echo ">>> Pod状态"
kubectl get pods -o wide

echo ""
echo ">>> GPU资源"
kubectl describe nodes | grep -A 5 "Allocated resources" | head -20

echo ""
echo ">>> 服务状态"
kubectl get svc

echo ""
echo ">>> 各节点健康检查"
for pod in $(kubectl get pods -o jsonpath='{.items[*].status.podIP}'); do
    echo -n "Pod $pod: "
    curl -s --max-time 3 "http://${pod}:8000/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"GPU: {d['gpu']['allocated_gb']:.1f}GB, 状态: {d['status']}\")" 2>/dev/null || echo "❌ 无响应"
done

echo ""
echo "=============================================="
