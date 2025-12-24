#!/usr/bin/env python3
"""
vLLM服务实时Metrics监控仪表板
支持终端显示和告警
"""

import requests
import time
import sys
import os
from datetime import datetime

# 配置
SERVICE_URL = os.getenv("VLLM_URL", "http://localhost:8000")
REFRESH_INTERVAL = 2  # 秒
ALERT_THRESHOLDS = {
    "num_waiting_requests": 5,    # 等待队列 > 5 告警
    "ttft_p95": 3.0,              # TTFT P95 > 3秒 告警
    "decoding_throughput_min": 10, # 吞吐量 < 10 tokens/s 告警
}

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def get_metrics():
    """获取metrics"""
    try:
        r = requests.get(f"{SERVICE_URL}/metrics", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_health():
    """获取健康状态"""
    try:
        r = requests.get(f"{SERVICE_URL}/health", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def format_value(value, unit="", precision=2):
    """格式化数值"""
    if isinstance(value, float):
        return f"{value:.{precision}f}{unit}"
    return f"{value}{unit}"

def check_alerts(metrics):
    """检查告警"""
    alerts = []
    
    if metrics.get("num_waiting_requests", 0) > ALERT_THRESHOLDS["num_waiting_requests"]:
        alerts.append(f"⚠️  队列等待过长: {metrics['num_waiting_requests']} 个请求")
    
    slo = metrics.get("slo", {})
    if slo.get("ttft_p95", 0) > ALERT_THRESHOLDS["ttft_p95"]:
        alerts.append(f"⚠️  TTFT P95 过高: {slo['ttft_p95']:.2f}s")
    
    if slo.get("decoding_throughput_mean", 0) > 0 and slo.get("decoding_throughput_mean", 999) < ALERT_THRESHOLDS["decoding_throughput_min"]:
        alerts.append(f"⚠️  吞吐量过低: {slo['decoding_throughput_mean']:.1f} tokens/s")
    
    return alerts

def render_dashboard(metrics, health):
    """渲染仪表板"""
    clear_screen()
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║           vLLM 服务实时监控仪表板                              ║")
    print(f"║  {now}                                    ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    
    if "error" in metrics:
        print(f"║  ❌ 服务异常: {metrics['error'][:45]:<45} ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        return
    
    # 服务状态
    status = health.get("status", "unknown")
    status_icon = "🟢" if status == "ok" else "🔴"
    print(f"║  状态: {status_icon} {status.upper():<55} ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    
    # 请求队列
    waiting = metrics.get("num_waiting_requests", 0)
    running = metrics.get("num_running_requests", 0)
    total = metrics.get("total_requests", 0)
    
    print("║  📊 请求状态                                                   ║")
    print(f"║     等待中: {waiting:<10} 运行中: {running:<10} 总请求: {total:<10} ║")
    
    # 进度条
    max_concurrent = 3
    bar_len = 20
    running_bar = int(running / max_concurrent * bar_len) if max_concurrent > 0 else 0
    bar = "█" * running_bar + "░" * (bar_len - running_bar)
    print(f"║     并发使用: [{bar}] {running}/{max_concurrent:<18} ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    
    # SLO指标
    slo = metrics.get("slo", {})
    print("║  ⏱️  SLO 指标                                                   ║")
    
    ttft_mean = slo.get("ttft_mean", 0)
    ttft_p50 = slo.get("ttft_p50", 0)
    ttft_p95 = slo.get("ttft_p95", 0)
    ttft_p99 = slo.get("ttft_p99", 0)
    
    print(f"║     TTFT (Time to First Token):                                ║")
    print(f"║       Mean: {ttft_mean:>6.2f}s  P50: {ttft_p50:>6.2f}s  P95: {ttft_p95:>6.2f}s  P99: {ttft_p99:>6.2f}s ║")
    
    throughput_mean = slo.get("decoding_throughput_mean", 0)
    throughput_p50 = slo.get("decoding_throughput_p50", 0)
    total_throughput = slo.get("total_throughput", 0)
    
    print(f"║     Decoding Throughput:                                       ║")
    print(f"║       Mean: {throughput_mean:>6.1f} tok/s  P50: {throughput_p50:>6.1f} tok/s  Total: {total_throughput:>6.1f} tok/s ║")
    
    print("╠════════════════════════════════════════════════════════════════╣")
    
    # GPU状态
    gpu = health.get("gpu", {})
    allocated = gpu.get("allocated_gb", 0)
    free = gpu.get("free_gb", 0)
    total_gpu = allocated + free
    
    print("║  🖥️  GPU 状态                                                   ║")
    gpu_bar_len = 30
    gpu_used_bar = int(allocated / total_gpu * gpu_bar_len) if total_gpu > 0 else 0
    gpu_bar = "█" * gpu_used_bar + "░" * (gpu_bar_len - gpu_used_bar)
    print(f"║     显存: [{gpu_bar}] {allocated:.1f}/{total_gpu:.1f}GB  ║")
    
    print("╠════════════════════════════════════════════════════════════════╣")
    
    # 告警
    alerts = check_alerts(metrics)
    if alerts:
        print("║  🚨 告警                                                       ║")
        for alert in alerts:
            print(f"║     {alert:<58} ║")
    else:
        print("║  ✅ 无告警                                                     ║")
    
    print("╠════════════════════════════════════════════════════════════════╣")
    print("║  按 Ctrl+C 退出                                                ║")
    print("╚════════════════════════════════════════════════════════════════╝")

def main():
    print(f"连接到服务: {SERVICE_URL}")
    print(f"刷新间隔: {REFRESH_INTERVAL}秒")
    print("启动监控...")
    time.sleep(1)
    
    try:
        while True:
            metrics = get_metrics()
            health = get_health()
            render_dashboard(metrics, health)
            time.sleep(REFRESH_INTERVAL)
    except KeyboardInterrupt:
        print("\n监控已停止")
        sys.exit(0)

if __name__ == "__main__":
    main()
