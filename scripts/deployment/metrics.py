"""
vLLM服务核心Metrics监控
包括：请求队列、TTFT、Decoding Throughput等SLO指标
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List
from threading import Lock
import statistics

@dataclass
class RequestMetrics:
    """单个请求的metrics"""
    request_id: str
    start_time: float
    first_token_time: float = 0.0
    end_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    
    @property
    def time_to_first_token(self) -> float:
        """TTFT: 从请求到第一个token的时间(秒)"""
        if self.first_token_time > 0:
            return self.first_token_time - self.start_time
        return 0.0
    
    @property
    def total_time(self) -> float:
        """总处理时间(秒)"""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def decoding_throughput(self) -> float:
        """Decoding吞吐量: tokens/秒"""
        if self.end_time > 0 and self.first_token_time > 0:
            decode_time = self.end_time - self.first_token_time
            if decode_time > 0 and self.output_tokens > 1:
                # 减去第一个token，因为TTFT已经计算了
                return (self.output_tokens - 1) / decode_time
        return 0.0


class MetricsCollector:
    """Metrics收集器 - 增强版可观测性"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.lock = Lock()
        
        # 服务启动时间
        self.start_time = time.time()
        
        # 当前状态
        self.num_waiting_requests = 0
        self.num_running_requests = 0
        self.peak_concurrent_requests = 0  # 峰值并发
        
        # 历史记录
        self.completed_requests: List[RequestMetrics] = []
        self.active_requests: Dict[str, RequestMetrics] = {}
        
        # 统计数据
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_input_tokens = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # 延迟直方图 (毫秒区间)
        self.latency_buckets = {
            "0-100ms": 0,
            "100-500ms": 0,
            "500-1000ms": 0,
            "1-2s": 0,
            "2-5s": 0,
            "5-10s": 0,
            ">10s": 0
        }
        
        # 错误统计
        self.error_counts: Dict[str, int] = {}
        
        # 每分钟请求数 (最近10分钟)
        self.requests_per_minute: List[Dict] = []
        
    def start_request(self, request_id: str, input_tokens: int = 0) -> RequestMetrics:
        """开始一个新请求"""
        with self.lock:
            metrics = RequestMetrics(
                request_id=request_id,
                start_time=time.time(),
                input_tokens=input_tokens
            )
            self.active_requests[request_id] = metrics
            self.num_running_requests += 1
            self.total_requests += 1
            self.total_input_tokens += input_tokens
            
            # 更新峰值并发
            if self.num_running_requests > self.peak_concurrent_requests:
                self.peak_concurrent_requests = self.num_running_requests
            
            return metrics
    
    def mark_first_token(self, request_id: str):
        """标记第一个token生成"""
        with self.lock:
            if request_id in self.active_requests:
                self.active_requests[request_id].first_token_time = time.time()
    
    def end_request(self, request_id: str, output_tokens: int = 0, success: bool = True, error: str = None):
        """结束请求"""
        with self.lock:
            if request_id in self.active_requests:
                metrics = self.active_requests.pop(request_id)
                metrics.end_time = time.time()
                metrics.output_tokens = output_tokens
                
                # 更新延迟直方图
                latency_ms = metrics.total_time * 1000
                if latency_ms < 100:
                    self.latency_buckets["0-100ms"] += 1
                elif latency_ms < 500:
                    self.latency_buckets["100-500ms"] += 1
                elif latency_ms < 1000:
                    self.latency_buckets["500-1000ms"] += 1
                elif latency_ms < 2000:
                    self.latency_buckets["1-2s"] += 1
                elif latency_ms < 5000:
                    self.latency_buckets["2-5s"] += 1
                elif latency_ms < 10000:
                    self.latency_buckets["5-10s"] += 1
                else:
                    self.latency_buckets[">10s"] += 1
                
                # 更新成功/失败计数
                if success:
                    self.successful_requests += 1
                else:
                    self.failed_requests += 1
                    if error:
                        error_type = error.split(":")[0] if ":" in error else error[:50]
                        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                
                # 保存到历史记录
                self.completed_requests.append(metrics)
                if len(self.completed_requests) > self.window_size:
                    self.completed_requests.pop(0)
                
                self.num_running_requests -= 1
                self.total_tokens_generated += output_tokens
    
    def add_waiting_request(self):
        """增加等待请求数"""
        with self.lock:
            self.num_waiting_requests += 1
    
    def remove_waiting_request(self):
        """减少等待请求数"""
        with self.lock:
            if self.num_waiting_requests > 0:
                self.num_waiting_requests -= 1
    
    def get_current_metrics(self) -> Dict:
        """获取当前metrics"""
        with self.lock:
            uptime = time.time() - self.start_time
            error_rate = (self.failed_requests / self.total_requests * 100) if self.total_requests > 0 else 0.0
            success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 100.0
            avg_tokens_per_request = self.total_tokens_generated / self.successful_requests if self.successful_requests > 0 else 0
            
            return {
                "num_waiting_requests": self.num_waiting_requests,
                "num_running_requests": self.num_running_requests,
                "peak_concurrent_requests": self.peak_concurrent_requests,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate_percent": round(success_rate, 2),
                "error_rate_percent": round(error_rate, 2),
                "total_tokens_generated": self.total_tokens_generated,
                "total_input_tokens": self.total_input_tokens,
                "avg_tokens_per_request": round(avg_tokens_per_request, 1),
                "uptime_seconds": round(uptime, 1),
                "requests_per_second": round(self.total_requests / uptime, 3) if uptime > 0 else 0,
            }
    
    def get_slo_metrics(self) -> Dict:
        """获取SLO相关metrics"""
        with self.lock:
            if not self.completed_requests:
                return {
                    "ttft_mean": 0.0,
                    "ttft_p50": 0.0,
                    "ttft_p95": 0.0,
                    "ttft_p99": 0.0,
                    "decoding_throughput_mean": 0.0,
                    "decoding_throughput_p50": 0.0,
                    "decoding_throughput_p95": 0.0,
                    "total_throughput": 0.0,
                }
            
            # 计算TTFT统计
            ttfts = [r.time_to_first_token for r in self.completed_requests if r.time_to_first_token > 0]
            
            # 计算Decoding Throughput统计
            throughputs = [r.decoding_throughput for r in self.completed_requests if r.decoding_throughput > 0]
            
            # 计算总吞吐量
            total_time = sum(r.total_time for r in self.completed_requests)
            total_tokens = sum(r.output_tokens for r in self.completed_requests)
            total_throughput = total_tokens / total_time if total_time > 0 else 0.0
            
            return {
                "ttft_mean": statistics.mean(ttfts) if ttfts else 0.0,
                "ttft_p50": statistics.median(ttfts) if ttfts else 0.0,
                "ttft_p95": self._percentile(ttfts, 95) if ttfts else 0.0,
                "ttft_p99": self._percentile(ttfts, 99) if ttfts else 0.0,
                "decoding_throughput_mean": statistics.mean(throughputs) if throughputs else 0.0,
                "decoding_throughput_p50": statistics.median(throughputs) if throughputs else 0.0,
                "decoding_throughput_p95": self._percentile(throughputs, 95) if throughputs else 0.0,
                "total_throughput": total_throughput,
                "sample_size": len(self.completed_requests),
            }
    
    def get_latency_histogram(self) -> Dict:
        """获取延迟直方图"""
        with self.lock:
            return dict(self.latency_buckets)
    
    def get_error_details(self) -> Dict:
        """获取错误详情"""
        with self.lock:
            return dict(self.error_counts)
    
    def get_request_latencies(self) -> Dict:
        """获取请求延迟统计"""
        with self.lock:
            if not self.completed_requests:
                return {
                    "latency_mean": 0.0,
                    "latency_p50": 0.0,
                    "latency_p95": 0.0,
                    "latency_p99": 0.0,
                    "latency_min": 0.0,
                    "latency_max": 0.0,
                }
            
            latencies = [r.total_time for r in self.completed_requests if r.total_time > 0]
            if not latencies:
                return {
                    "latency_mean": 0.0,
                    "latency_p50": 0.0,
                    "latency_p95": 0.0,
                    "latency_p99": 0.0,
                    "latency_min": 0.0,
                    "latency_max": 0.0,
                }
            
            return {
                "latency_mean": round(statistics.mean(latencies), 4),
                "latency_p50": round(statistics.median(latencies), 4),
                "latency_p95": round(self._percentile(latencies, 95), 4),
                "latency_p99": round(self._percentile(latencies, 99), 4),
                "latency_min": round(min(latencies), 4),
                "latency_max": round(max(latencies), 4),
            }
    
    def get_all_metrics(self) -> Dict:
        """获取所有metrics"""
        current = self.get_current_metrics()
        slo = self.get_slo_metrics()
        return {
            **current,
            "slo": slo
        }
    
    def get_extended_metrics(self) -> Dict:
        """获取扩展的可观测性metrics"""
        current = self.get_current_metrics()
        slo = self.get_slo_metrics()
        latency = self.get_request_latencies()
        histogram = self.get_latency_histogram()
        errors = self.get_error_details()
        
        return {
            "current": current,
            "slo": slo,
            "latency": latency,
            "latency_histogram": histogram,
            "errors": errors,
        }
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


# 全局metrics收集器
metrics_collector = MetricsCollector()
