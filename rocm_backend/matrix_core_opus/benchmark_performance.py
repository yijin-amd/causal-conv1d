#!/usr/bin/env python3
"""
性能基准测试脚本（Python版本）
功能：运行 casual_conv1d_opus.exe 多次，收集和分析性能数据
作者：AI Assistant
日期：2025-11-15
"""

import subprocess
import time
import statistics
import sys
import os
import re
from datetime import datetime
from pathlib import Path

# 颜色输出
class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'

def print_header():
    print(f"{Colors.BLUE}╔════════════════════════════════════════════════════════════════╗{Colors.NC}")
    print(f"{Colors.BLUE}║          Causal Conv1D 性能基准测试 (Python)                  ║{Colors.NC}")
    print(f"{Colors.BLUE}╚════════════════════════════════════════════════════════════════╝{Colors.NC}")

def print_section(title):
    print(f"\n{Colors.GREEN}▶ {title}{Colors.NC}")
    print("=" * 64)

def print_info(msg):
    print(f"{Colors.YELLOW}ℹ {msg}{Colors.NC}")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.NC}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.NC}")

class PerformanceBenchmark:
    def __init__(self, executable="./casual_conv1d_opus.exe", iterations=100):
        self.executable = executable
        self.iterations = iterations
        self.output_dir = Path("benchmark_results")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            'total_times': [],      # 总运行时间
            'success_count': 0,
            'failed_count': 0,
            'valid_runs': []        # 验证通过的运行
        }
        
    def check_executable(self):
        """检查可执行文件是否存在"""
        if not Path(self.executable).exists():
            print_error(f"找不到可执行文件: {self.executable}")
            print("请先编译程序")
            return False
        print_success(f"找到可执行文件: {self.executable}")
        return True
    
    def setup_output_dir(self):
        """创建输出目录"""
        self.output_dir.mkdir(exist_ok=True)
        print_success(f"输出目录: {self.output_dir}")
    
    def detect_config(self):
        """检测当前配置"""
        print_section("检测当前配置")
        
        cpp_file = Path("casual_conv1d_opus.cpp")
        if cpp_file.exists():
            content = cpp_file.read_text()
            
            # 提取宏定义
            silu_match = re.search(r'#define\s+ENABLE_SILU_ACTIVATION\s+(\d+)', content)
            verify_match = re.search(r'#define\s+ENABLE_HOST_VERIFICATION\s+(\d+)', content)
            
            silu = silu_match.group(1) if silu_match else "未知"
            verify = verify_match.group(1) if verify_match else "未知"
            
            print(f"ENABLE_SILU_ACTIVATION:    {silu} ({'启用' if silu == '1' else '禁用'})")
            print(f"ENABLE_HOST_VERIFICATION:  {verify} ({'启用' if verify == '1' else '禁用'})")
        
        # 运行一次获取配置信息
        try:
            result = subprocess.run([self.executable], 
                                   capture_output=True, 
                                   text=True, 
                                   timeout=30)
            output = result.stdout + result.stderr
            
            # 提取 batch 信息
            batch_match = re.search(r'batch:(\d+)', output)
            if batch_match:
                print(f"Batch Size:                {batch_match.group(1)}")
        except Exception as e:
            print_info(f"无法获取运行配置: {e}")
        
        print()
    
    def run_single_test(self, iteration):
        """运行单次测试"""
        start_time = time.time()
        
        try:
            result = subprocess.run([self.executable], 
                                   capture_output=True, 
                                   text=True, 
                                   timeout=60)
            
            elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒
            output = result.stdout + result.stderr
            
            # 检查是否验证通过
            is_valid = 'valid' in output.lower()
            
            if is_valid:
                self.results['success_count'] += 1
                self.results['valid_runs'].append(iteration)
            else:
                self.results['failed_count'] += 1
            
            self.results['total_times'].append(elapsed_time)
            
            return {
                'iteration': iteration,
                'time': elapsed_time,
                'valid': is_valid,
                'output': output
            }
            
        except subprocess.TimeoutExpired:
            print_error(f"迭代 {iteration} 超时")
            self.results['failed_count'] += 1
            return None
        except Exception as e:
            print_error(f"迭代 {iteration} 出错: {e}")
            self.results['failed_count'] += 1
            return None
    
    def run_benchmark(self):
        """运行基准测试"""
        print_section(f"运行性能测试 ({self.iterations} 次迭代)")
        
        all_results = []
        
        for i in range(1, self.iterations + 1):
            # 显示进度
            if i % 10 == 0 or i == 1:
                progress = i / self.iterations
                bar_length = 50
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"\r进度: [{bar}] {i}/{self.iterations} ({progress*100:.1f}%)", end='', flush=True)
            
            # 运行测试
            result = self.run_single_test(i)
            if result:
                all_results.append(result)
        
        print()  # 换行
        print_success(f"完成 {self.iterations} 次迭代")
        
        return all_results
    
    def calculate_statistics(self):
        """计算统计信息"""
        print_section("性能统计")
        
        times = self.results['total_times']
        
        if not times:
            print_error("没有有效的性能数据")
            return
        
        # 计算统计量
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        min_time = min(times)
        max_time = max(times)
        
        if len(times) > 1:
            stdev_time = statistics.stdev(times)
            cv = (stdev_time / mean_time) * 100  # 变异系数
        else:
            stdev_time = 0
            cv = 0
        
        # 计算成功率
        total_runs = self.results['success_count'] + self.results['failed_count']
        success_rate = (self.results['success_count'] / total_runs * 100) if total_runs > 0 else 0
        
        # 打印统计信息
        print(f"样本数量:     {len(times)}")
        print(f"成功运行:     {self.results['success_count']}/{total_runs}")
        print(f"成功率:       {success_rate:.2f}%")
        print()
        print(f"平均时间:     {mean_time:.2f} ms")
        print(f"标准差:       {stdev_time:.2f} ms")
        print(f"最小时间:     {min_time:.2f} ms")
        print(f"最大时间:     {max_time:.2f} ms")
        print(f"中位数:       {median_time:.2f} ms")
        print(f"变异系数:     {cv:.2f}%")
        
        # 计算百分位数
        if len(times) >= 4:
            p25 = statistics.quantiles(times, n=4)[0]
            p75 = statistics.quantiles(times, n=4)[2]
            p95 = statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max_time
            print(f"P25:          {p25:.2f} ms")
            print(f"P75:          {p75:.2f} ms")
            print(f"P95:          {p95:.2f} ms")
        
        return {
            'mean': mean_time,
            'median': median_time,
            'stdev': stdev_time,
            'min': min_time,
            'max': max_time,
            'cv': cv,
            'success_rate': success_rate
        }
    
    def save_report(self, stats):
        """保存报告"""
        print_section("生成报告")
        
        report_file = self.output_dir / f"benchmark_{self.timestamp}.txt"
        csv_file = self.output_dir / f"benchmark_{self.timestamp}.csv"
        
        # 生成文本报告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 64 + "\n")
            f.write("  Causal Conv1D 性能基准测试报告\n")
            f.write("=" * 64 + "\n\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"迭代次数: {self.iterations}\n")
            f.write(f"可执行文件: {self.executable}\n\n")
            
            f.write("性能统计:\n")
            f.write(f"  平均时间:     {stats['mean']:.2f} ms\n")
            f.write(f"  标准差:       {stats['stdev']:.2f} ms\n")
            f.write(f"  最小时间:     {stats['min']:.2f} ms\n")
            f.write(f"  最大时间:     {stats['max']:.2f} ms\n")
            f.write(f"  中位数:       {stats['median']:.2f} ms\n")
            f.write(f"  变异系数:     {stats['cv']:.2f}%\n")
            f.write(f"  成功率:       {stats['success_rate']:.2f}%\n")
            f.write("\n" + "=" * 64 + "\n")
        
        # 生成 CSV 报告
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("Iteration,Time(ms),Valid\n")
            for i, time_val in enumerate(self.results['total_times'], 1):
                valid = "Yes" if i in self.results['valid_runs'] else "No"
                f.write(f"{i},{time_val:.2f},{valid}\n")
        
        print_success(f"文本报告: {report_file}")
        print_success(f"CSV 数据: {csv_file}")
    
    def run(self):
        """运行完整的基准测试流程"""
        print_header()
        
        # 检查环境
        if not self.check_executable():
            return False
        
        self.setup_output_dir()
        self.detect_config()
        
        # 运行测试
        start_time = time.time()
        self.run_benchmark()
        total_time = time.time() - start_time
        
        # 计算统计
        stats = self.calculate_statistics()
        
        # 显示总时间
        print()
        print(f"总测试时间: {total_time:.2f} 秒")
        
        # 保存报告
        if stats:
            self.save_report(stats)
        
        print()
        print_success("基准测试完成！")
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Causal Conv1D 性能基准测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                    # 运行 100 次迭代
  %(prog)s -n 50              # 运行 50 次迭代
  %(prog)s -e ./my_exe        # 指定可执行文件
        """
    )
    
    parser.add_argument('-n', '--iterations', type=int, default=100,
                        help='迭代次数 (默认: 100)')
    parser.add_argument('-e', '--executable', type=str, 
                        default='./casual_conv1d_opus.exe',
                        help='可执行文件路径 (默认: ./casual_conv1d_opus.exe)')
    
    args = parser.parse_args()
    
    # 运行基准测试
    benchmark = PerformanceBenchmark(
        executable=args.executable,
        iterations=args.iterations
    )
    
    success = benchmark.run()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

