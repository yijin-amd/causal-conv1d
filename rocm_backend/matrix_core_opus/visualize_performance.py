#!/usr/bin/env python3
"""
可视化 causal_conv1d_opus 性能数据
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_and_visualize(trace_dir):
    """加载并可视化性能数据"""
    
    # 查找 CSV 文件
    kernel_stats_file = None
    hip_api_stats_file = None
    kernel_trace_file = None
    
    for file in os.listdir(trace_dir):
        if file.endswith('_kernel_stats.csv'):
            kernel_stats_file = os.path.join(trace_dir, file)
        elif file.endswith('_hip_api_stats.csv'):
            hip_api_stats_file = os.path.join(trace_dir, file)
        elif file.endswith('_kernel_trace.csv'):
            kernel_trace_file = os.path.join(trace_dir, file)
    
    if not kernel_stats_file or not hip_api_stats_file:
        print(f"Error: 找不到必要的 CSV 文件在 {trace_dir}")
        return
    
    # 读取数据
    kernel_stats = pd.read_csv(kernel_stats_file)
    hip_api_stats = pd.read_csv(hip_api_stats_file)
    
    # 创建图表
    fig = plt.figure(figsize=(20, 12))
    
    # ========== 1. Kernel 执行时间对比 ==========
    ax1 = plt.subplot(2, 3, 1)
    kernel_names = []
    kernel_times = []
    
    for idx, row in kernel_stats.iterrows():
        name = row['Name']
        # 简化 kernel 名称
        if 'matrix_core_kernel_block_v2' in name:
            short_name = 'GEMM Kernel'
        elif 'preprocess_input_kernel' in name:
            short_name = 'Input Preprocess'
        elif 'preprocess_weight_kernel' in name:
            short_name = 'Weight Preprocess'
        elif 'copyBuffer' in name:
            short_name = 'Copy Buffer'
        else:
            short_name = name[:30]
        
        kernel_names.append(short_name)
        kernel_times.append(row['TotalDurationNs'] / 1000)  # 转换为微秒
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax1.barh(kernel_names, kernel_times, color=colors[:len(kernel_names)])
    ax1.set_xlabel('Execution Time (us)', fontsize=12, fontweight='bold')
    ax1.set_title('GPU Kernel Execution Time', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, time) in enumerate(zip(bars, kernel_times)):
        ax1.text(time, bar.get_y() + bar.get_height()/2, 
                f'{time:.1f} us', 
                va='center', ha='left', fontsize=10, fontweight='bold')
    
    # ========== 2. Kernel 时间占比饼图 ==========
    ax2 = plt.subplot(2, 3, 2)
    percentages = [row['Percentage'] for _, row in kernel_stats.iterrows()]
    
    wedges, texts, autotexts = ax2.pie(percentages, labels=kernel_names, autopct='%1.1f%%',
                                         colors=colors[:len(kernel_names)],
                                         startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Kernel Time Distribution', fontsize=14, fontweight='bold')
    
    # ========== 3. HIP API 调用时间 ==========
    ax3 = plt.subplot(2, 3, 3)
    api_names = []
    api_times = []
    
    for idx, row in hip_api_stats.iterrows():
        api_names.append(row['Name'])
        api_times.append(row['TotalDurationNs'] / 1000)  # 转换为微秒
    
    bars = ax3.barh(api_names, api_times, color='#95E1D3')
    ax3.set_xlabel('Total Time (us)', fontsize=12, fontweight='bold')
    ax3.set_title('HIP API Call Time', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(axis='x', alpha=0.3)
    
    # ========== 4. Kernel 调用次数 ==========
    ax4 = plt.subplot(2, 3, 4)
    kernel_calls = [row['Calls'] for _, row in kernel_stats.iterrows()]
    bars = ax4.bar(range(len(kernel_names)), kernel_calls, color=colors[:len(kernel_names)])
    ax4.set_xticks(range(len(kernel_names)))
    ax4.set_xticklabels(kernel_names, rotation=45, ha='right')
    ax4.set_ylabel('Number of Calls', fontsize=12, fontweight='bold')
    ax4.set_title('Kernel Call Count', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, calls in zip(bars, kernel_calls):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(calls)}',
                ha='center', va='bottom', fontweight='bold')
    
    # ========== 5. 性能统计表格 ==========
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    # 计算总时间
    total_kernel_time = kernel_stats['TotalDurationNs'].sum()
    total_api_time = hip_api_stats['TotalDurationNs'].sum()
    
    # 创建统计表格
    stats_data = []
    stats_data.append(['Total Kernel Time', f'{total_kernel_time/1000:.1f} us'])
    stats_data.append(['Total HIP API Time', f'{total_api_time/1000:.1f} us'])
    stats_data.append(['Total Execution Time', f'{(total_kernel_time + total_api_time)/1000:.1f} us'])
    stats_data.append(['', ''])
    stats_data.append(['Kernel Breakdown:', ''])
    
    for idx, row in kernel_stats.iterrows():
        name = kernel_names[idx]
        time_us = row['TotalDurationNs'] / 1000
        stats_data.append([f'  {name}', f'{time_us:.1f} us ({row["Percentage"]:.1f}%)'])
    
    table = ax5.table(cellText=stats_data, cellLoc='left',
                     colWidths=[0.6, 0.4],
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(stats_data)):
        if i == 4:  # "Kernel Breakdown:" 行
            table[(i, 0)].set_facecolor('#E8E8E8')
            table[(i, 0)].set_text_props(weight='bold')
        if i < 3:
            table[(i, 0)].set_facecolor('#D5E8D4')
            table[(i, 1)].set_facecolor('#D5E8D4')
            table[(i, 0)].set_text_props(weight='bold')
    
    ax5.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    # ========== 6. API 调用次数与平均时间 ==========
    ax6 = plt.subplot(2, 3, 6)
    api_calls = [row['Calls'] for _, row in hip_api_stats.iterrows()]
    api_avg_times = [row['AverageNs'] / 1000 for _, row in hip_api_stats.iterrows()]
    
    x = np.arange(len(api_names))
    width = 0.35
    
    ax6_twin = ax6.twinx()
    bars1 = ax6.bar(x - width/2, api_calls, width, label='Calls', color='#74B9FF')
    bars2 = ax6_twin.bar(x + width/2, api_avg_times, width, label='Avg Time (us)', color='#FDCB6E')
    
    ax6.set_xlabel('HIP API', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Number of Calls', fontsize=11, fontweight='bold', color='#74B9FF')
    ax6_twin.set_ylabel('Average Time (us)', fontsize=11, fontweight='bold', color='#FDCB6E')
    ax6.set_xticks(x)
    ax6.set_xticklabels(api_names, rotation=45, ha='right', fontsize=9)
    ax6.set_title('HIP API Calls & Average Time', fontsize=14, fontweight='bold')
    ax6.tick_params(axis='y', labelcolor='#74B9FF')
    ax6_twin.tick_params(axis='y', labelcolor='#FDCB6E')
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')
    ax6.grid(axis='y', alpha=0.3)
    ax6_twin.set_yscale('log')
    
    plt.suptitle('Causal Conv1D Performance Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图表
    output_file = os.path.join(trace_dir, 'performance_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ 性能可视化图表已保存到: {output_file}")
    
    # 显示图表
    plt.show()
    
    # ========== 打印详细统计 ==========
    print("\n" + "="*80)
    print("性能分析报告".center(80))
    print("="*80)
    
    print(f"\n📊 总体统计:")
    print(f"  - GPU Kernel 总执行时间: {total_kernel_time/1000:.2f} us ({total_kernel_time/1000000:.4f} ms)")
    print(f"  - HIP API 总时间: {total_api_time/1000:.2f} us ({total_api_time/1000000:.4f} ms)")
    print(f"  - 总执行时间: {(total_kernel_time + total_api_time)/1000:.2f} us")
    
    print(f"\n🚀 GPU Kernel 性能:")
    for idx, row in kernel_stats.iterrows():
        name = kernel_names[idx]
        time_us = row['TotalDurationNs'] / 1000
        time_ms = time_us / 1000
        percentage = row['Percentage']
        calls = row['Calls']
        print(f"  [{percentage:5.1f}%] {name:25s}: {time_us:8.2f} us ({time_ms:.4f} ms) x {calls} calls")
    
    print(f"\n🔧 HIP API 调用:")
    for idx, row in hip_api_stats.iterrows():
        name = row['Name']
        time_us = row['TotalDurationNs'] / 1000
        avg_us = row['AverageNs'] / 1000
        calls = row['Calls']
        percentage = row['Percentage']
        print(f"  [{percentage:5.2f}%] {name:25s}: {time_us:10.1f} us (avg: {avg_us:.1f} us) x {calls} calls")
    
    print("\n" + "="*80)
    
    # 性能瓶颈分析
    print("\n💡 性能分析:")
    
    # 找出最耗时的 kernel
    max_kernel_idx = kernel_stats['TotalDurationNs'].idxmax()
    max_kernel_name = kernel_names[max_kernel_idx]
    max_kernel_time = kernel_stats.loc[max_kernel_idx, 'TotalDurationNs'] / 1000
    max_kernel_pct = kernel_stats.loc[max_kernel_idx, 'Percentage']
    
    print(f"  1. 最耗时的 Kernel: {max_kernel_name}")
    print(f"     - 时间: {max_kernel_time:.2f} us")
    print(f"     - 占比: {max_kernel_pct:.1f}% of total kernel time")
    
    # 分析预处理时间
    preprocess_time = 0
    for idx, row in kernel_stats.iterrows():
        if 'preprocess' in row['Name'].lower():
            preprocess_time += row['TotalDurationNs']
    
    if preprocess_time > 0:
        preprocess_pct = (preprocess_time / total_kernel_time) * 100
        print(f"\n  2. 预处理 Kernel 总时间: {preprocess_time/1000:.2f} us")
        print(f"     - 占总 Kernel 时间: {preprocess_pct:.1f}%")
    
    # 分析内存传输
    memcpy_time = 0
    for idx, row in hip_api_stats.iterrows():
        if 'hipMemcpy' in row['Name']:
            memcpy_time = row['TotalDurationNs']
            memcpy_calls = row['Calls']
            memcpy_avg = row['AverageNs'] / 1000
            break
    
    if memcpy_time > 0:
        print(f"\n  3. 内存传输 (hipMemcpy):")
        print(f"     - 总时间: {memcpy_time/1000:.2f} us")
        print(f"     - 调用次数: {memcpy_calls}")
        print(f"     - 平均时间: {memcpy_avg:.2f} us")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    trace_dir = 'trace_casual_conv1d'
    
    if not os.path.exists(trace_dir):
        print(f"Error: 目录 {trace_dir} 不存在")
        print("请先运行性能分析生成 trace 数据")
        exit(1)
    
    print("正在加载和可视化性能数据...")
    load_and_visualize(trace_dir)
    print("\n✓ 完成!")




