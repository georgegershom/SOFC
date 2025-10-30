#!/usr/bin/env python3
"""
SOFC数据集生成器快速测试
生成少量仿真数据进行测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sofc_dataset_generator import DatasetGenerator
import numpy as np

def quick_test():
    """快速测试数据集生成器"""
    print("=== SOFC数据集生成器快速测试 ===")
    
    # 配置参数（小规模测试）
    config = {
        'n_simulations': 5,  # 只生成5个仿真进行测试
        'n_parallel': 2,     # 使用2个并行进程
        'mesh_resolution': 20  # 降低网格分辨率以加快速度
    }
    
    print(f"测试配置: {config}")
    
    try:
        # 创建数据集生成器
        generator = DatasetGenerator(config)
        
        # 生成测试数据集
        print("开始生成测试数据集...")
        dataset = generator.generate_dataset(
            n_simulations=config['n_simulations'],
            n_parallel=config['n_parallel']
        )
        
        if dataset:
            print(f"✓ 成功生成 {len(dataset)} 个测试仿真")
            
            # 保存测试数据集
            test_filename = "sofc_test_dataset.h5"
            generator.save_dataset(test_filename)
            print(f"✓ 测试数据集已保存到: {test_filename}")
            
            # 创建测试摘要
            summary = generator.create_summary_report("test_summary.json")
            print(f"✓ 测试摘要已保存到: test_summary.json")
            
            # 显示一些统计信息
            print("\n=== 测试结果统计 ===")
            print(f"仿真数量: {len(dataset)}")
            
            if dataset:
                first_sim = dataset[0]
                print(f"网格节点数: {len(first_sim['mesh_points'])}")
                print(f"网格单元数: {len(first_sim['mesh_cells'])}")
                print(f"场数据类型: {list(first_sim['fields'].keys())}")
                
                # 显示温度场统计
                temp_data = first_sim['fields']['temperature']
                print(f"温度范围: {np.min(temp_data):.2f} - {np.max(temp_data):.2f} K")
                
                # 显示电流密度统计
                current_data = first_sim['fields']['current_density']
                current_magnitude = np.linalg.norm(current_data, axis=1)
                print(f"电流密度范围: {np.min(current_magnitude):.2e} - {np.max(current_magnitude):.2e} A/m²")
            
            print("\n✓ 快速测试完成！数据集生成器工作正常。")
            return True
            
        else:
            print("✗ 测试失败：没有生成任何仿真数据")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n可以继续运行完整的数据集生成器：")
        print("python sofc_dataset_generator.py")
    else:
        print("\n请检查错误信息并修复问题。")