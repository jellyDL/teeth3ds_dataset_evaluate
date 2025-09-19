import os
import sys
import json
import numpy as np

color_trans_table = [
    [1.0, 1.0, 1.0], [1.0, 0.6, 0.0], [1.0, 1.0, 0.8], [0.4, 0.8, 0.6],
	[0.5, 0.9, 0.0], [0.0, 0.0, 0.8], [0.3, 0.2, 0.5], [1.0, 0.8, 0.7],
	[0.8, 0.3, 0.3], [0.8, 0.5, 0.0], [1.0, 1.0, 0.0], [0.5, 1.0, 0.8],
	[0.0, 1.0, 0.2], [0.1, 0.5, 1.0], [0.4, 0.3, 0.8], [1.0, 0.9, 0.7],
    [0.5, 0.2, 0.2], [1.0, 0.5, 0.0], [1.0, 0.8, 0.0], [0.0, 0.4, 0.0],
    [0.8, 1.0, 0.8], [0.0, 0.7, 1.0], [0.5, 0.4, 1.0], [0.5, 0.5, 0.4],
    [0.9, 0.3, 0.2], [1.0, 0.6, 0.0], [0.8, 0.6, 0.1], [0.3, 0.4, 0.2],
    [0.8, 1.0, 0.4], [0.3, 0.5, 0.7], [0.6, 0.0, 0.8], [0.8, 0.7, 0.6],
    [1.0, 0.0, 0.6], [1.0, 0.7, 0.8], [0.7, 0.5, 0.0], [0.6, 0.7, 0.5]
]

# 添加PLY文件处理库
try:
    import open3d as o3d
except ImportError:
    print("请安装open3d库: pip install open3d")
    sys.exit(1)

def parse_json(label_file):
    # Read the JSON file
    with open(label_file, 'r') as f:
        data = json.load(f)
        fdi_labels = data.get('labels', []) # FDI
        ins_labels = data.get('instances', []) # FDI
        jaw_type = data.get('jaw')  # 修改变量名，避免与内置函数type冲突
    return jaw_type, fdi_labels, ins_labels

def read_ply_vertices_colors(ply_file):
    """读取PLY文件中顶点的颜色信息"""
    if not os.path.exists(ply_file):
        print(f"文件不存在: {ply_file}")
        return None
    
    try:
        # 使用open3d读取PLY文件
        pcd = o3d.io.read_point_cloud(ply_file)
        # 获取顶点颜色，转换为numpy数组
        colors = np.asarray(pcd.colors)
        vertices = np.asarray(pcd.points)
        return colors, vertices
    except Exception as e:
        print(f"读取PLY文件时出错: {e}")
        return None

def colors_to_labels(colors):
    """将颜色转换为标签"""
    labels = []
    for color in colors:
        color = np.round(color, 2).tolist()
        # 如果color 在color_trans_table中，将其索引作为标签
        if color in color_trans_table:
            labels.append(color_trans_table.index(color))
        else:
            print("LLLLLLLLLLL")
    
    return np.array(labels)

def num_to_fdi(num_labels):
    fidnum = []
    for num in num_labels:
        num = int(num)
        if num >= 2 and num <= 9:
            fidnum.append(20 - num)
        elif num >= 10 and num <= 17:
            fidnum.append(num + 11)
        elif num >= 18 and num <= 25:
            fidnum.append(56 - num)
        elif num >= 26 and num <= 33:
            fidnum.append(num + 15)
        else:
            fidnum.append(num)
    return fidnum
            

if __name__ == "__main__":
    
    raw_data_dirs = ["../data_part_5/upper", "../data_part_5/lower", \
        "../data_part_6/upper", "../data_part_5/lower", "../data_part_6/lower"]
    out_data_dirs = ["../data_part_5_proc/upper_proc", "../data_part_5_proc/lower_proc" \
        "../data_part_6_proc/upper_proc",,"../data_part_6_proc/lower_proc"]
    
    # raw_data_dirs = ["../data_part_5/lower"]
    # out_data_dirs = ["../data_part_5_proc/lower_proc"]
    
    total_accuracy = 0
    total_num = 0
    
    # 处理所有目录下的数据
    for dir_idx, raw_data_dir in enumerate(raw_data_dirs):
        out_data_dir = out_data_dirs[dir_idx]
        
        folder_list = os.listdir(raw_data_dir)
        total_folders = len(folder_list)
        
        for iter, folder in enumerate(folder_list):
            
            if iter % 10 == 0:
                print("Processing {}/{} in {} :  {}...".format(iter, total_folders, raw_data_dir, folder))
            
            folder_path = os.path.join(raw_data_dir, folder)
            if not os.path.isdir(folder_path):
                continue
                
            # 查找json文件
            json_found = False
            for file in os.listdir(folder_path):
                if file.endswith(".json"):
                    label_file = os.path.join(folder_path, file)
                    jaw_type, fdi_labels, ins_labels = parse_json(label_file)
                    json_found = True
                    break
            
            if not json_found:
                continue
                    
            pred_file = os.path.join(out_data_dir, folder + "_" + jaw_type + "_label.ply")
            if not os.path.exists(pred_file):
                continue
                
            # 读取predfile顶点的所有颜色
            result = read_ply_vertices_colors(pred_file)
            if result is None:
                continue
                
            vertices_colors, vertices_points = result
            vertices_num_labels = colors_to_labels(vertices_colors)
            vertices_fdi_labels = num_to_fdi(vertices_num_labels)
             
            # print("## ",len(vertices_fdi_labels), len(fdi_labels))
            # a = 3000
            # b = a+20
            # print(vertices_num_labels[a:b])
            # print(vertices_fdi_labels[a:b])
            # print(fdi_labels[a:b])
            
            
            if len(vertices_num_labels) != len(fdi_labels):
                continue
            
            # 计算准确率
            correct_predictions = np.sum(np.array(vertices_fdi_labels) == np.array(fdi_labels))
            total_points = len(vertices_fdi_labels)
            accuracy = correct_predictions / total_points
            
            total_accuracy += accuracy
            total_num += 1
    
    print("total_num: ", total_num)
    print("Over all Results:", total_accuracy / total_num if total_num > 0 else "No valid data processed")

