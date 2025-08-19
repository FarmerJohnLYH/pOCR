import cv2
import numpy as np
import easyocr
import os
import re
import time
import datetime
import pandas as pd
from PIL import Image

def extract_text_with_easyocr(image, output_dir, file_prefix):
    """
    使用EasyOCR提取图像中的文字
    """
    print("正在使用EasyOCR识别文字...")
    # 初始化OCR读取器 - 使用中文和英文
    reader = easyocr.Reader(['ch_sim', 'en'])
    
    # 执行OCR识别
    results = reader.readtext(image)
    
    # 提取识别结果
    extracted_text = []
    for (bbox, text, prob) in results:
        # 只保留置信度较高的结果
        if prob > 0.5:
            extracted_text.append((text, prob, bbox))
            print(f"识别文字: {text} (置信度: {prob:.6f})")
    
    # 在图像上标记识别区域
    annotated_img = image.copy()
    for (text, prob, bbox) in extracted_text:
        # 绘制边界框
        pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_img, [pts], True, (0, 255, 0), 2)
        # 添加文字标签
        cv2.putText(annotated_img, text, (int(bbox[0][0]), int(bbox[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 保存标注后的图像
    annotated_path = os.path.join(output_dir, f"{file_prefix}_annotated.jpg")
    cv2.imwrite(annotated_path, annotated_img)
    print(f"标注后的图像已保存至: {annotated_path}")
    
    return extracted_text, annotated_img

def extract_name_and_id(text_results):
    """
    从OCR结果中提取姓名和工号
    """
    name = None
    employee_id = None
    
    # 正则表达式模式
    id_pattern = re.compile(r'\d{5,}')  # 假设工号是5位以上数字
    
    for text, prob, _ in text_results:
        # 查找工号 - 通常是纯数字
        if employee_id is None and id_pattern.search(text):
            employee_id = id_pattern.search(text).group()
            print(f"找到可能的工号: {employee_id}")
        
        # 查找姓名 - 通常是2-4个中文字符
        # 这里使用简单的启发式方法，实际应用中可能需要更复杂的逻辑
        if name is None and len(text) >= 2 and len(text) <= 4 and all('\u4e00' <= char <= '\u9fff' for char in text):
            name = text
            print(f"找到可能的姓名: {name}")
    
    return name, employee_id

def detect_badge_area(image, output_dir, file_prefix):
    """
    检测并截取工牌区域作为头像
    """
    print("正在检测工牌区域...")
    
    # 复制原始图像用于处理
    img_copy = image.copy()
    
    # 转换为灰度图
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    # 保存边缘检测结果
    edges_path = os.path.join(output_dir, f"{file_prefix}_edges.jpg")
    cv2.imwrite(edges_path, edges)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 按轮廓面积排序，取最大的几个轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    # 寻找可能的工牌四边形区域
    badge_contour = None
    for contour in contours:
        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)
        # 近似多边形
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # 如果是四边形，可能是工牌区域
        if len(approx) == 4:
            badge_contour = approx
            break
    
    # 如果找到了可能的工牌区域
    if badge_contour is not None:
        # 在原图上标记工牌区域
        badge_marked = image.copy()
        cv2.drawContours(badge_marked, [badge_contour], 0, (0, 255, 0), 2)
        marked_path = os.path.join(output_dir, f"{file_prefix}_badge_detected.jpg")
        cv2.imwrite(marked_path, badge_marked)
        
        # 创建掩码并提取工牌区域
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [badge_contour], 0, 255, -1)
        badge_img = cv2.bitwise_and(image, image, mask=mask)
        
        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(badge_contour)
        # 截取工牌区域
        badge_img = image[y:y+h, x:x+w]
        
        badge_path = os.path.join(output_dir, f"{file_prefix}_badge_area.jpg")
        cv2.imwrite(badge_path, badge_img)
        print(f"工牌区域已保存至: {badge_path}")
        
        return badge_img, badge_path
    
    # 如果没有找到合适的四边形，使用备选方法
    print("未检测到明确的工牌四边形区域，使用备选方法...")
    
    # 尝试使用颜色分割或其他特征来估计工牌区域
    # 这里简单地使用图像的中心区域作为工牌区域
    h, w = image.shape[:2]
    # 取图像中心的较大区域
    margin_x = int(w * 0.15)  # 左右边距为图像宽度的15%
    margin_y = int(h * 0.15)  # 上下边距为图像高度的15%
    badge_img = image[margin_y:h-margin_y, margin_x:w-margin_x]
    
    badge_path = os.path.join(output_dir, f"{file_prefix}_estimated_badge_area.jpg")
    cv2.imwrite(badge_path, badge_img)
    print(f"估计的工牌区域已保存至: {badge_path}")
    
    return badge_img, badge_path

def process_badge(image_path, index=0):
    """
    处理工牌图片，识别名字、工号，并截取头像
    """
    # 记录开始处理时间
    start_time = time.time()
    process_datetime = datetime.datetime.now()
    process_minute = process_datetime.minute
    process_second = process_datetime.second
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：文件 {image_path} 不存在")
        return None
    
    # 读取图像
    print(f"正在读取图像: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("错误：无法读取图像")
        return None
    
    # 获取输入文件名（不含扩展名）
    input_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # 创建结果目录
    output_dir = os.path.join(os.path.dirname(image_path), "ocr_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 按照格式重命名结果图像：序号_输入文件名_分钟_秒
    file_prefix = f"{index}_{input_filename}_{process_minute}_{process_second}"
    
    # 保存原始图像副本
    original_copy_path = os.path.join(output_dir, f"{file_prefix}_original.jpg")
    cv2.imwrite(original_copy_path, image)
    print(f"原始图像已保存至: {original_copy_path}")
    
    # 图像预处理
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 保存灰度图
    gray_path = os.path.join(output_dir, f"{file_prefix}_gray.jpg")
    cv2.imwrite(gray_path, gray)
    
    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应阈值处理，增强文字区域
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 保存预处理后的图像
    thresh_path = os.path.join(output_dir, f"{file_prefix}_preprocessed.jpg")
    cv2.imwrite(thresh_path, thresh)
    print(f"预处理图像已保存至: {thresh_path}")
    
    # 修改extract_text_with_easyocr函数调用，传入文件前缀
    text_results, annotated_img = extract_text_with_easyocr(image, output_dir, file_prefix)
    
    # 提取姓名和工号
    name, employee_id = extract_name_and_id(text_results)
    
    # 检测并截取工牌区域作为头像
    badge_img, badge_path = detect_badge_area(image, output_dir, file_prefix)
    
    # 计算处理时间
    end_time = time.time()
    process_time = end_time - start_time
    
    # 返回处理结果
    result = {
        "image": image,
        "gray": gray,
        "preprocessed": thresh,
        "output_dir": output_dir,
        "text_results": text_results,
        "name": name,
        "employee_id": employee_id,
        "badge_img": badge_img,
        "badge_path": badge_path,
        "process_time": process_time,
        "process_minute": process_minute,
        "process_second": process_second,
        "file_prefix": file_prefix
    }
    
    return result

# 主函数
def main():
    input_dir = "/Users/yuanhaoliu/Desktop/audio_demos/pOCR/input"
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误：输入目录 {input_dir} 不存在")
        # 创建输入目录
        os.makedirs(input_dir, exist_ok=True)
        print(f"已创建输入目录: {input_dir}")
        print(f"请将工牌图片放入该目录后重新运行程序")
        return
    
    # 获取目录中的所有图片文件
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file_path)
    
    if not image_files:
        print(f"错误：在 {input_dir} 目录中未找到图片文件")
        print(f"请将工牌图片放入该目录后重新运行程序")
        return
    
    print(f"找到 {len(image_files)} 个图片文件，开始处理...")
    
    # 创建结果表格数据
    results_data = []
    
    # 处理每个图片
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] 处理图片: {os.path.basename(image_path)}")
        
        result = process_badge(image_path, i+1)
        if not result:
            print(f"处理失败: {image_path}")
            continue
        
        print("\n处理完成！结果摘要:")
        print("-" * 50)
        
        if result["name"]:
            print(f"识别到的姓名: {result['name']}")
        else:
            print("未能识别姓名")
            
        if result["employee_id"]:
            print(f"识别到的工号: {result['employee_id']}")
        else:
            print("未能识别工号")
        
        print(f"工牌区域已保存至: {result['badge_path']}")
        print(f"所有处理结果保存在: {result['output_dir']}")
        print(f"处理时间: {result['process_time']:.2f} 秒")
        print("-" * 50)
        
        # 显示所有识别到的文本
        print("\n所有识别到的文本:")
        for text, prob, _ in result["text_results"]:
            print(f"- {text} (置信度: {prob:.6f})")
        
        # 添加结果到表格数据
        row_data = {
            "序号": i+1,
            "文件名": os.path.basename(image_path),
            "姓名": result["name"] if result["name"] else "未识别",
            "工号": result["employee_id"] if result["employee_id"] else "未识别",
            "处理时间(秒)": f"{result['process_time']:.2f}",
            "处理分钟": result["process_minute"],
            "处理秒": result["process_second"],
            "工牌图像路径": result["badge_path"]
        }
        
        # 添加所有识别文本和置信度
        for j, (text, prob, _) in enumerate(result["text_results"]):
            row_data[f"文本{j+1}"] = text
            row_data[f"置信度{j+1}"] = f"{prob:.6f}"
        
        results_data.append(row_data)
    
    # 将结果保存为CSV表格
    if results_data:
        # 创建DataFrame
        df = pd.DataFrame(results_data)
        
        # 保存为CSV文件
        csv_path = os.path.join(input_dir, "ocr_results", "recognition_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n识别结果已保存到表格: {csv_path}")

if __name__ == "__main__":
    main()