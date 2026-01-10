"""
测试 YOLO 导入和模型加载
用于诊断 GitHub Actions 中 YOLO 检测被跳过的问题
"""
import os
import sys

print("=" * 70)
print("🔍 YOLO 导入测试")
print("=" * 70)

# 测试 1: 检查 Python 版本
print(f"\n1️⃣ Python 版本: {sys.version}")

# 测试 2: 尝试导入 ultralytics
print("\n2️⃣ 尝试导入 ultralytics...")
try:
    import ultralytics
    print(f"   ✅ ultralytics 导入成功")
    print(f"   📦 版本: {ultralytics.__version__}")
except ImportError as e:
    print(f"   ❌ ultralytics 导入失败: {e}")
    sys.exit(1)

# 测试 3: 尝试导入 YOLO
print("\n3️⃣ 尝试导入 YOLO 类...")
try:
    from ultralytics import YOLO
    print(f"   ✅ YOLO 类导入成功")
except ImportError as e:
    print(f"   ❌ YOLO 类导入失败: {e}")
    sys.exit(1)

# 测试 4: 检查模型文件
print("\n4️⃣ 检查模型文件...")
model_path = "model.onnx"
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path)
    print(f"   ✅ model.onnx 存在")
    print(f"   📦 文件大小: {file_size / (1024*1024):.2f} MB")
else:
    print(f"   ❌ model.onnx 不存在")
    print(f"   当前目录: {os.getcwd()}")
    print(f"   目录内容: {os.listdir('.')}")

# 测试 5: 尝试加载模型
if os.path.exists(model_path):
    print("\n5️⃣ 尝试加载 YOLO 模型...")
    try:
        model = YOLO(model_path, task="detect")
        print(f"   ✅ 模型加载成功")
        print(f"   📋 模型信息: {model}")
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("✅ 所有测试完成")
print("=" * 70)
