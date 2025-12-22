# <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/994287a9-556c-495f-8acf-1acae8d64ac0" height=24> YoloDotNet

🚀 **为 .NET 打造的超快、生产就绪的 YOLO 推理**

**YoloDotNet** 是一个功能齐全的 C# 库，用于使用 **YOLOv5u–v12**、**YOLO-World** 和 **YOLO-E** 模型进行实时计算机视觉。

基于 **.NET 8** 和 **ONNX Runtime** 构建，它在 **Windows、Linux 和 macOS** 上提供 **高性能、可预测的推理** —— 对执行、内存和预处理进行显式控制。

无需 Python。没有魔法。只有快速、确定性的 YOLO —— 为 .NET 正确实现。

## ⭐ 为什么选择 YoloDotNet？

YoloDotNet 为需要以下功能的开发者设计：

- ✅ **纯 .NET** — 无需 Python 运行时，无需脚本
- ✅ **真实性能** — CPU、CUDA / TensorRT、OpenVINO、CoreML
- ✅ **显式配置** — 可预测的精度和内存使用
- ✅ **生产就绪** — 引擎缓存、长期运行稳定性
- ✅ **多种视觉任务** — 检测、OBB、分割、姿态估计、分类

YoloDotNet 是构建 **桌面应用、后端服务或实时视觉管道** 的 .NET 开发者的理想选择，这些开发者需要可预测的性能和显式控制。

## 🆕 What’s New in v4.0

- Modular execution providers (CPU, CUDA/TensorRT, OpenVINO, CoreML)
- New OpenVINO and CoreML providers
- Cleaner dependency graph
- Improved GPU behavior and predictability
- Grayscale ONNX model support

📖 Full release history: [CHANGELOG.md](./CHANGELOG.md)

## 🚀 快速开始

### 1️⃣ 安装核心包
```bash
dotnet add package YoloDotNet
```
### 2️⃣ 安装一个执行提供程序
```bash
# CPU（推荐的起点）
dotnet add package YoloDotNet.ExecutionProvider.Cpu

# 可选的 GPU 加速
dotnet add package YoloDotNet.ExecutionProvider.Cuda
dotnet add package YoloDotNet.ExecutionProvider.OpenVino
dotnet add package YoloDotNet.ExecutionProvider.CoreML
```

💡 注意：CUDA 执行提供程序包含可选的 TensorRT 加速。不需要单独的 TensorRT 包。

### 3️⃣ 运行目标检测
```csharp
using SkiaSharp;
using YoloDotNet;
using YoloDotNet.ExecutionProvider.Cpu;

using var yolo = new Yolo(new YoloOptions
{
    ExecutionProvider = new CpuExecutionProvider("model.onnx")
});

using var image = SKBitmap.Decode("image.jpg");

var results = yolo.RunObjectDetection(image, confidence: 0.25, iou: 0.7);

image.Draw(results);
image.Save("result.jpg");
```

You’re now running YOLO 推理。

## 💡 重要：精度取决于配置

YOLO 推理精度 **不是自动的**。

预处理设置如图像调整大小模式、采样方法和置信度/IoU 阈值 **必须与模型的训练方式匹配**。这些设置直接控制精度-性能权衡，应被视为模型本身的一部分。

📖 **在调优模型或比较结果之前，请阅读：**  
👉 [精度与配置指南](./AccuracyAndConfiguration.md)

## 支持的任务

| 分类 | 目标检测 | OBB 检测 | 分割 | 姿态估计 |
|----------------|------------------|---------------|--------------|-----------------|
| <img src="https://user-images.githubusercontent.com/35733515/297393507-c8539bff-0a71-48be-b316-f2611c3836a3.jpg" width=300> | <img src="https://user-images.githubusercontent.com/35733515/273405301-626b3c97-fdc6-47b8-bfaf-c3a7701721da.jpg" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/d15c5b3e-18c7-4c2c-9a8d-1d03fb98dd3c" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/3ae97613-46f7-46de-8c5d-e9240f1078e6" width=300> | <img src="https://github.com/NickSwardh/YoloDotNet/assets/35733515/b7abeaed-5c00-4462-bd19-c2b77fe86260" width=300> |
| <sub>[图片来自 pexels.com](https://www.pexels.com/photo/hummingbird-drinking-nectar-from-blooming-flower-in-garden-5344570/)</sub> | <sub>[图片来自 pexels.com](https://www.pexels.com/photo/men-s-brown-coat-842912/)</sub> | <sub>[图片来自 pexels.com](https://www.pexels.com/photo/bird-s-eye-view-of-watercrafts-docked-on-harbor-8117665/)</sub> | <sub>[图片来自 pexels.com](https://www.pexels.com/photo/man-riding-a-black-touring-motorcycle-903972/)</sub> | <sub>[图片来自 pexels.com](https://www.pexels.com/photo/woman-doing-ballet-pose-2345293/)</sub> |

## 📁 示例

演示文件夹中提供了实践示例，涵盖常见的真实场景：

👉 [浏览示例项目](./Demo)

包括图像推理、视频流、GPU 加速、分割和大图像工作流程。

## 执行提供程序

| 执行提供程序 | Windows | Linux | macOS | 文档 |
| ------------------ | ------- | ----- | ----- | ------------- |
| CPU | ✅ | ✅ | ✅ | [CPU README](./YoloDotNet.ExecutionProvider.Cpu/README.md) |
| CUDA / TensorRT | ✅ | ✅ | ❌ | [CUDA README](./YoloDotNet.ExecutionProvider.Cuda/README.md) |
| OpenVINO | ✅ | ✅ | ❌ | [OpenVINO README](./YoloDotNet.ExecutionProvider.OpenVino/README.md) |
| CoreML | ❌ | ❌ | ✅ | [CoreML README](./YoloDotNet.ExecutionProvider.CoreML/README.md) |

每个执行提供程序都有自己的 README，涵盖安装、运行时要求和特定提供程序的配置。  
真实世界使用示例和推荐设置可以在示例项目中找到。

> ℹ️ 只能引用 **一个** 执行提供程序包。  
> 每个提供程序都附带自己的原生 ONNX Runtime 二进制文件；混合使用提供程序将导致运行时冲突。

## ⚡ 性能特征

YoloDotNet 专注于稳定、低开销的推理行为，其中运行时成本主要由所选的执行提供程序和模型决定，而不是框架开销。

📊 参见：[基准测试方法和结果](/test/YoloDotNet.Benchmarks/README.md)。

使用 **BenchmarkDotNet** 进行的内部基准测试涵盖分类、目标检测、OBB、姿态估计和分割，显示：

- 推理延迟在预热后保持稳定
- 性能随所选执行提供程序（CPU → GPU → TensorRT）清晰地扩展
- TensorRT 精度模式（FP32、FP16、INT8）表现符合预期
- 分配行为是可预测的，并受输出复杂性限制
- 整体吞吐量主要由执行提供程序和模型配置决定

对于基于 GPU 的提供程序，第一次推理可能由于初始化或引擎创建而较慢；后续运行以稳态性能运行。

YoloDotNet 适用于：
- 实时管道
- 长期运行的服务
- 高分辨率图像处理
- 确定性生产工作负载

## 🚀 模块化执行提供程序

YoloDotNet 使用 **完全模块化的执行架构**，使开发人员能够对原生依赖项和运行时行为进行显式控制。

- 核心包与执行提供程序无关
- 执行提供程序作为单独的 NuGet 包提供
- 原生 ONNX Runtime 依赖项按提供程序隔离

### 为什么这很重要
- 更少的原生依赖冲突
- 更清洁和更可预测的部署
- 跨平台和运行时的一致行为
- 更容易集成到生产和长期运行的服务中

💡 **现有用户注意**  
从早期版本升级的项目必须引用一个执行提供程序包并相应地更新提供程序设置。现有模型保持完全兼容。

## 支持 YoloDotNet
YoloDotNet is built and maintained independently. If you’ve found my project helpful, consider supporting its development:

⭐ 为仓库加星\
💬 分享反馈\
🤝 考虑赞助开发

[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/NickSwardh) [![PayPal](https://img.shields.io/badge/Support-PayPal-00457C?logo=paypal&logoColor=white)](https://paypal.me/nickswardh)


谢谢您。 ❤️

## 参考资料与致谢

https://github.com/ultralytics/ultralytics \
https://github.com/sstainba/Yolov8.Net \
https://github.com/mentalstack/yolov5-net

## 许可证

YoloDotNet 版权所有 © 2023–2025 Niklas Swärd ([GitHub](https://github.com/NickSwardh/YoloDotNet))  
根据 **GNU General Public License v3.0 或更高版本** 授权。

根据 GPL v3 条款允许商业使用；但是，衍生作品必须遵守相同的许可证。

![License: GPL v3 or later](https://img.shields.io/badge/License-GPL_v3_or_later-blue)  
完整的许可证文本请参见 [LICENSE](./LICENSE.txt) 文件。

This software is provided “as is”, 不作任何形式的保证。  
作者对其使用所造成的任何损害不承担责任。