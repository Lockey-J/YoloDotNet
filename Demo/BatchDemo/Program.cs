// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using SkiaSharp;
using System.Diagnostics;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.ExecutionProvider.Cuda;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;

namespace BatchDemo
{
    /// <summary>
    /// 演示使用 YoloDotNet 对静态图像进行批量对象检测，具有并行
    /// 处理功能，可在大数据集上实现更快的推理。
    ///
    /// 此演示：
    /// - 从文件夹加载图像
    /// - 并行运行 YOLO 对象检测
    /// - 绘制检测结果（边界框、标签、置信度分数）
    /// - 将标注图像保存到桌面结果文件夹
    ///
    /// 主要亮点：
    /// - 并行批处理以加速推理
    /// - 灵活的执行提供程序配置（CPU、CUDA、TensorRT、OpenVINO）
    /// - 可自定义的绘制选项，用于文本、颜色和边界框样式
    /// </summary>
    internal class Program
    {
        private static string _outputFolder = default!;
        private static DetectionDrawingOptions _drawingOptions = default!;

        static void Main(string[] args)
        {
            CreateOutputFolder();
            SetDrawingOptions();

            // 初始化 YoloDotNet。
            // YoloOptions 配置模型、硬件设置和图像处理行为。
            using var yolo = new Yolo(new YoloOptions
            {
                // 选择执行提供程序（确定推理的执行方式和位置）。
                // 可用的执行提供程序：
                // 
                //   - CpuExecutionProvider
                //     完全在 CPU 上运行推理。所有硬件上通用支持。
                //
                //   - CudaExecutionProvider
                //     使用 CUDA 在 NVIDIA GPU 上执行推理以获得加速性能。  
                //     可选地与 TensorRT 集成以进一步优化，支持 FP32、FP16  
                //     和 INT8 精度模式。这在兼容的 GPU 上提供显著的速度改进。  
                //     有关详细配置和最佳实践，请参见 TensorRT 演示和文档。
                //
                //   - OpenVinoExecutionProvider
                //     在 Intel GPU 上运行加速推理，在 Intel 硬件上获得优化性能。
                //
                //   - CoreMLExecutionProvider
                //     在 Apple GPU 上运行加速推理，在 macOS 和 iOS 设备上获得高效性能。
                //
                //   重要提示：  
                //     - 选择与您可用硬件和性能要求匹配的提供程序。  
                //     - 如果使用启用 TensorRT 的 CUDA，请确保您的环境具有兼容的 CUDA、cuDNN 和 TensorRT 设置。
                //     - 有关详细的设置说明和示例，请参见 README：
                //
                //   有关执行提供程序和设置说明的更多信息可以在 README 中找到：
                //   https://github.com/NickSwardh/YoloDotNet

                ExecutionProvider = new CudaExecutionProvider(

                    // 要加载的 ONNX 模型的路径或字节数组。
                    model: SharedConfig.GetTestModelV11(ModelType.ObjectDetection),

                    // 用于推理的GPU设备ID。-1 = CPU，0+ = GPU设备ID。
                    gpuId: 0),

                // 推理前应用的调整大小模式。Proportional 保持宽高比（如果需要则添加填充），
                // 而 Stretch 在不保持宽高比的情况下调整图像大小以适应目标大小。
                // 相应地设置此选项，因为它直接影响推理结果。
                ImageResize = ImageResize.Proportional,

                // 调整大小的采样选项；影响推理速度和质量。
                // 其他采样选项的示例，请参见基准测试：https://github.com/NickSwardh/YoloDotNet/tree/master/test/YoloDotNet.Benchmarks
                SamplingOptions = new(SKFilterMode.Nearest, SKMipmapMode.None) // YoloDotNet 默认值
            });

            // 打印模型类型
            Console.WriteLine($"Loaded ONNX Model: {yolo.ModelInfo}");

            // 收集图像
            var images = Directory.GetFiles(@"path\to\image\folder");

            Parallel.ForEach(images, image =>
            {
                // 将输入图像加载为 SKBitmap（或 SKImage）
                using var img = SKBitmap.Decode(image);

                // 运行对象检测推理
                var results = yolo.RunObjectDetection(img, 0.25, 0.7);

                // 使用自定义 _drawingOptions 绘制结果（可选）
                img.Draw(results, _drawingOptions);

                // 保存图像
                var fileName = Path.Combine(_outputFolder, $"ObjectDetection_{Guid.NewGuid()}.jpg");
                img.Save(fileName, SKEncodedImageFormat.Jpeg, 80);
            });

            DisplayOutputFolder();
        }

        private static void SetDrawingOptions()
        {
            // 设置绘制选项
            _drawingOptions = new DetectionDrawingOptions
            {
                DrawBoundingBoxes = true,
                DrawConfidenceScore = true,
                DrawLabels = true,
                EnableFontShadow = true,

                // SKTypeface 定义用于文本渲染的字体。
                // SKTypeface.Default 使用系统默认字体。
                // 加载自定义字体：
                //   - 使用 SKTypeface.FromFamilyName("fontFamilyName", SKFontStyle) 按字体系列名称加载（如果已安装）。
                //   - 使用 SKTypeface.FromFile("path/to/font.ttf") 直接从文件加载字体。
                // 示例：
                //   Font = SKTypeface.FromFamilyName("Arial", SKFontStyle.Normal)
                //   Font = SKTypeface.FromFile("C:\\Fonts\\CustomFont.ttf")
                Font = SKTypeface.Default,

                FontSize = 18,
                FontColor = SKColors.White,
                DrawLabelBackground = true,
                EnableDynamicScaling = true,
                BorderThickness = 2,

                // 默认情况下，YoloDotNet 自动为边界框分配颜色。
                // 要覆盖这些默认颜色，您可以定义自己的十六进制颜色代码数组。
                // 数组中的每个元素对应模型中的类别索引。
                // 示例：
                //   BoundingBoxHexColors = ["#00ff00", "#547457", ...] // 每个类别 ID 的颜色

                BoundingBoxOpacity = 128,

                // 以下选项配置跟踪对象尾迹，用于可视化
                // 检测对象在一系列帧或图像中的移动路径。
                //
                // ⚠ 在使用并行处理运行批量推理时不建议使用跟踪。
                // 为了使跟踪（例如，使用 SortTracker）正常工作，必须按顺序处理帧
                // 以便跟踪器可以在帧之间维持对象状态。
                // 在独立的帧上并行运行跟踪将产生不正确或不可预测的结果。
                //
                // 此功能在 VideoStream 演示中进行了展示，其中强制执行顺序处理。
                //
                // DrawTrackedTail = false,
                // TailPaintColorEnd = new(),
                // TailPaintColorStart = new(),
                // TailThickness = 0,
            };
        }

        private static void CreateOutputFolder()
        {
            _outputFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "YoloDotNet_Results");
            _outputFolder = Path.Combine(_outputFolder, "batch");

            if (Directory.Exists(_outputFolder) is false)
                Directory.CreateDirectory(_outputFolder);
        }

        private static void DisplayOutputFolder()
        {
            var shell = OperatingSystem.IsWindows() ? "explorer"
                     : OperatingSystem.IsLinux() ? "xdg-open"
                     : OperatingSystem.IsMacOS() ? "open"
                     : null;

            if (shell is not null)
                Process.Start(shell, _outputFolder);
            else
                Console.WriteLine($"Results saved to: {_outputFolder}");
        }
    }
}
