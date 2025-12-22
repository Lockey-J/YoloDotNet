// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using SkiaSharp;
using System.Diagnostics;
using System.Globalization;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.ExecutionProvider.Cpu;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;
using YoloDotNet.Test.Common.Enums;

namespace OBBDetectionDemo
{
    /// <summary>
    /// 使用 YoloDotNet 库演示静态图像的定向边界框（OBB）检测。
    /// 
    /// 此演示加载示例图像，运行具有定向边界框（OBB）的对象检测，
    /// 叠加旋转的边界框与标签和置信度分数，并将处理后的图像保存到磁盘。
    /// 
    /// 包含的功能：
    /// - 使用可配置的硬件加速和预处理选项进行模型初始化
    /// - 用于检测具有定向边界框的对象的静态图像推理
    /// 可自定义的检测结果渲染（标签、置信度分数、旋转框）
    /// - 使用质量控制和自动输出文件夹创建保存标注的输出图像
    /// - 控制台报告检测结果
    /// 
    /// 执行提供程序：
    /// - CpuExecutionProvider：完全在 CPU 上运行推理。通用但速度较慢。
    /// - CudaExecutionProvider：使用 CUDA 在 NVIDIA GPU 上执行推理以获得加速性能。
    ///   可选择与 TensorRT 集成以进行进一步优化，支持 FP32、FP16 和 INT8 精度模式。
    /// 
    /// 重要说明：
    /// - 选择与您可用硬件和性能要求匹配的执行提供程序。
    /// - DetectionDrawingOptions 允许自定义字体、颜色、框渲染和标签样式。
    /// - 此演示在桌面上创建输出文件夹以存储处理后的结果。
    /// - 有关设置说明和最佳实践，请参阅 README：  
    ///   https://github.com/NickSwardh/YoloDotNet
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
                //     完全在 CPU 上运行推理。在所有硬件上通用支持。
                //
                //   - CudaExecutionProvider
                //     使用 CUDA 在 NVIDIA GPU 上执行推理以获得加速性能。  
                //     可选择与 TensorRT 集成以进一步优化，支持 FP32、FP16  
                //     和 INT8 精度模式。这在兼容的 GPU 上提供显著的速度改进。  
                //     有关详细配置和最佳实践，请参见 TensorRT 演示和文档。
                //
                //   - OpenVinoExecutionProvider
                //     使用 Intel 的 OpenVINO 工具包运行推理，在 Intel 硬件上获得优化性能。
                //
                //   - CoreMLExecutionProvider
                //     使用 Apple 的 CoreML 框架执行推理，在 macOS 和 iOS 设备上获得高效性能。
                //
                //   重要提示：  
                //     - 选择与您可用硬件和性能要求匹配的提供程序。  
                //     - 如果使用启用 TensorRT 的 CUDA，请确保您的环境具有兼容的 CUDA、cuDNN 和 TensorRT 设置。
                //     - 有关详细的设置说明和示例，请参见 README：
                //
                //   有关执行提供程序和设置说明的更多信息可以在 README 中找到：
                //   https://github.com/NickSwardh/YoloDotNet

                ExecutionProvider = new CpuExecutionProvider(SharedConfig.GetTestModelV11(ModelType.ObbDetection)),

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

            // 将输入图像加载为 SKBitmap（或 SKImage）
            // 图像从 SharedConfig 获取，用于测试/演示目的。
            using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.Island));

            // 执行 OBB 检测推理。
            var results = yolo.RunObbDetection(image, confidence: 0.25, iou: 0.7);

            // 绘制结果（可选）
            image.Draw(results, _drawingOptions);

            // 如果使用 SKImage，Draw 方法返回一个带有绘制结果的新 SKBitmap。
            // 示例：
            // using var resultImage = image.Draw(results, _drawingOptions);

            // 保存图像（可选）
            var fileName = Path.Combine(_outputFolder, "OBBDetection.jpg");
            image.Save(fileName, SKEncodedImageFormat.Jpeg, 80);

            PrintResults(results);
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
                // 只有启用跟踪时才能绘制尾迹（例如，使用 SortTracker）。
                // 此功能在 VideoStream 演示中进行了展示。

                // DrawTrackedTail = false,
                // TailPaintColorEnd = new(),
                // ailPaintColorStart = new(),
                // TailThickness = 0,
            };
        }

        private static void PrintResults(List<OBBDetection> results)
        {
            Console.WriteLine();
            Console.WriteLine($"Inference Results: {results.Count} objects");
            Console.WriteLine(new string('=', 80));

            Console.ForegroundColor = ConsoleColor.Blue;

            foreach (var result in results)
            {
                var label = result.Label.Name;
                var confidence = (result.Confidence * 100).ToString("0.##", CultureInfo.InvariantCulture);
                Console.WriteLine($"{label} ({confidence}%)");
            }

            Console.ForegroundColor = ConsoleColor.Gray;
        }

        private static void CreateOutputFolder()
        {
            _outputFolder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "YoloDotNet_Results");

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
