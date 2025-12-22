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

namespace YoloE_SegmentationDemo
{
    /// <summary>
    /// 使用 YoloDotNet 库和 YoloE 模型演示在静态图像上进行语义分割。
    /// 
    /// YoloE 是下一代实时多任务视觉模型，旨在"看见一切"。
    /// 此演示通过处理静态图像来展示 YoloE 的分割能力，绘制分割掩码
    /// 以及边界框、标签和置信度分数，然后将带注释的结果保存到磁盘。
    /// 
    /// 🔔 重要提示：
    /// - YoloDotNet 要求 YoloE 模型导出到 ONNX **时嵌入文本/视觉提示** 以进行零样本推理。
    ///   有关如何导出带自定义提示的 YoloE 模型的说明，请参见 README。
    /// 
    /// 演示的关键功能：
    /// - 使用自定义 YoloE ONNX 模型（带文本提示的零样本）进行模型初始化，可配置 CPU 或 GPU。
    /// - 像素级分割，可选的边界框、标签、置信度分数和彩色覆盖层。
    /// - 在静态上下文中演示的实时架构。
    /// - 对象和像素置信度的可调整阈值。
    /// - 带置信度分数的检测对象控制台日志记录。
    /// - 可调整质量和输出路径自定义的输出图像保存。
    /// - 支持自定义字体渲染、边界框颜色和动态缩放。
    /// - 通过 SegmentationDrawingOptions 扩展绘制管道。
    /// 
    /// 执行提供程序：
    /// - CpuExecutionProvider：在 CPU 上运行推理，通用支持但速度较慢。
    /// - CudaExecutionProvider：使用 CUDA 在 NVIDIA GPU 上执行推理以获得加速性能。  
    ///   可选地与 TensorRT 集成以进一步优化，支持 FP32、FP16 和 INT8 精度模式。  
    ///   这在兼容的 GPU 上提供显著的速度改进。
    /// 
    /// 重要说明：
    /// - 根据您可用的硬件和性能要求选择执行提供程序。  
    /// - 如果使用启用 TensorRT 的 CUDA，请确保您的环境具有兼容的 CUDA、cuDNN 和 TensorRT 设置。  
    /// - 此示例为简单起见使用静态图像，但 YoloE 也针对实时视频进行了优化。  
    /// - YoloDotNet 支持对象跟踪，但此处未演示。详情请参见 VideoStream 演示。  
    /// - 有关详细的设置说明和示例，请参见 README：  
    ///   https://github.com/NickSwardh/YoloDotNet
    /// </summary>
    internal class Program
    {
        private static string _outputFolder = default!;
        private static SegmentationDrawingOptions _drawingOptions = default!;

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

                ExecutionProvider = new CpuExecutionProvider(model: @"path\to\yoloE_model.onnx"),

                // 推理前应用的调整大小模式。Proportional 保持宽高比（如果需要则添加填充），
                // 而 Stretch 在不保持宽高比的情况下调整图像大小以适应目标大小。
                // 相应地设置此选项，因为它直接影响推理结果。
                ImageResize = ImageResize.Stretched,

                // 调整大小的采样选项；影响推理速度和质量。
                // 其他采样选项的示例，请参见基准测试：https://github.com/NickSwardh/YoloDotNet/tree/master/test/YoloDotNet.Benchmarks
                SamplingOptions = new(SKFilterMode.Nearest, SKMipmapMode.None) // YoloDotNet 默认值
            });

            // 打印模型类型
            Console.WriteLine($"Loaded ONNX Model: {yolo.ModelInfo}");

            // 加载输入图像为 SKBitmap（或 SKImage）
            // 图像从 SharedConfig 获取，用于测试/演示目的。
            using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.People));

            // 运行推理
            var results = yolo.RunSegmentation(image, confidence: 0.20, pixelConfedence: 0.5, iou: 0.7);

            // 绘制结果
            image.Draw(results, _drawingOptions);

            // 如果使用 SKImage，Draw 方法返回一个带有绘制结果的新 SKBitmap。
            // 示例：
            // using var resultImage = image.Draw(results, _drawingOptions);

            // 保存图像
            var fileName = Path.Combine(_outputFolder, $"Segmentation.jpg");
            image.Save(fileName, SKEncodedImageFormat.Jpeg, 80);

            PrintResults(results);
            DisplayOutputFolder();
        }

        private static void SetDrawingOptions()
        {
            // 设置绘制选项
            _drawingOptions = new SegmentationDrawingOptions
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
                DrawSegmentationPixelMask = true

                // 以下选项配置跟踪对象的轨迹，用于可视化 
                // 检测对象在一系列帧或图像中的移动路径。
                // 只有启用跟踪时才能绘制轨迹（例如，使用 SortTracker）。
                // 这在 VideoStream 演示中进行了演示。

                // DrawTrackedTail = false,
                // TailPaintColorEnd = new(),
                // TailPaintColorStart = new(),
                // TailThickness = 0,
            };
        }

        private static void PrintResults(List<Segmentation> results)
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
