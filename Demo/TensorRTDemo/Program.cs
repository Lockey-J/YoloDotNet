// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using SkiaSharp;
using System.Diagnostics;
using System.Globalization;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.ExecutionProvider.Cuda;
using YoloDotNet.ExecutionProvider.Cuda.TensorRT;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;
using YoloDotNet.Test.Common.Enums;

namespace TensorRTDemo
{
    /// <summary>
    /// 演示如何通过 CUDA 和 TensorRT 使用 YoloDotNet 进行 GPU 加速。
    /// 
    /// 此演示使用 YOLOv11 ONNX 模型在静态图像上执行对象检测，
    /// 通过带 TensorRT 优化的 CUDA（FP32、FP16 或 INT8）进行加速。
    /// 图像使用边界框、类别标签和置信度分数进行注释，
    /// 结果保存到磁盘。
    /// 
    /// 它展示了：
    /// - 带 TensorRT 精度（FP32、FP16、INT8）的 CUDA 支持 GPU 推理
    /// - 可配置硬件和预处理选项的模型初始化
    /// - 使用边界框检测对象的静态图像推理
    /// - 检测结果的可自定义渲染（标签、置信度、框）
    /// - 将带注释的输出保存到磁盘
    /// - 推理结果的控制台报告
    /// 
    /// 执行提供程序：
    /// - CpuExecutionProvider：在 CPU 上运行推理（较慢但通用支持）。
    /// - CudaExecutionProvider：通过 NVIDIA GPU 使用 CUDA 进行更快推理。  
    ///   可选地与 TensorRT 集成以获得高度优化的性能  
    ///   具有可配置精度（FP32、FP16、INT8）。
    /// 
    /// 重要说明：
    /// - 根据硬件和性能要求选择提供程序。
    /// - 对于 TensorRT 加速，请配置 CudaExecutionProvider 的 `trtConfig` 参数。
    /// - 需要兼容的 NVIDIA GPU、CUDA/cuDNN 和已安装的 TensorRT 运行时。
    /// </summary>
    internal class Program
    {
        private static string _outputFolder = default!;
        private static string _trtEngineCacheFolder = default!;
        private static DetectionDrawingOptions _drawingOptions = default!;

        static void Main(string[] args)
        {
            CreateOutputFolder();
            SetDrawingOptions();

            Console.WriteLine("Loading or building TensorRT engine cache...\n" +
                "- If a compatible engine cache exists, it will be loaded automatically.\n" +
                "- Otherwise, TensorRT will build a new optimized engine. This may take\n" +
                "  several seconds or minutes depending on model complexity and hardware.\n" +
                "- For more details on TensorRT usage and configuration, see the README\n" +
                "  included with this demo.\n");

            // 初始化 YoloDotNet。
            // YoloOptions 配置模型、硬件设置和图像处理行为。
            using var yolo = new Yolo(new YoloOptions
            {
                // 选择执行提供程序（确定推理的执行方式和位置）。
                // 可用的执行提供程序：
                // 
                // - CpuExecutionProvider  
                //   Runs inference entirely on the CPU. Universally supported but typically slower.
                // 
                // - CudaExecutionProvider  
                //   Executes inference on an NVIDIA GPU using CUDA for accelerated performance.  
                //   Optionally integrates with TensorRT for further optimization, supporting FP32, FP16,  
                //   and INT8 precision modes. This delivers significant speed improvements on compatible GPUs.  
                //   See the TensorRT demo and documentation for detailed configuration and best practices.
                // 
                // 重要提示：  
                // - Choose the provider that matches your available hardware and performance requirements.  
                // - If using CUDA with TensorRT enabled, ensure your environment has a compatible CUDA, cuDNN, and TensorRT setup.
                // - For detailed setup instructions and examples, see the README:  
                //   https://github.com/NickSwardh/YoloDotNet

                ExecutionProvider = new CudaExecutionProvider(

                    // 要加载的 ONNX 模型的路径或字节数组。
                    model: SharedConfig.GetTestModelV11(ModelType.ObjectDetection),

                    // 用于推理的GPU设备ID。-1 = CPU，0+ = GPU设备ID。
                    gpuId: 0,

                    // TensorRT 执行的可选配置。
                    trtConfig: new TensorRt
                    {
                        Precision = TrtPrecision.FP16,
                        // - FP32: Full precision (32-bit float). Default mode. Highest accuracy, default execution.
                        // - FP16: Half precision (16-bit float). Offers improved performance on supported GPUs with minimal accuracy loss.
                        // - INT8: Integer precision (8-bit). Fastest inference performance, but requires calibration.
                        //
                        //   Note: INT8 mode enables **mixed precision execution**.
                        //   TensorRT will use INT8 precision where supported, and automatically fall back to FP16 or FP32
                        //   for layers or operations that are not quantizable — due to model structure, unsupported ops,
                        //   dynamic ranges, or numerical stability concerns.
                        //
                        //   See Int8CalibrationCacheFile for calibration file requirements.

                        BuilderOptimizationLevel = 3,
                        // 设置构建新引擎缓存时使用的构建器优化级别。更高级别
                        // 允许 TensorRT 在更多优化选项上花费更多构建时间。
                        //
                        // 警告：低于 3 的级别不能保证良好的引擎性能，但会大大改善
                        // 构建时间。默认为 3，有效范围为 [0 - 5]。

                        EngineCachePath = _trtEngineCacheFolder,
                        // 指定 TensorRT 将存储和加载引擎缓存文件的目录。
                        //
                        // 引擎缓存避免在每次启动时重新构建 TensorRT 引擎，
                        // 显著改善初始化时间。
                        //
                        // 如果缓存文件已存在于当前模型、硬件、精度和配置，
                        // 它们将被自动重用。否则，将在此处构建并存储新的引擎缓存。
                        //
                        // 注意：现有缓存文件永远不会被自动删除。
                        // 您必须根据需要从此目录手动删除过时或未使用的缓存文件。

                        EngineCachePrefix = "YoloDotNet",
                        // 为生成的 TensorRT 引擎和配置文件缓存设置文件名前缀。
                        //
                        // 这有助于区分来自不同模型、版本或配置的缓存文件，
                        // 特别是当多个引擎存储在同一个 EngineCachePath 中时。
                        //
                        // 如果留空，将使用默认内部前缀。

                        Int8CalibrationCacheFile = Path.Join(SharedConfig.AbsoluteAssetsPath, "cache", "yolov11s.cache"),
                        // TensorRT INT8 校准缓存文件的可选路径。
                        // 仅在明确启用 INT8 精度模式时使用；否则，它将被忽略。
                        // 如果您不使用 INT8 模式，可以留空。
                        //
                        // 指定引擎构建期间使用的 INT8 校准缓存文件的路径。
                        // 在 INT8 模式下使用非量化模型时需要此文件。
                        // TensorRT 使用它为张量分配动态范围。
                        //
                        // 校准缓存必须使用原始 model.pt 数据预先生成。
                        //
                        // 🔧 To generate the calibration cache, export the model using the Ultralytics CLI:
                        //
                        //   yolo export model=your_model.pt format=engine int8=true simplify=true data=your_model_dataset.yaml opset=17
                        //
                        // 此命令生成：
                        //   - 标准 ONNX 模型（未量化，基于 FP32）
                        //   - 为 INT8 精度优化的 TensorRT 引擎
                        //   - 校准缓存文件：<model_name>.cache
                        //
                        // 必须指定 <model_name>.cache 的路径才能在 INT8 混合精度模式下运行 YOLO ONNX 模型。
                        // 示例：
                        //   Int8CalibrationCacheFile = @"path\to\<model_name>.cache"
                    }),

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
            using var image = SKBitmap.Decode(SharedConfig.GetTestImage(ImageType.Street));

            // 运行对象检测推理
            var results = yolo.RunObjectDetection(image, confidence: 0.15, iou: 0.7);

            // 绘制结果
            image.Draw(results, _drawingOptions);

            // 如果使用 SKImage，Draw 方法返回一个带有绘制结果的新 SKBitmap。
            // 示例：
            // using var resultImage = image.Draw(results, _drawingOptions);

            // 保存图像
            var fileName = Path.Combine(_outputFolder, "ObjectDetection.jpg");
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

        private static void PrintResults(List<ObjectDetection> results)
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
            _outputFolder = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "YoloDotNet_Results");
            _trtEngineCacheFolder = Path.Join(_outputFolder, "TensorRT_Engine_Cache");

            var folder = _trtEngineCacheFolder;

            if (Directory.Exists(folder) is false)
                Directory.CreateDirectory(folder);
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
