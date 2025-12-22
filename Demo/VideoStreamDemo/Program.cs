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
using YoloDotNet.Test.Common.Enums;
using YoloDotNet.Trackers;
using YoloDotNet.Video;

namespace VideoStreamDemo
{
    /// <summary>
    /// 演示使用 YoloDotNet 库在视频或直播流上进行对象检测和跟踪。
    /// 
    /// 需要安装 FFmpeg 和 FFprobe 并添加到系统 PATH：
    /// https://ffmpeg.org/download.html
    ///
    /// 此演示加载视频源（文件、直播流或摄像头），在每一帧上运行对象检测和可选跟踪，
    /// 绘制结果（边界框、标签、置信度分数和跟踪轨迹），并可选择保存输出视频。
    ///
    /// 它展示了：
    /// - 使用可配置的硬件和预处理选项进行模型初始化
    /// - 使用 YoloDotNet 在视频流上进行实时对象检测
    /// - 按类别标签过滤检测结果
    /// - 使用 SORT 跟踪器跨帧跟踪多个对象
    /// - 直接在视频帧上渲染检测和跟踪结果
    /// - 保存处理后的视频输出并可选择分割成块
    /// - 使用可自定义回调进行进度报告和流结束处理
    ///
    /// 视频源输入示例：
    /// - 本地视频文件路径：
    ///     Example: @"C:\videos\test.mp4"
    ///
    /// - 直播流 URL (RTMP, HTTP, 等):
    ///     示例: "rtmp://your.server/stream"
    ///
    /// - 带有明确分辨率和帧率的视频捕获设备（网络摄像头）:
    ///     格式: "device=<DeviceName>:<Width>:<Height>:<FPS>"
    ///
    ///     Windows 示例: "device=Logitech BRIO:1920:1080:30"
    ///     Linux 示例:   "device=/dev/video0:1280:720:30"
    ///
    ///     注意: 宽度、高度和帧率必须与设备支持的捕获模式匹配。
    ///
    /// 执行提供程序:
    /// - CpuExecutionProvider: 在 CPU 上运行推理，通用支持但速度较慢。
    /// - CudaExecutionProvider: 通过 CUDA 使用 NVIDIA GPU 获得加速性能。
    ///   可选地与 TensorRT 集成以进一步优化，支持 FP32、FP16
    ///   和 INT8 精度模式。这在兼容的 GPU 上提供显著的速度改进。
    ///
    /// 重要说明:
    /// - 根据硬件和性能要求选择执行提供程序。
    /// - FFmpeg 和 FFprobe 必须添加到系统 PATH 变量。下载并安装: https://ffmpeg.org/download.html
    /// - 演示在桌面上创建输出文件夹以存储处理结果。
    /// </summary>
    internal class Program
    {
        private static string _outputFolder = default!;
        private static DetectionDrawingOptions _drawingOptions = default!;
        private static SortTracker _sortTracker = default!;

        static void Main(string[] args)
        {
            // （可选）使用可配置参数创建新的 SortTracker 实例：
            // - costThreshold：将检测分配到轨道的匹配成本阈值（越低 = 匹配越严格）。
            // - maxAge：在删除前保持未匹配轨道的帧数。
            // - tailLength：用于可视化或分析的轨道历史长度。
            // 注意：没有一刀切的设置；这些参数通常需要一些调整才能为您的特定用例找到最佳平衡。
            _sortTracker = new SortTracker(0.5f, 5, 60);

            CreateOutputFolder();
            SetDrawingOptions();
            Console.CursorVisible = false;

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

            // 列出系统上检测到的所有可用视频输入设备。
            var devices = Yolo.GetVideoDevices();
            Console.WriteLine();
            Console.WriteLine("检测到的视频输入设备（可用于 VideoOptions）：");

            if (devices.Count != 0)
            {
                foreach (string device in devices)
                    Console.WriteLine($"  {device}");
            }
            else
                Console.WriteLine("未找到输入设备");

            // 设置视频选项。
            yolo.InitializeVideo(new VideoOptions
            {
                // 💡 输入视频源。接受的格式：
                // 
                // 1. 本地视频文件：
                //    示例：@"C:\videos\test.mp4"
                //
                // 2. 直播流 URL（例如 RTMP、HTTP）：
                //    示例："rtmp://your.rtmp.server/stream"
                //
                // 3. 视频捕获设备（例如网络摄像头）：
                //    格式："device=<DeviceName>:<Width>:<Height>:<FPS>"
                //
                //    ⮞ 在 Windows 上：
                //       示例："device=Logitech BRIO:1920:1080:30"
                //
                //    ⮞ 在 Linux 上：
                //       示例："device=/dev/video0:1280:720:30"
                //
                // 📌 宽度、高度和 FPS 值必须匹配您的摄像头支持的有效捕获模式。
                //
                // 🔍 要发现可用的视频设备：
                //    使用 `yolo.GetVideoDevices()` — 此方法列出可用视频捕获设备的名称。
                //    它不会列出支持的分辨率或帧率。
                //
                //    要确定有效的宽度/高度/FPS 组合，请参考您的设备规格
                VideoInput = SharedConfig.GetTestVideo(VideoType.PeopleWalking),

                // 💡 可选：保存处理后输出视频文件的路径。
                // 如果您不想保存输出，请保持未设置（null 或空）。
                VideoOutput = Path.Combine(_outputFolder, "video_output.mp4"),

                // 💡 写入输出时使用的编码器。
                // 注意：确保所选编码器与以下内容兼容：
                //   - 您的操作系统和硬件
                //   - FFmpeg 已编译所选编码器
                //   - 输出文件格式/容器
                VideoEncoder = VideoEncoder.H264Nvenc, // 使用 NVIDIA GPU 编码器 (h264_nvenc)。编码器替代方案请参见 Encoder 枚举注释。

                // 💡 输出视频的帧率。
                // FrameRate.AUTO 将尝试匹配 the input video’s frame rate.
                FrameRate = FrameRate.AUTO,

                // 💡 输出视频的宽度（像素）。
                // 保持未设置 (0) 以使用原始宽度。
                // 设置为 -2 以在保持宽高比的同时根据指定高度自动计算宽度。
                // 注意：一次只能将宽度或高度中的一个设置为 -2。
                Width = 720,

                // 💡 输出视频的高度（像素）。
                // 保持未设置 (0) 以使用原始高度。
                // 设置为 -2 以在保持宽高比的同时根据指定宽度自动计算高度。
                // 注意：一次只能将宽度或高度中的一个设置为 -2。
                Height = -2,

                // 💡 输出视频的压缩质量（1-51）。
                // 较低的值 = 更好的质量，更大的文件大小。
                // 较高的值 = 更强的压缩，更小的文件大小，更低的质量。
                // 推荐范围：20-35 以获得合理的平衡。
                CompressionQuality = 30,

                // 💡 可选：自动将输出视频分割成块。
                // 每个块的持续时间（秒）。
                // 示例：600 = 分割成 10 分钟的片段。
                // 0 = 不分割（生成单个文件）。
                VideoChunkDuration = 0,

                // 💡 处理每隔 N 帧。
                // 0 = 处理所有帧（默认）。
                // 示例：30 = 处理每第 30 帧（对于不需要全帧检测的监控很有用）。
                FrameInterval = 0,

                // 💡 可选：定义要处理的视频的特定片段。
                // 对于测试或仅处理视频的一部分很有用。
                // 注意：仅适用于视频文件。
                // 0 = 从开始处开始。
                StartTimeSeconds = 0,

                // 💡 可选：要处理的视频片段的持续时间（秒）。
                // 对于测试或仅处理视频的一部分很有用。
                // 注意：仅适用于视频文件。
                // 0 = 处理到视频结束。
                DurationSeconds = 0
            });

            // 在处理开始前显示基本视频元数据。
            var metadata = yolo.GetVideoMetaData();
            PrintMetaData(metadata);

            var listedDevices = devices.Count == 0 ? 1 : devices.Count;

            var progressStats = "Progress: ";
            var progressStatsLength = progressStats.Length;
            var textRow = 17 + listedDevices; // What row in the console window to draw progress
            var progress = 0;

            Console.WriteLine();
            Console.WriteLine("使用 YOLOv11 在视频上运行对象检测");
            Console.WriteLine(new string('=', 80));
            Console.Write(progressStats);

            // 为传入的视频帧分配处理器。
            // 💡 此 Action *每处理一帧调用一次*。
            // 它提供：
            //   - `frame`：当前视频帧作为 SKBitmap。
            //   - `frameIndex`：序列中帧的从零开始的索引。
            //
            // 您可以分配方法或 lambda 表达式。
            yolo.OnVideoFrameReceived = (SKBitmap frame, long frameIndex) =>
            {
                // 💡 在当前帧上运行对象检测。
                // 参数：
                //   - confidence：检测的最小置信度阈值 (0.0 - 1.0)。
                //   - iou：非极大值抑制的交并比阈值。
                //
                // 这将返回一个带有边界框和分数的检测对象列表。
                var result = yolo.RunObjectDetection(frame, confidence: 0.25, iou: 0.5)

                    // 💡 （可选）过滤结果以仅包含指定的类别标签。
                    // 在本例中：仅保留 "person" 的检测结果。
                    .FilterLabels(["person", "cat", "dog"])

                    // 💡 （可选）应用对象跟踪以在帧之间保持对象身份。
                    .Track(_sortTracker);

                // 💡 （可选）直接在当前帧上绘制检测和跟踪结果。
                // `_drawingOptions` controls appearance (e.g., color, thickness, font).
                // 如果未提供，将应用默认绘制设置。
                frame.Draw(result, _drawingOptions);

                // 如需要，在此添加额外的处理逻辑...

                // 💡 （可选）将处理后的帧保存为图像文件。
                // 用于调试、审计或生成图像数据集。
                // 示例：
                // var framePath = Path.Combine(_outputFolder, $"frame_{frameIndex}.jpg");
                // frame.Save(framePath, SKEncodedImageFormat.Jpeg, 80);

                // 显示进度。
                progress = (int)((double)(frameIndex) / metadata.TargetTotalFrames * 100);
                var str = $"{progress}% [frame {frameIndex} of {metadata.TargetTotalFrames}]";

                Console.SetCursorPosition(progressStatsLength, textRow);
                Console.Write(new string(' ', str.Length));
                Console.SetCursorPosition(progressStatsLength, textRow);
                Console.Write(str);
            };

            // 为视频处理结束时分配处理器。
            // 💡 此 Action *仅调用一次* 在视频处理结束时。
            // 它用于清理、报告、记录或触发下游操作。
            //
            // 您可以分配方法或 lambda 表达式。
            yolo.OnVideoEnd = () =>
            {
                Console.WriteLine();
                Console.WriteLine();

                if (progress == 100)
                {
                    Console.WriteLine("视频处理成功完成。");
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("警告：视频处理未成功完成。");
                }

                Console.ForegroundColor = ConsoleColor.Gray;
            };

            // 开始处理视频流。
            yolo.StartVideoProcessing();

            DisplayOutputFolder();

            Console.CursorVisible = true;
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

                // 用于绘制跟踪尾迹的属性。默认值：
                DrawTrackedTail = true,
                TailPaintColorStart = new SKColor(255, 105, 180),   // #FF69B4 - 耀眼的泡泡糖轰炸机粉色
                TailPaintColorEnd = SKColor.Empty.WithAlpha(0),              // 尾迹淡化结束。
                TailThickness = 4,
            };
        }

        private static void PrintMetaData(VideoMetadata metaData)
        {
            Console.WriteLine();
            Console.WriteLine($"视频元数据：");
            Console.WriteLine(new string('=', 80));

            Console.ForegroundColor = ConsoleColor.Blue;
            Console.WriteLine($"width           : {metaData.Width}");
            Console.WriteLine($"height          : {metaData.Height}");
            Console.WriteLine($"fps             : {metaData.FPS}");
            Console.WriteLine($"duration        : {metaData.Duration}");
            Console.WriteLine();
            Console.WriteLine($"target width    : {metaData.TargetWidth}");
            Console.WriteLine($"target height   : {metaData.TargetHeight}");
            Console.WriteLine($"target fps      : {metaData.FPS}");

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
