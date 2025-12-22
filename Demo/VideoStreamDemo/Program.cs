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
            // (Optional) Create a new SortTracker instance with configurable parameters:
            // - costThreshold: matching cost threshold for assigning detections to tracks (lower = stricter matching).
            // - maxAge: number of frames to keep unmatched tracks before removal.
            // - tailLength: length of the track history for visualization or analysis.
            // Note: There is no one-size-fits-all setting; these parameters often require some tinkering to find the best balance for your specific use case.
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
                //     Runs inference entirely on the CPU. Universally supported on all hardware.
                //
                //   - CudaExecutionProvider
                //     Executes inference on an NVIDIA GPU using CUDA for accelerated performance.  
                //     Optionally integrates with TensorRT for further optimization, supporting FP32, FP16,  
                //     and INT8 precision modes. This delivers significant speed improvements on compatible GPUs.  
                //     See the TensorRT demo and documentation for detailed configuration and best practices.
                //
                //   - OpenVinoExecutionProvider
                //     Runs inference using Intel's OpenVINO toolkit for optimized performance on Intel hardware.
                //
                //   - CoreMLExecutionProvider
                //     Executes inference using Apple's CoreML framework for efficient performance on macOS and iOS devices.
                //
                //   Important:  
                //     - Choose the provider that matches your available hardware and performance requirements.  
                //     - If using CUDA with TensorRT enabled, ensure your environment has a compatible CUDA, cuDNN, and TensorRT setup.
                //     - For detailed setup instructions and examples, see the README:
                //
                //   More information about execution providers and setup instructions can be found in the README:
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
            Console.WriteLine("Detected video input devices (usable with VideoOptions):");

            if (devices.Count != 0)
            {
                foreach (string device in devices)
                    Console.WriteLine($"  {device}");
            }
            else
                Console.WriteLine("No input devices found");

            // 设置视频选项。
            yolo.InitializeVideo(new VideoOptions
            {
                // 💡 Input video source. Accepted formats:
                // 
                // 1. Local video file:
                //    Example: @"C:\videos\test.mp4"
                //
                // 2. Livestream URL (e.g., RTMP, HTTP):
                //    Example: "rtmp://your.rtmp.server/stream"
                //
                // 3. Video capture device (e.g., webcam):
                //    Format: "device=<DeviceName>:<Width>:<Height>:<FPS>"
                //
                //    ⮞ On Windows:
                //       Example: "device=Logitech BRIO:1920:1080:30"
                //
                //    ⮞ On Linux:
                //       Example: "device=/dev/video0:1280:720:30"
                //
                // 📌 The Width, Height, and FPS values must match a valid capture mode supported by your camera.
                //
                // 🔍 To discover available video devices:
                //    Use `yolo.GetVideoDevices()` — this method lists the names of available video capture devices.
                //    It does NOT list supported resolutions or framerates.
                //
                //    To determine valid width/height/fps combinations, refer to your device specifications
                VideoInput = SharedConfig.GetTestVideo(VideoType.PeopleWalking),

                // 💡 Optional: Path to save the processed output video file.
                // 如果您不想保存输出，请保持未设置（null 或空）。
                VideoOutput = Path.Combine(_outputFolder, "video_output.mp4"),

                // 💡 Encoder to use when writing output.
                // 注意：确保所选编码器与以下内容兼容：
                //   - 您的操作系统和硬件
                //   - FFmpeg 已编译所选编码器
                //   - 输出文件格式/容器
                VideoEncoder = VideoEncoder.H264Nvenc, // 使用 NVIDIA GPU 编码器 (h264_nvenc)。编码器替代方案请参见 Encoder 枚举注释。

                // 💡 Frame rate for the output video.
                // FrameRate.AUTO will attempt to match the input video’s frame rate.
                FrameRate = FrameRate.AUTO,

                // 💡 Output video width in pixels.
                // Leave unset (0) to use the original width.
                // Set to -2 to automatically calculate the width while maintaining the aspect ratio, based on the specified height.
                // Note: Only one of Width or Height can be set to -2 at a time.
                Width = 720,

                // 💡 Output video height in pixels.
                // Leave unset (0) to use the original height.
                // Set to -2 to automatically calculate the height while maintaining the aspect ratio, based on the specified width.
                // Note: Only one of Width or Height can be set to -2 at a time.
                Height = -2,

                // 💡 Compression quality for the output video (1-51).
                // Lower values = better quality, larger file size.
                // Higher values = stronger compression, smaller file size, lower quality.
                // Recommended range: 20-35 for reasonable balance.
                CompressionQuality = 30,

                // 💡 Optional: Automatically split output video into chunks.
                // Duration in seconds for each chunk.
                // Example: 600 = split into 10-minute segments.
                // 0 = do not split (generate a single file).
                VideoChunkDuration = 0,

                // 💡 Process every Nth frame.
                // 0 = process all frames (default).
                // Example: 30 = process every 30th frame (useful for surveillance where full-frame detection is unnecessary).
                FrameInterval = 0,

                // 💡 Optional: Define a specific segment of the video to process.
                // Useful for testing or processing only a portion of the video.
                // Note: Only applies to video files.
                // 0 = start from the beginning.
                StartTimeSeconds = 0,

                // 💡 Optional: Duration of the video segment to process in seconds.
                // Useful for testing or processing only a portion of the video.
                // Note: Only applies to video files.
                // 0 = process until the end of the video.
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
            Console.WriteLine("Running Object Detection on Video with YOLOv11");
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
                // 💡 Run object detection on the current frame.
                // 参数：
                //   - confidence：检测的最小置信度阈值 (0.0 - 1.0)。
                //   - iou：非极大值抑制的交并比阈值。
                //
                // 这将返回一个带有边界框和分数的检测对象列表。
                var result = yolo.RunObjectDetection(frame, confidence: 0.25, iou: 0.5)

                    // 💡 (Optional) Filter results to include only specified class labels.
                    // 在本例中：仅保留 "person" 的检测结果。
                    .FilterLabels(["person", "cat", "dog"])

                    // 💡 (Optional) Apply object tracking to maintain object identities across frames.
                    .Track(_sortTracker);

                // 💡 (Optional) Draw detection and tracking results directly onto the current frame.
                // `_drawingOptions` controls appearance (e.g., color, thickness, font).
                // 如果未提供，将应用默认绘制设置。
                frame.Draw(result, _drawingOptions);

                // 如需要，在此添加额外的处理逻辑...

                // 💡 (Optional) Save the processed frame as an image file.
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
                    Console.WriteLine("Video processing completed successfully.");
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Warning: Video processing did not complete successfully.");
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

                // Properties for drawing tracked tail. Default values:
                DrawTrackedTail = true,
                TailPaintColorStart = new SKColor(255, 105, 180),   // #FF69B4 - Blazing Bubblegum Bomber Pink
                TailPaintColorEnd = SKColor.Empty.WithAlpha(0),              // Fade end of tail.
                TailThickness = 4,
            };
        }

        private static void PrintMetaData(VideoMetadata metaData)
        {
            Console.WriteLine();
            Console.WriteLine($"Video MetaData:");
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
