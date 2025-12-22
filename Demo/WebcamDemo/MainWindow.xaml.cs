// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2024-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using OpenCvSharp;
using SkiaSharp;
using SkiaSharp.Views.Desktop;
using System.Diagnostics;
using System.Windows;
using System.Windows.Threading;
using YoloDotNet;
using YoloDotNet.Enums;
using YoloDotNet.ExecutionProvider.Cuda;
using YoloDotNet.Extensions;
using YoloDotNet.Models;
using YoloDotNet.Test.Common;
using YoloDotNet.Trackers;
using Window = System.Windows.Window;

namespace WebcamDemo
{
    /// <summary>
    /// 使用 YoloDotNet 和 EmguCV 演示从网络摄像头进行实时对象检测和跟踪。
    /// 
    /// 此演示直接从网络摄像头捕获帧，使用 YOLO 模型执行对象检测，
    /// 并可选地应用多对象跟踪（SORT）。检测的对象绘制在帧上，
    /// 包括边界框、标签、置信度分数和跟踪轨迹。
    /// 
    /// 它展示了：
    /// - 可配置硬件加速（CUDA，可选 TensorRT 集成）和预处理选项的模型初始化
    /// - 使用 YoloDotNet 在实时网络摄像头输入上进行对象检测
    /// - 可选的类别标签过滤（例如，仅检测人员）
    /// - 使用 SORT 跟踪器在帧间进行可选的多对象跟踪
    /// - 在实时视频源上直接渲染检测和跟踪结果
    /// - 帧处理时间报告以进行性能监控
    /// 
    /// 网络摄像头源示例：
    /// - 默认网络摄像头设备（索引 0）
    /// - 按索引的附加设备（例如，1 表示辅助摄像头）
    /// 
    /// 执行提供程序：
    /// - CpuExecutionProvider：在 CPU 上运行推理，通用支持但速度较慢。
    /// - CudaExecutionProvider：使用 CUDA 在 NVIDIA GPU 上执行推理以获得加速性能。
    ///   可选地与 TensorRT 集成以进一步优化，支持 FP32、FP16 和 INT8 精度模式。
    ///   这在兼容的 GPU 上提供显著的速度改进。
    /// 
    /// 重要说明：
    /// - 根据硬件和性能要求选择执行提供程序。
    /// - 如果使用启用 TensorRT 的 CUDA，请确保您的环境具有兼容的 CUDA、cuDNN 和 TensorRT 设置。
    /// - 有关详细的设置说明和示例，请参见 README：
    ///   https://github.com/NickSwardh/YoloDotNet
    /// </summary>
    public partial class MainWindow : Window
    {
        #region Fields

        private readonly Yolo _yolo = default!;
        private readonly SortTracker _sortTracker = default!;
        private SKBitmap _currentFrame = default!;
        private Dispatcher _dispatcher = default!;

        private bool _runDetection = false;
        private SKRect _rect;
        private Stopwatch _stopwatch = default!;

        private SKImageInfo _imageInfo = default!;
        private bool _isTrackingEnabled;
        private bool _isFilteringEnabled;
        private double _confidenceThreshold;

        #endregion

        #region Constants

        const int WEBCAM_WIDTH = 1280;
        const int WEBCAM_HEIGHT = 720;
        const int FPS = 30;

        #endregion

        public MainWindow()
        {
            InitializeComponent();

            // 初始化秒表，用于简单测量帧处理时间
            _stopwatch = new Stopwatch();

            // （可选）使用可配置参数创建新的 SortTracker 实例：
            // - costThreshold：将检测分配到轨道的匹配成本阈值（越低 = 匹配越严格）。
            // - maxAge：在删除前保持未匹配轨道的帧数。
            // - tailLength：用于可视化或分析的轨道历史长度。
            // 注意：没有一刀切的设置；这些参数通常需要一些调整才能为您的特定用例找到最佳平衡。
            _sortTracker = new SortTracker(costThreshold: 0.5f, maxAge: 5, tailLength: 30);

            // 初始化 YoloDotNet。
            // YoloOptions 配置模型、硬件设置和图像处理行为。
            _yolo = new Yolo(new YoloOptions
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

                ExecutionProvider = new CudaExecutionProvider(

                    // 要加载的 ONNX 模型的路径或字节数组。
                    model: SharedConfig.GetTestModelV11(ModelType.ObjectDetection),

                    // 用于推理的 GPU 设备 ID。-1 = CPU，0+ = GPU 设备 ID。
                    gpuId: 0),

                // 推理前应用的调整大小模式。Proportional 保持宽高比（如果需要则添加填充），
                // 而 Stretch 调整图像大小以适应目标大小而不保持宽高比。
                // 相应地设置此选项，因为它直接影响推理结果。
                ImageResize = ImageResize.Proportional,

                // 调整大小的采样选项；影响推理速度和质量。
                // 其他采样选项的示例，请参见基准测试：https://github.com/NickSwardh/YoloDotNet/tree/master/test/YoloDotNet.Benchmarks
                SamplingOptions = new(SKFilterMode.Nearest, SKMipmapMode.None) // YoloDotNet 默认值
            });

            _dispatcher = Dispatcher.CurrentDispatcher;

            _currentFrame = new SKBitmap(WEBCAM_WIDTH, WEBCAM_HEIGHT);
            _rect = new SKRect(0, 0, WEBCAM_WIDTH, bottom: WEBCAM_HEIGHT);
            _imageInfo = new SKImageInfo(WEBCAM_WIDTH, WEBCAM_HEIGHT, SKColorType.Bgra8888, SKAlphaType.Premul);

            // 在后台线程上启动网络摄像头捕获
            Task.Run(() => WebcamAsync());
        }

        private async Task WebcamAsync()
        {
            // 初始化网络摄像头
            using var capture = new VideoCapture(0, VideoCaptureAPIs.DSHOW);

            capture.Set(VideoCaptureProperties.Fps, FPS);
            capture.Set(VideoCaptureProperties.FrameWidth, WEBCAM_WIDTH);
            capture.Set(VideoCaptureProperties.FrameHeight, WEBCAM_HEIGHT);

            // 如果摄像头支持 MJPEG，它在 CPU 上比未压缩的帧更便宜。
            capture.Set(VideoCaptureProperties.FourCC, VideoWriter.FourCC('M', 'J', 'P', 'G'));

            using var mat = new Mat();
            using var bgraMat = new Mat();

            while (true)
            {
                // 从网络摄像头捕获当前帧
                capture.Read(mat);
                
                // 将帧转换为 BGRA 颜色空间
                Cv2.CvtColor(mat, bgraMat, ColorConversionCodes.BGR2BGRA);

                // 从 BGRA Mat 创建用于处理的 SKBitmap
                using var frame = SKImage.FromPixels(_imageInfo, bgraMat.Data);

                _currentFrame?.Dispose();
                _currentFrame = SKBitmap.FromImage(frame);

                if (_runDetection)
                {
                    _stopwatch.Restart();

                    // 在当前帧上运行对象检测
                    var results = _yolo.RunObjectDetection(_currentFrame, _confidenceThreshold, iou: 0.7);

                    _stopwatch.Stop();

                    if (_isFilteringEnabled)
                        results = results.FilterLabels(["person", "cat", "dog"]);  // （可选）过滤结果以仅包含特定的类别（例如，"person", "cat", "dog"）

                    if (_isTrackingEnabled)
                        results.Track(_sortTracker); // （可选）使用 SortTracker 跟踪对象

                    // 在当前帧上绘制检测和跟踪结果
                    _currentFrame.Draw(results);
                }
               
                // 更新 GUI
                await _dispatcher.InvokeAsync(() =>
                {
                    WebCamFrame.InvalidateVisual(); // 通知 SKiaSharp 更新帧。

                    // 显示处理时间和最大 fps
                    if (_runDetection)
                    {
                        var milliseconds = _stopwatch.Elapsed.TotalMilliseconds;
                        var yoloFps = 1000.0 / milliseconds;

                        FrameProcess.Text = $"Processed Frame: {milliseconds:F1}ms ({yoloFps:F1} fps)";
                    }
                });
            }
        }

        private void UpdateWebcamFrame(object sender, SKPaintSurfaceEventArgs e)
        {
            using var canvas = e.Surface.Canvas;
            canvas.DrawBitmap(_currentFrame, _rect);
            //canvas.Flush();
        }

        private void StartClick(object sender, RoutedEventArgs e)
            => _runDetection = true;

        private void StopClick(object sender, RoutedEventArgs e)
            => _runDetection = false;

        private void EnableFiltering_Checked(object sender, RoutedEventArgs e)
            => _isFilteringEnabled = true;

        private void EnableFiltering_Unchecked(object sender, RoutedEventArgs e)
            => _isFilteringEnabled = false;

        private void EnableTracking_Checked(object sender, RoutedEventArgs e)
            => _isTrackingEnabled = true;

        private void EnableTracking_Unchecked(object sender, RoutedEventArgs e)
            => _isTrackingEnabled = false;

        private void WindowClosing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            _runDetection = false;
            _yolo?.Dispose();
            _currentFrame?.Dispose();
        }

        private void ConfidenceTreshold_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            _confidenceThreshold = e.NewValue;
        }
    }
}