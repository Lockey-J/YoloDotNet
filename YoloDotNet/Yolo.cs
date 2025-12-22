// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet
{
    /// <summary>
    /// 初始化 YoloDotNet 的新实例。
    /// </summary>
    /// <param name="options">初始化 YoloDotNet 模型的选项。</param>
    public class Yolo(YoloOptions options) : IDisposable
    {
        #region Private fields

        private readonly IModule _detection = ModuleFactory.CreateModule(options);
        private FFmpegService _ffmpegService = default!;

        #endregion

        #region Public Fields

        public OnnxModel OnnxModel => _detection.OnnxModel;
        public Action<SKBitmap, long> OnVideoFrameReceived = default!;
        public Action OnVideoEnd = default!;

        #endregion

        #region Classification

        /// <summary>
        /// 在图像上运行图像分类。
        /// </summary>
        /// <param name="img">要分类的 SKBitmap。</param>
        /// <param name="classes">要返回的类别数量（默认为 1）。</param>
        /// <returns>分类结果列表。</returns>
        public List<Classification> RunClassification(SKBitmap img, int classes = 1)
            => ((IClassificationModule)_detection).ProcessImage(img, classes, 0, 0);

        /// <summary>
        /// 在图像上运行图像分类。
        /// </summary>
        /// <param name="img">要分类的 SKImage。</param>
        /// <param name="classes">要返回的类别数量（默认为 1）。</param>
        /// <returns>分类结果列表。</returns>
        public List<Classification> RunClassification(SKImage img, int classes = 1)
            => ((IClassificationModule)_detection).ProcessImage(img, classes, 0, 0);

        #endregion

        #region Object Detection

        /// <summary>
        /// 在图像上运行对象检测。
        /// </summary>
        /// <param name="img">要进行 OBB 检测的 SKBitmap。</param>
        /// <param name="confidence">检测对象的置信度阈值（默认为 0.2）。</param>
        /// <param name="iou">用于移除重叠边界框的 IoU（交并比）重叠阈值（默认：0.7）。</param>
        /// <returns>分类结果列表。</returns>
        public List<ObjectDetection> RunObjectDetection(SKBitmap img, double confidence = 0.2, double iou = 0.7)
            => ((IObjectDetectionModule)_detection).ProcessImage(img, confidence, 0, iou);

        /// <summary>
        /// 在图像上运行对象检测。
        /// </summary>
        /// <param name="img">要进行 OBB 检测的 SKImage。</param>
        /// <param name="confidence">检测对象的置信度阈值（默认为 0.2）。</param>
        /// <param name="iou">用于移除重叠边界框的 IoU（交并比）重叠阈值（默认：0.7）。</param>
        /// <returns>分类结果列表。</returns>
        public List<ObjectDetection> RunObjectDetection(SKImage img, double confidence = 0.2, double iou = 0.7)
             => ((IObjectDetectionModule)_detection).ProcessImage(img, confidence, 0, iou);

        #endregion

        #region OBB (Oriented Bounding Box)

        /// <summary>
        /// 在图像上运行定向边界框检测。
        /// </summary>
        /// <param name="img">要进行 OBB 检测的 SKBitmap。</param>
        /// <param name="confidence">检测对象的置信度阈值（默认为 0.2）。</param>
        /// <param name="iou">用于移除重叠边界框的 IoU（交并比）重叠阈值（默认：0.7）。</param>
        /// <returns>分割结果列表。</returns>
        public List<OBBDetection> RunObbDetection(SKBitmap img, double confidence = 0.2, double iou = 0.7)
            => ((IOBBDetectionModule)_detection).ProcessImage(img, confidence, 0, iou);

        /// <summary>
        /// 在图像上运行定向边界框检测。
        /// </summary>
        /// <param name="img">要进行 OBB 检测的 SKImage。</param>
        /// <param name="confidence">检测对象的置信度阈值（默认为 0.2）。</param>
        /// <param name="iou">用于移除重叠边界框的 IoU（交并比）重叠阈值（默认：0.7）。</param>
        /// <returns>分割结果列表。</returns>
        public List<OBBDetection> RunObbDetection(SKImage img, double confidence = 0.2, double iou = 0.7)
            => ((IOBBDetectionModule)_detection).ProcessImage(img, confidence, 0, iou);

        #endregion

        #region Segmentation

        /// <summary>
        /// 在图像上运行分割。
        /// </summary>
        /// <param name="img">要分割的 SKBitmap。</param>
        /// <param name="confidence">检测对象的置信度阈值（默认为 0.2）。</param>
        /// <param name="iou">用于移除重叠边界框的 IoU（交并比）重叠阈值（默认：0.7）。</param>
        /// <returns>分割结果列表。</returns>
        public List<Segmentation> RunSegmentation(SKBitmap img, double confidence = 0.2, double pixelConfedence = 0.65, double iou = 0.7)
            => ((ISegmentationModule)_detection).ProcessImage(img, confidence, pixelConfedence, iou);

        /// <summary>
        /// 在图像上运行分割。
        /// </summary>
        /// <param name="img">要分割的 SKImage。</param>
        /// <param name="confidence">检测对象的置信度阈值（默认为 0.2）。</param>
        /// <param name="iou">用于移除重叠边界框的 IoU（交并比）重叠阈值（默认：0.7）。</param>
        /// <returns>分割结果列表。</returns>
        public List<Segmentation> RunSegmentation(SKImage img, double confidence = 0.2, double pixelConfedence = 0.65, double iou = 0.7)
            => ((ISegmentationModule)_detection).ProcessImage(img, confidence, pixelConfedence, iou);

        #endregion

        #region Pose Estimation

        /// <summary>
        /// 在图像上运行姿态估计。
        /// </summary>
        /// <param name="img">要进行姿态估计的 SKBitmap。</param>
        /// <param name="confidence">检测对象的置信度阈值（默认为 0.2）。</param>
        /// <param name="iou">用于移除重叠边界框的 IoU（交并比）重叠阈值（默认：0.7）。</param>
        /// <returns>分割结果列表。</returns>
        public List<PoseEstimation> RunPoseEstimation(SKBitmap img, double confidence = 0.2, double iou = 0.7)
            => ((IPoseEstimationModule)_detection).ProcessImage(img, confidence, 0, iou);

        /// <summary>
        /// 在图像上运行姿态估计。
        /// </summary>
        /// <param name="img">要进行姿态估计的 SKImage。</param>
        /// <param name="confidence">检测对象的置信度阈值（默认为 0.2）。</param>
        /// <param name="iou">用于移除重叠边界框的 IoU（交并比）重叠阈值（默认：0.7）。</param>
        /// <returns>分割结果列表。</returns>
        public List<PoseEstimation> RunPoseEstimation(SKImage img, double confidence = 0.2, double iou = 0.7)
            => ((IPoseEstimationModule)_detection).ProcessImage(img, confidence, 0, iou);

        #endregion

        #region Video

        /// <summary>
        /// 使用指定的 <see cref="VideoOptions"/> 初始化视频流，并设置帧处理和视频完成的事件处理程序。
        /// </summary>
        /// <param name="videoOptions"></param>
        public void InitializeVideo(VideoOptions videoOptions)
        {
            _ffmpegService = new(videoOptions, options)
            {
                OnFrameReady = (frame, frameIndex) => OnVideoFrameReceived.Invoke(frame, frameIndex),
                OnVideoEnd = () => OnVideoEnd?.Invoke()
            };
        }

        /// <summary>
        /// 获取当前系统上检测到的可用视频输入设备列表。
        /// </summary>
        /// <exception cref="YoloDotNetVideoException"></exception>
        public static List<string> GetVideoDevices()
            => FFmpegService.GetVideoDevicesOnSystem() ?? throw new YoloDotNetVideoException(
                "没有初始化视频。在尝试检索元数据之前请调用 InitializeVideo()。");

        /// <summary>
        /// 检索有关流或初始化视频的元数据，如持续时间、帧率和分辨率。
        /// </summary>
        /// <exception cref="YoloDotNetVideoException"></exception>
        public VideoMetadata GetVideoMetaData()
            => _ffmpegService.VideoMetadata ?? throw new YoloDotNetVideoException(
                "没有初始化视频。在尝试检索元数据之前请调用 InitializeVideo()。");

        /// <summary>
        /// 开始解码和处理来自初始化视频流的视频帧。
        /// </summary>
        public void StartVideoProcessing()
            => _ffmpegService.Start();

        /// <summary>
        /// 停止视频帧处理并释放与视频流关联的资源。
        /// </summary>
        public void StopVideoProcessing()
            => _ffmpegService.Stop();

        #endregion

        #region Model Info

        /// <summary>
        /// 获取当前加载的 YOLO 模型的描述，
        /// 包括模型类型和版本。如果没有模型被初始化，
        /// 则返回 "No model loaded"。
        /// </summary>
        public string ModelInfo =>
            _detection.OnnxModel == null
                ? "没有加载模型"
                : $"{_detection.OnnxModel.ModelType} (yolo {_detection.OnnxModel.ModelVersion.ToString().ToLower()})";

        #endregion

        #region Dispose

        public void Dispose()
        {
            _detection.Dispose();
            _ffmpegService?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}