// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Core
{
    /// <summary>
    /// 初始化 Yolo 核心类的新实例。
    /// </summary>
    internal class YoloCore(YoloOptions yoloOptions) : IDisposable
    {
        #region Fields
        private bool _isDisposed;
        private PinnedMemoryBufferPool _pinnedMemoryPool = default!;
        private readonly object _progressLock = new();
        private SKImageInfo _imageInfo;
        private long[] _inputShape = default!;
        private int _inputShapeSize;
        #endregion

        public OnnxModel OnnxModel { get; private set; } = default!;
        public YoloOptions YoloOptions { get => yoloOptions; init => yoloOptions = value; }
        public ModelType ModelType => OnnxModel.ModelType;

        /// <summary>
        /// 使用指定的模型类型初始化 YOLO 模型。
        /// </summary>
        public void InitializeYolo()
        {
            if (YoloOptions.ExecutionProvider is null)
                throw new YoloDotNetModelException("Execution Provider is missing. Please add an execution provider.", nameof(YoloOptions));

            OnnxModel = yoloOptions.ExecutionProvider.OnnxData.GetOnnxProperties();

            VerifyExpectedModelType(OnnxModel.ModelType);

            _inputShape = OnnxModel.InputShape;
            _inputShapeSize = OnnxModel.InputShapeSize;

            var format = (OnnxModel.Input.Channels == 1) ? SKColorType.Gray8 : SKColorType.Rgb888x;
            _imageInfo = new SKImageInfo(OnnxModel.Input.Width, OnnxModel.Input.Height, format, SKAlphaType.Opaque);

            _pinnedMemoryPool = new PinnedMemoryBufferPool(_imageInfo);

            FrameSaveService.Start();
        }

        /// <summary>
        /// 在提供的图像上运行 YOLO 模型并返回推理结果。
        /// </summary>
        /// <param name="image">要处理的输入图像。</param>
        /// <returns>InferenceResult()</returns>
        public InferenceResult Run<T>(T image)
        {
            lock (_progressLock)
            {
                var pinnedBuffer = _pinnedMemoryPool.Rent();

                var normalizedPixelsFloatBuffer = ArrayPool<float>.Shared.Rent(_inputShapeSize);
                var normalizedPixelsUshortBuffer = ArrayPool<ushort>.Shared.Rent(_inputShapeSize);

                try
                {
                    // 将图像调整大小为模型输入大小并存储在固定缓冲区中以加快访问速度
                    var originalImageSize = YoloOptions.ImageResize == ImageResize.Proportional
                        ? image.ResizeImageProportional(YoloOptions.SamplingOptions, pinnedBuffer)
                        : image.ResizeImageStretched(YoloOptions.SamplingOptions, pinnedBuffer);

                    // 声明结构体来保存推理结果
                    InferenceResult inferenceResult;

                    // 使用选定的执行提供程序和模型数据类型运行推理
                    if (OnnxModel.ModelDataType == ModelDataType.Float16)
                    {
                        pinnedBuffer.Pointer.NormalizePixelsToArray(_inputShape, _inputShapeSize, normalizedPixelsUshortBuffer);
                        inferenceResult = YoloOptions.ExecutionProvider.Run<ushort>(normalizedPixelsUshortBuffer);
                    }
                    else
                    {
                        pinnedBuffer.Pointer.NormalizePixelsToArray(_inputShape, _inputShapeSize, normalizedPixelsFloatBuffer);
                        inferenceResult = YoloOptions.ExecutionProvider.Run<float>(normalizedPixelsFloatBuffer);
                    }

                    // 将原始图像大小附加到推理结果，用于计算到原始图像大小的边界框。
                    inferenceResult.ImageOriginalSize = originalImageSize;

                    return inferenceResult;
                }
                finally
                {
                    // 为了性能，将租用的缓冲区返回到各自的池中而不清除。
                    // 警告：仅在下一次使用时数据被覆盖的情况下才禁用清除，以避免数据泄漏！
                    ArrayPool<float>.Shared.Return(normalizedPixelsFloatBuffer, false);
                    ArrayPool<ushort>.Shared.Return(normalizedPixelsUshortBuffer, false);

                    _pinnedMemoryPool.Return(pinnedBuffer);
                }
            }
        }

        #region Helper methods

        /// <summary>
        /// 移除对象检测结果列表中的重叠边界框。
        /// </summary>
        /// <param name="predictionSpan">包含预测结果的 Span</param>
        /// <param name="iouThreshold">更高的 IoU 阈值通过排除重叠框导致更少的检测。</param>
        /// <returns>基于置信度分数过滤的非重叠边界框的 Span<ObjectResult>。</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<ObjectResult> RemoveOverlappingBoxes(Span<ObjectResult> predictionSpan, double iouThreshold)
        {
            var totalPredictions = predictionSpan.Length;

            if (totalPredictions == 0)
                return [];

            // 按置信度排序
            MemoryExtensions.Sort(predictionSpan, ConfidenceComparer.Instance);

            var buffer = ArrayPool<ObjectResult>.Shared.Rent(totalPredictions);

            try
            {
                var counter = 0;
                for (int i = 0; i < totalPredictions; i++)
                {
                    var item = predictionSpan[i];

                    bool overlapFound = false;
                    for (int j = 0; j < counter; j++)
                    {
                        if (CalculateIoU(item.BoundingBox, buffer[j].BoundingBox) > iouThreshold)
                        {
                            overlapFound = true;
                            break;
                        }
                    }

                    if (!overlapFound)
                        buffer[counter++] = item;
                }
                return buffer.AsSpan(0, counter);
            }
            finally
            {
                ArrayPool<ObjectResult>.Shared.Return(buffer, false);
            }
        }

        /// <summary>
        /// 将缓冲池大小计算为2的次幂以确保数组池效率。
        /// </summary>
        public static int CalculateBufferPoolSize(int bufferSize) => 1 << (int)Math.Ceiling(Math.Log2(bufferSize));

        /// <summary>
        /// 将值压缩到0和1之间的数字
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sigmoid(float value) => 1 / (1 + MathF.Exp(-value));

        /// <summary>
        /// 计算像素亮度
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte CalculatePixelLuminance(float value) => (byte)(255 - value * 255);

        /// <summary>
        /// 按字节计算像素置信度
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CalculatePixelConfidence(byte value) => value / 255F;

        /// <summary>
        /// 计算弧度到角度
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CalculateRadianToDegree(float value) => value * (180 / (float)Math.PI);

        /// <summary>
        /// 计算两个矩形之间的交并比 (IoU)。
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float CalculateIoU(in SKRectI a, in SKRectI b)
        {
            // 在参数上使用 "in" 关键字以通过引用传递而不允许修改，以获得更好的性能。

            int left = Math.Max(a.Left, b.Left);
            int top = Math.Max(a.Top, b.Top);
            int right = Math.Min(a.Right, b.Right);
            int bottom = Math.Min(a.Bottom, b.Bottom);

            int width = right - left;
            int height = bottom - top;

            if (width <= 0 || height <= 0)
                return 0f;

            int intersection = width * height;

            int areaA = (a.Right - a.Left) * (a.Bottom - a.Top);
            int areaB = (b.Right - b.Left) * (b.Bottom - b.Top);

            return (float)intersection / (areaA + areaB - intersection);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public (float, float, float, float) CalculateGain(SKSizeI size)
            => YoloOptions.ImageResize == ImageResize.Proportional ? CalculateProportionalGain(size) : CalculateStretchedGain(size);

        /// <summary>
        /// 计算调整边界框所需的填充和缩放因子，
        /// 以便检测到的对象可以调整大小以匹配原始图像大小。
        /// </summary>
        /// <param name="size">原始图像大小。</param>
        public (float, float, float, float) CalculateProportionalGain(SKSizeI size)
        {
            var model = OnnxModel;

            var (w, h) = (size.Width, size.Height);

            var gain = Math.Max((float)w / model.Input.Width, (float)h / model.Input.Height);
            var ratio = Math.Min(model.Input.Width / (float)size.Width, model.Input.Height / (float)size.Height);
            var (xPad, yPad) = ((model.Input.Width - w * ratio) / 2, (model.Input.Height - h * ratio) / 2);

            return (xPad, yPad, gain, 0);
        }

        /// <summary>
        /// 计算调整边界框所需的填充和缩放因子，
        /// 以便检测到的对象可以调整大小以匹配原始图像大小。
        /// </summary>
        /// <param name="size">原始图像大小。</param>
        public (float, float, float, float) CalculateStretchedGain(SKSizeI size)
        {
            var model = OnnxModel;

            var (w, h) = (size.Width, size.Height); // 图像宽度和高度
            var (xGain, yGain) = (model.Input.Width / (float)w, model.Input.Height / (float)h); // x, y 增益
            var (xPad, yPad) = ((model.Input.Width - w * xGain) / 2, (model.Input.Height - h * yGain) / 2); // 左右填充

            return (xPad, yPad, xGain, yGain);
        }

        /// <summary>
        /// 验证加载的模型是否为预期类型
        /// </summary>
        public void VerifyExpectedModelType(ModelType expectedModelType)
        {
            if (expectedModelType.Equals(OnnxModel.ModelType) is false)
                throw new YoloDotNetModelMismatchException($"Loaded ONNX-model is of type {OnnxModel.ModelType} and can't be used for {expectedModelType}.");
        }

        /// <summary>
        /// 释放资源并禁止当前对象的终结器。
        /// </summary>
        public void Dispose()
        {
            if (_isDisposed)
                return;

            _pinnedMemoryPool?.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}
