// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V8
{
    internal class ObjectDetectionModuleV8 : IObjectDetectionModule
    {
        private readonly YoloCore _yoloCore;
        private readonly int _labels;
        private readonly int _channels;
        private readonly int _channels2;
        private readonly int _channels3;
        private readonly int _channels4;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;

        public ObjectDetectionModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            _labels = _yoloCore.OnnxModel.Labels.Length;
            _channels = _yoloCore.OnnxModel.Outputs[0].Channels;
            _channels2 = _channels * 2;
            _channels3 = _channels * 3;
            _channels4 = _channels * 4;
        }

        public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
        {
            var result = _yoloCore.Run(image);
            var detections = ObjectDetection(result, confidence, iou);

            // 转换为 List<ObjectDetection>
            var results = new List<ObjectDetection>(detections.Length);
            for (int i = 0; i < detections.Length; i++)
                results.Add((ObjectDetection)detections[i]);

            return results;
        }

        #region Helper methods

        /// <summary>
        /// 在张量中检测对象并返回 ObjectDetection 列表。
        /// </summary>
        /// <param name="imageSize">与张量数据关联的图像。</param>
        /// <param name="confidenceThreshold">接受对象检测的置信度阈值。</param>
        /// <param name="overlapThreshold">过滤检测的重叠框的阈值。</param>
        /// <returns>表示检测到对象的结果模型列表。</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] // 内联此小方法以获得更好的性能
        public Span<ObjectResult> ObjectDetection(InferenceResult inferenceResult, double confidenceThreshold, double overlapThreshold)
        {
            var imageSize = inferenceResult.ImageOriginalSize;
            var ortSpan = inferenceResult.OrtSpan0;

            if (ortSpan == null)
                return [];

            var (xPad, yPad, xGain, yGain) = _yoloCore.CalculateGain(imageSize);

            var width = imageSize.Width;
            var height = imageSize.Height;

            int validBoxCount = 0;
            //var boxes = _yoloCore.customSizeObjectResultPool.Rent(_channels);
            var boxes = ArrayPool<ObjectResult>.Shared.Rent(_channels);

            try
            {
                for (int i = 0; i < _channels; i++)
                {
                    // 前进到第一个标签的置信度值
                    var labelOffset = i + _channels4;

                    float bestConfidence = 0f;
                    int bestLabelIndex = -1;

                    // 获取当前边界框的置信度和标签
                    for (var l = 0; l < _labels; l++, labelOffset += _channels)
                    {
                        var boxConfidence = ortSpan[labelOffset];

                        if (boxConfidence > bestConfidence)
                        {
                            bestConfidence = boxConfidence;
                            bestLabelIndex = l;
                        }
                    }

                    // 如果置信度低，提前停止
                    if (bestConfidence < confidenceThreshold)
                        continue;

                    float x = ortSpan[i];
                    float y = ortSpan[i + _channels];
                    float w = ortSpan[i + _channels2];
                    float h = ortSpan[i + _channels3];

                    var (xMin, yMin, xMax, yMax) = (0, 0, 0, 0);

                    // 边界框计算基于输入图像的调整大小方式
                    // 'Proportional' 保持原始纵横比并在图像周围添加填充
                    // 'Stretched' 缩放图像以填充尺寸，可能会扭曲纵横比
                    if (_yoloCore.YoloOptions.ImageResize == ImageResize.Proportional)
                    {
                        var gain = xGain; // 比例调整大小的缩放因子

                        xMin = (int)((x - w / 2 - xPad) * gain);
                        yMin = (int)((y - h / 2 - yPad) * gain);
                        xMax = (int)((x + w / 2 - xPad) * gain);
                        yMax = (int)((y + h / 2 - yPad) * gain);
                    }
                    else
                    {
                        var halfW = w / 2;
                        var halfH = h / 2;

                        // 计算调整拉伸缩放和填充的边界框坐标
                        // Clamp 确保坐标保持在图像的有效边界内。
                        //xMin = Math.Clamp((int)((x - halfW - xPad) / xGain), 0, width - 1);
                        //yMin = Math.Clamp((int)((y - halfH - yPad) / yGain), 0, height - 1);
                        //xMax = Math.Clamp((int)((x + halfW - xPad) / xGain), 0, width - 1);
                        //yMax = Math.Clamp((int)((y + halfH - yPad) / yGain), 0, height - 1);

                        int val = (int)((x - halfW - xPad) / xGain);
                        xMin = val < 0 ? 0 : (val > width - 1 ? width - 1 : val);

                        int valY = (int)((y - halfH - yPad) / yGain);
                        yMin = valY < 0 ? 0 : (valY > height - 1 ? height - 1 : valY);

                        int valXMax = (int)((x + halfW - xPad) / xGain);
                        xMax = valXMax < 0 ? 0 : (valXMax > width - 1 ? width - 1 : valXMax);

                        int valYMax = (int)((y + halfH - yPad) / yGain);
                        yMax = valYMax < 0 ? 0 : (valYMax > height - 1 ? height - 1 : valYMax);
                    }

                    // 调整后输入图像的未缩放坐标
                    var sxMin = (int)(x - w / 2);
                    var syMin = (int)(y - h / 2);
                    var sxMax = (int)(x + w / 2);
                    var syMax = (int)(y + h / 2);

                    var boundingBox = new SKRectI(xMin, yMin, xMax, yMax);
                    var boundingBoxUnscaled = new SKRectI(sxMin, syMin, sxMax, syMax);

                    boxes[validBoxCount++] = new ObjectResult
                    {
                        Label = _yoloCore.OnnxModel.Labels[bestLabelIndex],
                        Confidence = bestConfidence,
                        BoundingBox = boundingBox,
                        BoundingBoxUnscaled = boundingBoxUnscaled,
                        BoundingBoxIndex = i,
                        OrientationAngle = _yoloCore.OnnxModel.ModelType == ModelType.ObbDetection ? ortSpan[i + _channels * (4 + _labels)] : 0
                    };
                }

                // 获取租用数组的有效部分
                var resultArray = boxes.AsSpan(0, validBoxCount);

                // 使用非极大值抑制(NMS)移除重叠框
                return _yoloCore.RemoveOverlappingBoxes(resultArray, overlapThreshold);
            }
            finally
            {
                //_yoloCore.customSizeObjectResultPool.Return(boxes, clearArray: false);
                ArrayPool<ObjectResult>.Shared.Return(boxes, false);
            }
        }

        public void Dispose()
        {
            _yoloCore?.Dispose();
            GC.SuppressFinalize(this);
        }

        #endregion
    }
}