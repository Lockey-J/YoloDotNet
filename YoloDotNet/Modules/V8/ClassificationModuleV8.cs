// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Modules.V8
{
    internal class ClassificationModuleV8 : IClassificationModule
    {
        private readonly YoloCore _yoloCore;

        public OnnxModel OnnxModel => _yoloCore.OnnxModel;
        private ArrayPool<ClassificationEntry> _classificationPool = default!;

        public ClassificationModuleV8(YoloCore yoloCore)
        {
            _yoloCore = yoloCore;

            var poolSize = YoloCore.CalculateBufferPoolSize(_yoloCore.OnnxModel.Labels.Length);
            _classificationPool = ArrayPool<ClassificationEntry>.Create(poolSize, 10);
        }

        public List<Classification> ProcessImage<T>(T image, double classes, double pixelConfidence, double iou)
        {
            var inferenceResult = _yoloCore.Run(image);

            return ClassifyTensor(inferenceResult.OrtSpan0, (int)classes);
        }

        #region Classification

        /// <summary>
        /// 对张量进行分类并返回 Classification 列表
        /// </summary>
        /// <param name="numberOfClasses">类别数量</param>
        private List<Classification> ClassifyTensor(ReadOnlySpan<float> span, int numberOfClasses)
        {
            var poolBuffer = _classificationPool.Rent(span.Length);

            try
            {
                // 用置信度和标签ID填充池缓冲区
                for (int i = 0; i < span.Length; i++)
                {
                    poolBuffer[i] = new ClassificationEntry(span[i], i);
                }

                // 按置信度降序排序
                Array.Sort(poolBuffer, (a, b) => b.Confidence.CompareTo(a.Confidence));

                // 根据 numberOfClasses 取前 N 个类别
                var results = new List<Classification>(numberOfClasses);
                for (int i = 0; i < numberOfClasses; i++)
                {
                    var entry = poolBuffer[i];
                    results.Add(new Classification
                    {
                        Confidence = entry.Confidence,
                        Label = _yoloCore.OnnxModel.Labels[entry.LabelId].Name
                    });
                }

                return results;
            }
            finally
            {
                _classificationPool.Return(array: poolBuffer, clearArray: true);
            }
        }

        #endregion

        #region Helper methods

        public void Dispose()
        {
            _yoloCore.Dispose();

            GC.SuppressFinalize(this);
        }

        #endregion
    }

    internal readonly struct ClassificationEntry(float confidence, int labelId)
    {
        public readonly float Confidence = confidence;  // 置信度分数
        public readonly int LabelId = labelId;          // 标签数组的索引
    }
}
