// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    public class ObjectResult
    {
        /// <summary>
        /// 与检测到的对象关联的标签信息。
        /// </summary>
        public LabelModel Label { get; set; } = new();

        /// <summary>
        /// 检测到的对象的置信度分数。
        /// </summary>
        public double Confidence { get; set; }

        /// <summary>
        /// 检测到的对象的感兴趣区域（边界框）。
        /// </summary>
        public SKRectI BoundingBox { get; set; }

        /// <summary>
        /// 针对 ONNX 模型维度的检测对象的感兴趣区域（边界框）。
        /// </summary>
        public SKRect BoundingBoxUnscaled { get; set; }

        /// <summary>
        /// 边界框的索引
        /// </summary>
        public int BoundingBoxIndex { get; set; }

        /// <summary>
        /// 位压缩掩码，其中每个位表示置信度高于阈值的像素（1 = 存在，0 = 不存在）。
        /// </summary>
        public byte[] BitPackedPixelMask { get; set; } = [];

        /// <summary>
        /// 姿态估计关键点的置信度值、X 和 Y 坐标
        /// </summary>
        public KeyPoint[] KeyPoints { get; set; } = [];

        /// <summary>
        /// OBB 检测的边界框的方向角度。
        /// </summary>
        public float OrientationAngle { get; set; }

        #region 映射方法
        public static explicit operator ObjectDetection(ObjectResult result) => new()
        {
            Label = result.Label,
            Confidence = result.Confidence,
            BoundingBox = result.BoundingBox
        };

        public static explicit operator OBBDetection(ObjectResult result) => new()
        {
            Label = result.Label,
            Confidence = result.Confidence,
            BoundingBox = result.BoundingBox,
            OrientationAngle = result.OrientationAngle
        };

        public static explicit operator Segmentation(ObjectResult result) => new()
        {
            Label = result.Label,
            Confidence = result.Confidence,
            BoundingBox = result.BoundingBox,
            BitPackedPixelMask = result.BitPackedPixelMask
        };

        public static explicit operator PoseEstimation(ObjectResult result) => new()
        {
            Label = result.Label,
            Confidence = result.Confidence,
            BoundingBox = result.BoundingBox,
            KeyPoints = result.KeyPoints
        };
        #endregion
    }
}
