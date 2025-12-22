// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    /// <summary>
    /// 表示对象检测结果，包括标签信息、置信度分数和边界框。
    /// </summary>
    public class OBBDetection : TrackingInfo, IDetection
    {
        /// <summary>
        /// 与检测到的对象关联的标签信息。
        /// </summary>
        public LabelModel Label { get; init; } = new();

        /// <summary>
        /// 检测到的对象的置信度分数。
        /// </summary>
        public double Confidence { get; init; }

        /// <summary>
        /// 定义检测到的对象的感兴趣区域（边界框）的矩形。
        /// </summary>
        public SKRectI BoundingBox { get; init; }

        /// <summary>
        /// 边界框的方向角度
        /// </summary>
        public float OrientationAngle { get; set; }

    }
}
