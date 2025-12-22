// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    public class PoseEstimation : TrackingInfo, IDetection
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
        /// 带有 x、y 坐标和置信度分数的关键点
        /// </summary>
        public KeyPoint[] KeyPoints { get; set; } = [];
    }
}
