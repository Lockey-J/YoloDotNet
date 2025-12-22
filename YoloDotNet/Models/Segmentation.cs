// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    public class Segmentation : TrackingInfo, IDetection
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
        /// 位压缩掩码，其中每个位表示置信度高于阈值的像素（1 = 存在，0 = 不存在）。
        /// 可以使用 <c>UnpackToBitmap</c> 扩展方法解包为 <see cref="SKBitmap"/>。
        /// </summary>
        public byte[] BitPackedPixelMask { get; set; } = [];
    }
}
