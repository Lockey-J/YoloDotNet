// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    /// <summary>
    /// 表示图像分类的结果
    /// </summary>
    public class Classification : IClassification
    {
        /// <summary>
        /// 分类图像的标签。
        /// </summary>
        public string Label { get; set; } = default!;

        /// <summary>
        /// 分类图像的置信度分数。
        /// </summary>
        public double Confidence { get; set; }
    }
}
