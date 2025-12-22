// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    /// <summary>
    /// 用于绘制分割结果的配置选项。
    /// </summary>
    public class SegmentationDrawingOptions : CommonBoundingBoxOptions
    {
        /// <summary>
        /// 是否从分割结果绘制像素掩码。
        /// </summary>
        public bool DrawSegmentationPixelMask { get; set; } = true;

        #region Mapping method
        public static explicit operator DetectionDrawingOptions(SegmentationDrawingOptions options) => new()
        {
            Font = options.Font,
            FontColor = options.FontColor,
            EnableFontShadow = options.EnableFontShadow,
            EnableDynamicScaling = options.EnableDynamicScaling,
            DrawLabelBackground = options.DrawLabelBackground,
            DrawBoundingBoxes = options.DrawBoundingBoxes,
            DrawLabels = options.DrawLabels,
            DrawConfidenceScore = options.DrawConfidenceScore,
            DrawTrackedTail = options.DrawTrackedTail,
            BorderThickness = options.BorderThickness,
            BoundingBoxOpacity = options.BoundingBoxOpacity,
            BoundingBoxHexColors = options.BoundingBoxHexColors,
            TailThickness = options.TailThickness,
            TailPaintColorStart = options.TailPaintColorStart,
            TailPaintColorEnd = options.TailPaintColorEnd,
        };
        #endregion
    }
}