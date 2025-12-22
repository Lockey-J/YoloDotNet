// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    public class PoseDrawingOptions : CommonBoundingBoxOptions
    {
        /// <summary>
        /// 显示关键点的置信度阈值。
        /// </summary>
        public double PoseConfidence { get; set; } = ImageConfig.POSE_KEYPOINT_THRESHOLD;

        /// <summary>
        /// 当未通过 <see cref="KeyPointMarkers"/> 提供特定颜色或标记时，用于绘制关键点的默认颜色。
        /// </summary>
        public SKColor DefaultPoseColor { get; set; } = ImageConfig.PoseMarkerColor;

        /// <summary>
        /// 用户定义的映射，用于确定如何连接关键点并指定相关颜色。
        /// </summary>
        public KeyPointMarker[] KeyPointMarkers { get; set; } = [];

        #region Mapping method
        public static explicit operator DetectionDrawingOptions(PoseDrawingOptions options) => new()
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

    /// <summary>
    /// 表示关键点及其连接之间的映射。
    /// </summary>
    public record KeyPointMarker
    {
        /// <summary>
        /// 与关键点关联的颜色。
        /// </summary>
        public string Color { get; init; } = default!;

        /// <summary>
        /// 定义关键点之间的连接。
        /// </summary>
        public KeyPointConnection[] Connections { get; init; } = [];
    }

    /// <summary>
    /// 表示姿态标记与其父标记之间的连接，由索引和颜色定义。
    /// </summary>
    /// <param name="Index">索引</param>
    /// <param name="Color">颜色</param>
    public record KeyPointConnection(int Index, string Color);
}
