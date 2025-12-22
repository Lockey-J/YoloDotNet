// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    [EditorBrowsable(EditorBrowsableState.Never)]
    public class CommonBoundingBoxOptions : CommonDrawingFontOptions
    {
        /// <summary>
        /// 是否在检测到的对象周围绘制边界框。
        /// </summary>
        public bool DrawBoundingBoxes { get; set; } = true;

        /// <summary>
        /// 是否在检测到的对象上绘制标签。
        /// </summary>
        public bool DrawLabels { get; set; } = true;

        /// <summary>
        /// 是否在标签旁边绘制置信度分数。
        /// </summary>
        public bool DrawConfidenceScore { get; set; } = true;

        /// <summary>
        /// 是否绘制跟踪对象的轨迹。
        /// </summary>
        public bool DrawTrackedTail { get; set; } = true;

        /// <summary>
        /// 边界框边框的粗细，以像素为单位。
        /// </summary>
        public float BorderThickness { get; set; } = ImageConfig.BORDER_THICKNESS;

        /// <summary>
        /// 边界框的不透明度级别 (0-255)。
        /// </summary>
        public int BoundingBoxOpacity { get; set; } = ImageConfig.DEFAULT_OPACITY;

        /// <summary>
        /// 边界框颜色的十六进制字符串数组（例如，"#FF0000"）。
        /// 如果未指定颜色，将使用默认颜色。
        /// </summary>
        public string[] BoundingBoxHexColors { get; set; } = YoloDotNetColors.Get();

        /// <summary>
        /// 为对象跟踪可视化绘制的轨迹线的粗细。
        /// </summary>
        public float TailThickness { get; set; } = ImageConfig.TailThickness;

        /// <summary>
        /// 跟踪轨迹线中使用的渐变的起始颜色。
        /// </summary>
        public SKColor TailPaintColorStart { get; set; } = ImageConfig.TailPaintColorStart;

        /// <summary>
        /// 跟踪轨迹线中使用的渐变的结束颜色。
        /// </summary>
        public SKColor TailPaintColorEnd { get; set; } = ImageConfig.TailPaintColorEnd;
    }
}
