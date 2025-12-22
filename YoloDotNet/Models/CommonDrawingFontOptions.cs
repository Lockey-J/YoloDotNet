// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    [EditorBrowsable(EditorBrowsableState.Never)]
    public class CommonDrawingFontOptions
    {
        /// <summary>
        /// 用于绘制标签和置信度分数的字体。
        /// </summary>
        public SKTypeface Font { get; set; } = SKTypeface.Default;

        /// <summary>
        /// 字体大小。
        /// </summary>
        public float FontSize { get; set; } = ImageConfig.FONT_SIZE;

        /// <summary>
        /// 用于绘制标签和置信度分数的字体颜色。
        /// </summary>
        public SKColor FontColor { get; set; } = ImageConfig.FontColor;

        /// <summary>
        /// 是否启用文本后面的阴影效果。
        /// </summary>
        public bool EnableFontShadow { get; set; } = true;

        /// <summary>
        /// 是否根据图像分辨率动态缩放字体大小和边框粗细。
        /// </summary>
        public bool EnableDynamicScaling { get; set; } = true;

        /// <summary>
        /// 是否绘制标签背景。
        /// </summary>
        public bool DrawLabelBackground { get; set; } = true;
    }
}
