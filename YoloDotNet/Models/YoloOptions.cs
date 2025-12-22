// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    /// <summary>
    /// 表示配置 Yolo 对象的选项。
    /// </summary>
    public class YoloOptions
    {
        /// <summary>
        /// 获取或设置执行提供程序（CPU、CUDA 或 TensorRT）。
        /// </summary>
        public IExecutionProvider ExecutionProvider { get; set; } = default!;

        /// <summary>
        /// 获取或设置 onnx 模型所需的图像调整大小类型。
        /// </summary>
        public ImageResize ImageResize { get; set; }

        /// <summary>
        /// 针对高效缩小优化的 SkiaSharp 采样选项。
        /// </summary>
        /// <remarks>
        /// - **默认值：** 线性过滤（`SKFilterMode.Linear`），无多级渐远纹理插值（`SKMipmapMode.None`）。
        /// - **性能：** 缩小图像时快速高效。
        /// - **质量：** 产生平滑的结果，具有最小的锯齿。
        /// - **最佳使用场景：** 在减小图像大小的同时保持速度和质量之间的平衡时理想。
        /// - **可修改性：** 此属性可以在运行时更改以调整过滤行为。
        /// </remarks>
        public SKSamplingOptions SamplingOptions { get; set; } = ImageConfig.DefaultSamplingOptions;
    }
}
