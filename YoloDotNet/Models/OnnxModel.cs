// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    /// <summary>
    /// 表示用于对象检测的 ONNX 模型的配置和元数据。
    /// </summary>
    public record OnnxModel
    {
        /// <summary>
        /// 获取模型的类型，例如对象检测、分类等。
        /// </summary>
        public ModelType ModelType { get; init; }

        /// <summary>
        /// 获取或设置所用模型的版本。
        /// </summary>
        public ModelVersion ModelVersion { get; set; }

        /// <summary>
        /// 获取模型的数据类型。Float32 或 Float16。
        /// </summary>
        public ModelDataType ModelDataType { get; init; }

        /// <summary>
        /// ONNX 模型中输入张量的名称。
        /// </summary>
        public string InputName { get; init; } = default!;

        /// <summary>
        /// ONNX 模型中输出张量的名称。
        /// </summary>
        public List<string> OutputNames { get; init; } = default!;

        /// <summary>
        /// ONNX 模型的输入张量配置。
        /// </summary>
        public Input Input { get; init; } = default!;

        /// <summary>
        /// ONNX 模型的输出张量配置。
        /// </summary>
        public List<Output> Outputs { get; init; } = default!;

        /// <summary>
        /// 用于对象检测的标签模型数组。
        /// </summary>
        public LabelModel[] Labels { get; init; } = default!;

        /// <summary>
        /// 用于创建张量的 ONNX 模型输入形状
        /// </summary>
        public long[] InputShape { get; init; } = default!;

        /// <summary>
        /// 获取用于张量分配和数组池化的输入形状的大小。
        /// </summary>
        public int InputShapeSize { get; init; }

        /// <summary>
        /// ONNX 自定义元数据
        /// </summary>
        public Dictionary<string, string> CustomMetaData { get; set; } = [];
    }

    /// <summary>
    /// 表示 ONNX 模型在 BCHW 顺序中输入数据的配置
    /// [Batch, Channels, Height, Width]
    /// </summary>
    /// <param name="BatchSize">输入数据的批次大小。</param>
    /// <param name="Channels">输入通道的数量。</param>
    /// <param name="Height">输入数据的高度。</param>
    /// <param name="Width">输入数据的宽度。</param>
    public record Input(int BatchSize, int Channels, int Height, int Width)
    {
        public static Input Shape(int[] dimensions)
            => new(dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
    }

    /// <summary>
    /// 表示 ONNX 模型输出数据的配置。
    /// </summary>
    /// <param name="BatchSize">输入数据的批次大小。</param>
    /// <param name="Elements">输入数据的元素数量。</param>
    /// <param name="Channels">输入数据的通道数量。</param>
    /// <param name="Width">输入数据的宽度。</param>
    /// <param name="Height">输入数据的高度。</param>
    public record Output(int BatchSize, int Elements, int Channels, int Width, int Height)
    {
        public static Output Classification(int[] dimensions)
            => new(dimensions[0], dimensions[1], 0, 0, 0);

        public static Output Detection(int[] dimensions)
            => new(dimensions[0], dimensions[1], dimensions[2], 0, 0);

        public static Output Segmentation(int[] dimensions)
            => new(dimensions[0], 0, dimensions[1], dimensions[2], dimensions[3]);

        public static Output Empty() => new(0, 0, 0, 0, 0);
    }
}