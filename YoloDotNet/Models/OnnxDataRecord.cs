// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    /// <summary>
    /// 用于保存来自执行提供程序的 ONNX 模型数据的记录。
    /// </summary>
    /// <param name="MetaData">元数据</param>
    /// <param name="ModelDataType">模型数据类型</param>
    /// <param name="InputName">输入名称</param>
    /// <param name="OutputNames">输出名称</param>
    /// <param name="InputShape">输入形状</param>
    /// <param name="OutputShapes">输出形状</param>
    /// <param name="InputShapeSize">输入形状大小</param>
    /// <param name="Labels">标签</param>
    public record OnnxDataRecord(
        Dictionary<string, string> MetaData,
        ModelDataType ModelDataType,
        string InputName,
        string[] OutputNames,
        int[] InputShape,
        int[][] OutputShapes,
        int InputShapeSize,
        string Labels
    );
}