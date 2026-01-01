// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models.Interfaces
{
    /// <summary>
    /// 执行提供程序实现的接口，用于在 ONNX 模型上运行推理。
    /// </summary>
    public interface IExecutionProvider
    {
        /// <summary>
        /// 使用提供的归一化像素数据在模型上运行推理的方法。
        /// </summary>
        /// <typeparam name="T">数据类型</typeparam>
        /// <param name="normalizedPixels">归一化像素数据</param>
        /// <returns>推理结果</returns>
        public InferenceResult Run<T>(T[] normalizedPixels) where T : unmanaged;

        /// <summary>
        /// 包含有关 ONNX 模型元数据的记录。
        /// </summary>
        public OnnxDataRecord OnnxData { get; }
    }
}
