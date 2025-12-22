// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider.Cuda
{
    public static class GpuExtension
    {
        /// <summary>
        /// 为输入数据分配 GPU 内存并确保内存同步。
        /// </summary>
        public static void AllocateGpuMemory(this InferenceSession session,
            OrtIoBinding ortIoBinding,
            RunOptions runOptions,
            TensorElementType tensorElementType)
        {
            // 获取输入形状。
            var inputShape = Array.ConvertAll(session.InputMetadata[session.InputNames[0]].Dimensions, Convert.ToInt64);

            // 根据模型数据类型确定字节大小。
            var byteSize = tensorElementType == TensorElementType.Float ? sizeof(float) : sizeof(ushort);

            // 计算输入大小。
            var inputSizeInBytes = ShapeUtils.GetSizeForShape(inputShape) * byteSize;

            // 分配非托管内存。
            nint allocPtr = Marshal.AllocHGlobal((int)inputSizeInBytes);

            // 使用分配的内存作为数据缓冲区创建 OrtValue。
            using (var ortValueTensor = OrtValue.CreateTensorValueWithData(
                OrtMemoryInfo.DefaultInstance,
                tensorElementType,
                inputShape,
                allocPtr,
                inputSizeInBytes))
            {
                ortIoBinding.BindInput(session.InputNames[0], ortValueTensor);
            }

            // 绑定输出
            ortIoBinding.BindOutputToDevice(session.OutputNames[0], OrtMemoryInfo.DefaultInstance);

            // 在运行推理前确保输入数据与内存正确同步。
            ortIoBinding.SynchronizeBoundInputs();

            // 在 OrtIoBinding 上运行推理并绑定分配的 GPU 内存。
            session.RunWithBinding(runOptions, ortIoBinding);

            // 在运行推理后确保输出数据与内存正确同步。
            ortIoBinding.SynchronizeBoundOutputs();
        }
    }
}
