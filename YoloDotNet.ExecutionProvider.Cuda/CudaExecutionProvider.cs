// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider.Cuda
{
    public class CudaExecutionProvider : IExecutionProvider, IDisposable
    {
        public OnnxDataRecord OnnxData { get; private set; } = default!;

        #region Fields
        private InferenceSession _session = default!;
        private OrtIoBinding _ortIoBinding = default!;
        private RunOptions _runOptions = default!;

        private float[] _outputBuffer0 = default!;
        private float[] _outputBuffer1 = default!;

        private long[] _inputShape = default!;
        private int _inputShapeSize;
        private TensorElementType _elementDataType = default!;
        private int _dataTypeSize;
        #endregion

        #region Constructors
        /// <summary>
        /// 构造用于使用 CUDA 和可选的 TensorRT 运行 ONNX 模型的 CudaExecutionProvider。
        /// </summary>
        /// <param name="model"></param>
        /// <param name="gpuId"></param>
        /// <param name="trtConfig"></param>
        public CudaExecutionProvider(string model, int gpuId = 0, TensorRt? trtConfig = null)
        {
            InitializeYolo(model, gpuId, trtConfig);
        }

        /// <summary>
        /// 重载：构造用于使用 CUDA 和可选的 TensorRT 运行 ONNX 模型的 CudaExecutionProvider。
        /// </summary>
        /// <param name="model"></param>
        /// <param name="gpuId"></param>
        /// <param name="trtConfig"></param>
        public CudaExecutionProvider(byte[] model, int gpuId = 0, TensorRt? trtConfig = null)
        {
            InitializeYolo(model, gpuId, trtConfig);
        }
        #endregion

        #region Initialization
        /// <summary>
        /// 初始化 ONNX Runtime 会话，配置 CUDA 执行提供程序并分配资源。
        /// </summary>
        /// <param name="model"></param>
        /// <param name="gpuId"></param>
        /// <param name="trtConfig"></param>
        private void InitializeYolo(object model, int gpuId, TensorRt? trtConfig)
        {
            ConfigureOrtEnv();

            var options = CreateSessionOptions(gpuId, trtConfig);

            // 如果可用，使用字节数组创建会话；否则从文件加载并使用选定的提供程序。
            _session = (model is byte[] modelBytes)
                ? new InferenceSession(modelBytes, options)
                : new InferenceSession((string)model, options);

            GetOnnxMetaData();
            AllocateOutputBuffers();

            _runOptions = new RunOptions();
            _ortIoBinding = _session.CreateIoBinding();
            _session.AllocateGpuMemory(_ortIoBinding, _runOptions, _elementDataType);

            // Set the input shape for creating tensors during inference.
            _inputShape = [.. OnnxData.InputShape.Select(i => (long)i)];

        }
        #endregion

        #region Run Inference
        /// <summary>
        /// 对提供的归一化像素数据运行推理。
        /// </summary>
        /// <param name="normalizedPixels"></param>
        /// <returns></returns>
        unsafe public InferenceResult Run<T>(T[] normalizedPixels) where T : unmanaged
        {
            // 在内存中固定输入像素数据，防止垃圾回收器移动它。
            fixed (T* pData = normalizedPixels)
            {
                // 从固定数据创建 OrtValue 张量
                using var inputOrtValue = OrtValue.CreateTensorValueWithData(
                    OrtMemoryInfo.DefaultInstance,
                    _elementDataType,
                    _inputShape,
                    (IntPtr)pData,
                    _inputShapeSize * _dataTypeSize // 大小以字节为单位（ushort 为 2 字节，float 为 4 字节）
                );

                // 运行推理
                using var result = _session.Run(
                    _runOptions,
                    [OnnxData.InputName],
                    [inputOrtValue],
                    OnnxData.OutputNames);

                // 根据模型的数据类型处理输出
                if (_elementDataType == TensorElementType.Float)
                {
                    // 从结果中提取张量数据
                    var tensorData0 = result[0].GetTensorDataAsSpan<float>();
                    var tensorData1 = ReadOnlySpan<float>.Empty;

                    if (result.Count == 2)
                        tensorData1 = result[1].GetTensorDataAsSpan<float>();

                    // 返回包含输出张量数据的推理结果
                    return new InferenceResult(tensorData0, tensorData1);
                }
                else
                {
                    var tensorData0 = result[0].GetTensorDataAsSpan<Float16>();
                    var tensorData1 = ReadOnlySpan<Float16>.Empty;

                    if (result.Count == 2)
                        tensorData1 = result[1].GetTensorDataAsSpan<Float16>();

                    ConvertFloat16ToFloat(tensorData0, tensorData1);

                    // 返回包含输出张量数据的推理结果
                    return new InferenceResult(_outputBuffer0, _outputBuffer1);
                }
            }
        }
        #endregion

        #region CUDA and TensorRT helper methods
        /// <summary>
        /// 为使用 Float16 数据类型的模型分配 float 输出缓冲区。
        /// </summary>
        private void AllocateOutputBuffers()
        {
            // 如果模型使用 Float16，则预分配输出缓冲区以避免推理过程中重复分配。
            if (OnnxData.ModelDataType == ModelDataType.Float)
                return;

            int items;
            int elements;

            // 计算每个输出张量的元素总数并分配缓冲区。

            // 分类模型只有一个输出张量，形状为 [1, num_classes]
            if (OnnxData.OutputShapes[0].Length == 2)
            {
                (items, elements) = (OnnxData.OutputShapes[0][0], OnnxData.OutputShapes[0][1]);
                _outputBuffer0 = new float[elements * items];
            }
            // 所有其他模型的输出张量形状为 [1, num_boxes, num_attributes]
            else
            {
                (items, elements) = (OnnxData.OutputShapes[0][1], OnnxData.OutputShapes[0][2]);
                _outputBuffer0 = new float[elements * items];
            }

            // 如果有第二个输出张量（分割），也为它分配缓冲区。

            // 如果有第二个输出张量，也为它分配缓冲区。
            if (OnnxData.OutputShapes.Length == 2)
            {
                (items, elements) = (OnnxData.OutputShapes[1][1], OnnxData.OutputShapes[1][2]);
                _outputBuffer1 = new float[elements * items];
            }
        }

        /// <summary>
        /// 为 ONNX Runtime 会话创建和配置会话选项。
        /// </summary>
        /// <param name="gpuId"></param>
        /// <param name="trtConfig"></param>
        private SessionOptions CreateSessionOptions(int gpuId, TensorRt? trtConfig)
        {
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };

            if (gpuId >= 0)
            {
                if (trtConfig is not null)
                {
                    options.ConfigureTensorRT(gpuId, trtConfig);
                }
                else
                {
                    ConfigureCuda(gpuId, options);
                }
            }
            else if (gpuId == -1)
            {
                options.EnableCpuMemArena = true;
            }
            else
            {
                throw new ArgumentOutOfRangeException(
                    paramName: nameof(gpuId),
                    actualValue: gpuId,
                    message: "指定的 gpuId 无效。使用 -1 进行 CPU 执行，或使用 0 及以上值作为 GPU 设备 ID。");
            }

            return options;
        }

        /// <summary>
        /// 使用自定义日志选项配置全局 OrtEnv 实例。
        /// </summary>
        private static void ConfigureOrtEnv()
        {
            try
            {
                // 记录错误和致命错误
                var envOptions = new EnvironmentCreationOptions
                {
                    logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
                };

                OrtEnv.CreateInstanceWithOptions(ref envOptions);
            }
            catch (OnnxRuntimeException ex) when (ex.Message.Contains("OrtEnv singleton instance already exists"))
            {
                // OrtEnv 已经初始化 — 忽略并优雅地继续...
            }
        }

        /// <summary>
        /// 配置会话选项以使用带有指定选项的 CUDA 执行提供程序。
        /// </summary>
        /// <param name="gpuId"></param>
        /// <param name="options"></param>
        private static void ConfigureCuda(int gpuId, SessionOptions options)
        {
            var cudaOptions = new OrtCUDAProviderOptions();

            cudaOptions.UpdateOptions(new Dictionary<string, string>
            {
                { "device_id", gpuId.ToString() },
                // 指定要使用的 GPU 设备（如果未设置，默认为 0）。

                { "arena_extend_strategy", "kNextPowerOfTwo" }, 
                // 控制当需要更多内存时 GPU 内存 arena 的增长方式。
                // kNextPowerOfTwo 将分配大小翻倍到下一个二次幂，
                // 这会减少 CUDA malloc/free 调用的频率并最小化碎片 
                // 在长时间运行或高吞吐量推理场景（如 YOLO 目标检测）中。

                { "cudnn_conv_algo_search", "EXHAUSTIVE" },
                // 强制 cuDNN 在模型初始化期间对所有可用的卷积算法进行基准测试
                // 并为硬件 + 模型组合选择最快的一个。
                // 这在运行时提供最佳的卷积核性能，特别对大型或自定义卷积层有益。

                // 通过在支持时使用默认流启用复制来减少主机/设备同步。
                // 这可以在某些 ONNX Runtime 构建中避免隐式流同步。
                { "do_copy_in_default_stream", "1" },

                // 允许 cuDNN 使用最大工作空间（可能增加内存使用但可以提高内核性能）。
                { "cudnn_conv_use_max_workspace", "1" }

            });

            options.AppendExecutionProvider_CUDA(cudaOptions);
        }

        /// <summary>
        /// 将 Float16 张量数据转换为 Float32 并存储在预分配的输出缓冲区中。
        /// </summary>
        /// <param name="tensorData0"></param>
        /// <param name="tensorData1"></param>
        unsafe private void ConvertFloat16ToFloat(ReadOnlySpan<Float16> tensorData0, ReadOnlySpan<Float16> tensorData1)
        {
            fixed (Float16* src0 = tensorData0)
            fixed (Float16* src1 = tensorData1)
            fixed (float* dst0 = _outputBuffer0)
            fixed (float* dst1 = _outputBuffer1)
            {
                int len0 = tensorData0.Length;
                int len1 = tensorData1.Length;

                for (int i = 0; i < len0; i++)
                    dst0[i] = (float)src0[i];

                for (int i = 0; i < len1; i++)
                    dst1[i] = (float)src1[i];
            }
        }

        /// <summary>
        /// 从 ONNX 模型中提取元数据和输入/输出形状。
        /// </summary>
        private void GetOnnxMetaData()
        {
            // 从 ONNX 模型中提取自定义元数据。
            var metaData = _session.ModelMetadata.CustomMetadataMap;

            // 获取输入形状和大小。
            var inputShape = Array.ConvertAll(_session.InputMetadata[_session.InputNames[0]].Dimensions, Convert.ToInt64);

            _inputShapeSize = (int)ShapeUtils.GetSizeForShape(inputShape);
            _elementDataType = GetModelElementType();
            _dataTypeSize = _elementDataType == TensorElementType.Float16 ? sizeof(ushort) : sizeof(float);

            // 确定模型数据类型（Float32 或 Float16）。
            var modelDataType = _elementDataType == TensorElementType.Float16
                ? ModelDataType.Float16
                : ModelDataType.Float;

            // 创建 OnnxDataRecord 来保存模型信息。
            OnnxData = new OnnxDataRecord(
                metaData,
                modelDataType,
                _session.InputNames[0],
                [.. _session.OutputNames],
                _session.InputMetadata.Values.Select(x => x.Dimensions).First(),
                [.. _session.OutputMetadata.Values.Select(x => x.Dimensions)],
                _inputShapeSize,
                metaData["names"]
            );
        }

        /// <summary>
        /// 获取模型使用的张量元素类型（例如，Float32 或 Float16）。
        /// </summary>
        internal TensorElementType GetModelElementType()
            => _session.InputMetadata["images"].ElementDataType;

        public void Dispose()
        {
            _session?.Dispose();
            _runOptions?.Dispose();
            _ortIoBinding?.Dispose();

            GC.SuppressFinalize(this);
        }
        #endregion
    }
}