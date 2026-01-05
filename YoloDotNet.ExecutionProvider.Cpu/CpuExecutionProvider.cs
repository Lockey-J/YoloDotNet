// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider.Cpu
{
    public class CpuExecutionProvider : IExecutionProvider, IDisposable
    {
        public OnnxDataRecord OnnxData { get; private set; } = default!;

        #region Fields
        private InferenceSession _session = default!;
        private RunOptions _runOptions = default!;
        private long[] _inputShape = default!;

        private float[] _outputBuffer0 = default!;
        private float[] _outputBuffer1 = default!;

        private TensorElementType _elementDataType = default!;
        private int _inputShapeSize;
        #endregion

        #region Constructors
        /// <summary>
        /// 构造用于在 CPU 上运行 ONNX 模型的 CpuExecutionProvider。
        /// </summary>
        /// <param name="model">ONNX 模型文件路径。</param>
        /// <param name="customOnnxDataRecord">自定义的 ONNX 数据记录，如果为 null 则自动从模型提取。</param>
        public CpuExecutionProvider(string model, OnnxDataRecord? customOnnxDataRecord = null)
        {
            InitializeYolo(model, customOnnxDataRecord);
        }

        /// <summary>
        /// 构造用于在 CPU 上运行 ONNX 模型的 CpuExecutionProvider。
        /// </summary>
        /// <param name="model">ONNX 模型字节数组。</param>
        /// <param name="customOnnxDataRecord">自定义的 ONNX 数据记录，如果为 null 则自动从模型提取。</param>
        public CpuExecutionProvider(byte[] model, OnnxDataRecord? customOnnxDataRecord = null)
        {
            InitializeYolo(model, customOnnxDataRecord);
        }
        #endregion

        #region Initialization
        private void InitializeYolo(object model, OnnxDataRecord? customOnnxDataRecord)
        {
            ConfigureOrtEnv();

            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                EnableCpuMemArena = true
            };

            // 如果可用，使用字节数组创建会话；否则从文件加载并使用选定的提供程序。
            _session = (model is byte[] modelBytes)
                ? new InferenceSession(modelBytes, options)
                : new InferenceSession((string)model, options);

            // 使用自定义 OnnxDataRecord 或自动提取
            if (customOnnxDataRecord != null)
            {
                OnnxData = customOnnxDataRecord;
            }
            else
            {
                GetOnnxMetaData();
            }

            AllocateOutputBuffers();

            _runOptions = new RunOptions();

            //设置在推理过程中创建张量的输入形状。
            _inputShape = [.. OnnxData.InputShape.Select(i => (long)i)];
        }
        #endregion

        #region Run Inference
        /// <summary>
        /// 对提供的归一化像素数据运行推理。
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="normalizedPixels"></param>
        /// <returns></returns>
        unsafe public InferenceResult Run<T>(T[] normalizedPixels) where T : unmanaged
        {
            // 固定 ushort[] 以便我们可以获取原始指针
            fixed (T* pData = normalizedPixels)
            {

                var elementType = (typeof(T) == typeof(float)) ? TensorElementType.Float : TensorElementType.Float16;

                using var inputOrtValue = OrtValue.CreateTensorValueWithData(
                    OrtMemoryInfo.DefaultInstance,
                    elementType,   // 👈 强制 ONNX 将缓冲区解释为 Float16
                    _inputShape,
                    (IntPtr)pData,
                    _inputShapeSize * sizeof(T) // size in bytes (ushort is 2 bytes, float is 4 bytes)
                );

                // 运行推理
                using var result = _session.Run(
                    _runOptions,
                    [OnnxData.InputName],
                    [inputOrtValue],
                    OnnxData.OutputNames);

                if (elementType == TensorElementType.Float)
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

        #region Helper methods

        /// <summary>
        /// Configures the ONNX Runtime environment to log errors and fatals only.
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
        /// 为 float16 模型分配输出缓冲区。
        /// </summary>
        private void AllocateOutputBuffers()
        {
            // 如果模型使用 Float16 数据类型，则预分配输出缓冲区。
            if (OnnxData.ModelDataType == ModelDataType.Float16)
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
            if (OnnxData.OutputShapes.Length == 2)
            {
                (items, elements) = (OnnxData.OutputShapes[1][1], OnnxData.OutputShapes[1][2]);
                _outputBuffer1 = new float[elements * items];
            }
        }

        private void ConvertFloat16ToFloat(ReadOnlySpan<Float16> tensorData0, ReadOnlySpan<Float16> tensorData1)
        {
            // 将 Float16 转换为 float 并存储在预分配的缓冲区中
            for (int i = 0; i < tensorData0.Length; i++)
                _outputBuffer0[i] = (float)tensorData0[i];

            for (int i = 0; i < tensorData1.Length; i++)
                _outputBuffer1[i] = (float)tensorData1[i];
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
            var inputSize = (int)ShapeUtils.GetSizeForShape(inputShape);

            _elementDataType = GetModelElementType();
            _inputShapeSize = (int)ShapeUtils.GetSizeForShape(inputShape);

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

            GC.SuppressFinalize(this);
        }

        #endregion
    }
}