// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.ExecutionProvider.OpenVino
{
    public class OpenVinoExecutionProvider : IExecutionProvider, IDisposable
    {
        public OnnxDataRecord OnnxData { get; private set; } = default!;

        #region Fields
        private InferenceSession _session = default!;
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
        /// 构造用于使用 Intel GPU 运行 ONNX 模型的 OpenVinoExecutionProvider。
        /// </summary>
        /// <param name="model"></param>
        /// <param name="openVino"></param>
        public OpenVinoExecutionProvider(string model, OpenVino? openVino = null)
        {
            InitializeYolo(model, openVino);
        }

        /// <summary>
        /// 构造用于使用 Intel GPU 运行 ONNX 模型的 OpenVinoExecutionProvider。
        /// </summary>
        /// <param name="model"></param>
        /// <param name="openVino"></param>
        public OpenVinoExecutionProvider(object model, OpenVino? openVino = null)
        {
            InitializeYolo(model, openVino);
        }
        #endregion

        #region Initialization
        private void InitializeYolo(object model, OpenVino? openVino)
        {
            ConfigureOrtEnv();

            var options = CreateSessionOptions(openVino);

            _session = (model is byte[] modelBytes)
                ? new InferenceSession(modelBytes, options)
                : new InferenceSession((string)model, options);

            _runOptions = new RunOptions();

            GetOnnxMetaData();
            AllocateOutputBuffers();

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

        #region OpenVino helper methods
        private static SessionOptions CreateSessionOptions(OpenVino? openVino)
        {
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };

            if (openVino is null
                || string.IsNullOrEmpty(openVino.DeviceType)
                || openVino.DeviceType.StartsWith("CPU", StringComparison.OrdinalIgnoreCase))
            {
                options.EnableCpuMemArena = true;
            }
            else
            {
                var ovOptions = new Dictionary<string, string>
                {
                    // OpenVINO EP 支持 device_type、precision、num_threads、num_streams、model_priority
                    ["device_type"] = openVino.DeviceType.ToUpper(),

                    // 精度模式：FP16 提高速度，FP32 提高精度。ACCURACY 使用模型定义的精度。
                    ["precision"] = openVino.Precision.ToString().ToUpper(), 

                    // 线程和流
                    ["num_of_threads"] = openVino.Threads.ToString(), // OpenVINO 默认为 8 个线程。
                    ["num_streams"] = openVino.Streams.ToString(), // OpenVINO 默认为 1 个流。

                    // 用于存储编译模型的缓存目录
                    ["cache_dir"] = openVino.CachePath.ToString(),

                    // 性能调整选项
                    ["disable_dynamic_shapes"] = "True",
                    ["model_priority"] = openVino.ModelPriority.ToString().ToUpper()
                };

                options.AppendExecutionProvider("OpenVINOExecutionProvider", ovOptions);
            }

            return options;
        }

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

        private static void ConfigureOrtEnv()
        {
            try
            {
                var envOptions = new EnvironmentCreationOptions
                {
                    logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
                };

                OrtEnv.CreateInstanceWithOptions(ref envOptions);
            }
            catch (OnnxRuntimeException ex) when (ex.Message.Contains("OrtEnv singleton instance already exists"))
            {
                // OrtEnv 已经初始化 - 忽略并优雅地继续...
            }
        }

        public void Dispose()
        {
            _session?.Dispose();
            _runOptions?.Dispose();

            GC.SuppressFinalize(this);
        }
        #endregion
    }
}
