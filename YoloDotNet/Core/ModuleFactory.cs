// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Core
{
    /// <summary>
    /// 用于根据模型版本和类型创建 YOLO 检测模块的工厂类。
    /// </summary>
    internal class ModuleFactory
    {
        // 将模型版本和类型映射到各自模块创建函数的字典。
        private static readonly Dictionary<ModelVersion, Dictionary<ModelType, Func<YoloCore, IModule>>> _versionModuleMap =
        new()
        {
            {
                ModelVersion.V5U, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core =>throw new NotImplementedException() },
                    { ModelType.ObjectDetection, core => new ObjectDetectionModuleV5U(core) },
                    { ModelType.ObbDetection, core => throw new NotImplementedException() },
                    { ModelType.Segmentation, core => throw new NotImplementedException() },
                    { ModelType.PoseEstimation, core => throw new NotImplementedException() }
                }
            },
            {
                ModelVersion.V8, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core => new ClassificationModuleV8(core) },
                    { ModelType.ObjectDetection, core => new ObjectDetectionModuleV8(core) },
                    { ModelType.ObbDetection, core => new OBBDetectionModuleV8(core) },
                    { ModelType.Segmentation, core => new SegmentationModuleV8(core) },
                    { ModelType.PoseEstimation, core => new PoseEstimationModuleV8(core) }
                }
            },
            {
                ModelVersion.V8E, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core => throw new NotImplementedException() },
                    { ModelType.ObjectDetection, core => throw new NotImplementedException() },
                    { ModelType.ObbDetection, core => throw new NotImplementedException() },
                    { ModelType.Segmentation, core => new SegmentationModuleV8E(core) },
                    { ModelType.PoseEstimation, core => throw new NotImplementedException() }
                }
            },
            {
                ModelVersion.V9, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core => throw new NotImplementedException() },
                    { ModelType.ObjectDetection, core => new ObjectDetectionModuleV9(core) },
                    { ModelType.ObbDetection, core => throw new NotImplementedException() },
                    { ModelType.Segmentation, core => throw new NotImplementedException() },
                    { ModelType.PoseEstimation, core => throw new NotImplementedException() }
                }
            },
            {
                ModelVersion.V10, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core => throw new NotImplementedException() },
                    { ModelType.ObjectDetection, core => new ObjectDetectionModuleV10(core) },
                    { ModelType.ObbDetection, core => throw new NotImplementedException() },
                    { ModelType.Segmentation, core => throw new NotImplementedException() },
                    { ModelType.PoseEstimation, core => throw new NotImplementedException() }
                }
            },
            {
                ModelVersion.V11, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core => new ClassificationModuleV11(core) },
                    { ModelType.ObjectDetection, core => new ObjectDetectionModuleV11(core) },
                    { ModelType.ObbDetection, core => new OBBDetectionModuleV11(core) },
                    { ModelType.Segmentation, core => new SegmentationModuleV11(core) },
                    { ModelType.PoseEstimation, core => new PoseEstimationModuleV11(core) }
                }
            },
            {
                ModelVersion.V11E, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core => throw new NotImplementedException() },
                    { ModelType.ObjectDetection, core => throw new NotImplementedException() },
                    { ModelType.ObbDetection, core => throw new NotImplementedException() },
                    { ModelType.Segmentation, core => new SegmentationModuleV11E(core) },
                    { ModelType.PoseEstimation, core => throw new NotImplementedException() }
                }
            },
            {
                ModelVersion.V12, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core => new ClassificationModuleV12(core) },
                    { ModelType.ObjectDetection, core => new ObjectDetectionModuleV12(core) },
                    { ModelType.ObbDetection, core => new OBBDetectionModuleV12(core) },
                    { ModelType.Segmentation, core => new SegmentationModuleV12(core) },
                    { ModelType.PoseEstimation, core => new PoseEstimationModuleV12(core) }
                }
            },
            {
                ModelVersion.WORLDV2, new Dictionary<ModelType, Func<YoloCore, IModule>>
                {
                    { ModelType.Classification, core => throw new NotImplementedException() },
                    { ModelType.ObjectDetection, core => new ObjectDetectionModuleWorldV2(core) },
                    { ModelType.ObbDetection, core =>  throw new NotImplementedException() },
                    { ModelType.Segmentation, core =>  throw new NotImplementedException() },
                    { ModelType.PoseEstimation, core =>  throw new NotImplementedException() }
                }
            }
        };

        /// <summary>
        /// 根据指定的 YOLO 选项创建检测模块。
        /// </summary>
        /// <param name="options">创建 YOLO 检测模块的选项。</param>
        /// <returns>适当检测模块的实例。</returns>
        /// <exception cref="YoloDotNetModelException">如果模型版本或类型不受支持，则抛出。</exception>
        public static IModule CreateModule(YoloOptions options)
        {
            var yoloCore = InitializeYoloCore(options);

            // 获取模型版本和类型
            var modelVersion = yoloCore.OnnxModel.ModelVersion;
            var modelType = yoloCore.ModelType;

            // 根据模型版本从模块映射中获取字典
            var versionSelected = _versionModuleMap.TryGetValue(modelVersion, out var moduleMap);
            var moduleSelected = moduleMap!.TryGetValue(modelType, out var createModule);

            if (versionSelected && moduleSelected)
                return createModule!(yoloCore);

            throw new YoloDotNetModelException($"Unsupported detection type {modelType} or model version {modelVersion}.");
        }

        /// <summary>
        /// 根据指定选项初始化 YoloCore。
        /// </summary>
        /// <param name="options">初始化 Yolo 模型的选项。</param>
        /// <returns>初始化的 YoloCore 实例。</returns>
        private static YoloCore InitializeYoloCore(YoloOptions options)
        {
            //var yoloCore = new YoloCore(options.OnnxModel, options.Cuda, options.PrimeGpu, options.GpuId);
            var yoloCore = new YoloCore(options);
            yoloCore.InitializeYolo();
            return yoloCore;
        }
    }
}
