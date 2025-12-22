// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Trackers
{
    public partial class SortTracker
    {
        private readonly Dictionary<int, TrackedObject> _trackedObjects = [];

        private int _nextId = 0;
        private readonly int _maxAge;
        private readonly float _costThreshold;
        private readonly int _tailLength;

        public SortTracker(float costThreshold = 0.5f, int maxAge = 3, int tailLength = 30)
        {
            _costThreshold = costThreshold;
            _maxAge = maxAge;
            _tailLength = tailLength;
        }

        /// <summary>
        /// 使用当前检测结果更新跟踪器状态。
        /// </summary>
        /// <typeparam name="T">实现 <see cref="IDetection"/> 的检测类型。</typeparam>
        /// <param name="detections">来自当前帧的检测结果列表。</param>
        /// <remarks>
        /// 该方法执行以下步骤：
        /// <list type="bullet">
        /// <item>使用卡尔曼滤波器预测现有跟踪对象的新位置。</item>
        /// <item>使用成本矩阵和分配算法将当前检测结果匹配到活动轨迹。</item>
        /// <item>当没有找到匹配轨迹时，将新的检测结果添加为未跟踪对象。</item>
        /// <item>如果没有活动轨迹存在，则初始化跟踪。</item>
        /// <item>移除超过最大允许年龄的旧跟踪对象。</item>
        /// </list>
        /// </remarks>
        public void UpdateTracker<T>(List<T> detections) where T : IDetection
        {
            // 如果没有要跟踪的内容，则无需进一步处理...
            if (detections.Count == 0)
            {
                RemoveOldTrackedObjects();
                return;
            }

            // 使用卡尔曼滤波器预测新位置
            foreach (var trackedObject in _trackedObjects.Values)
                trackedObject.KalmanPredict();

            // 根据之前的轨迹计数器获取之前的边界框
            var activeTracks = _trackedObjects.Where(x => x.Value.Age <= _maxAge).ToList();

            // 更新现有的跟踪对象并添加新的未跟踪对象。
            if (activeTracks.Count > 0)
            {
                var costMatrix = CalculateCostMatrix(activeTracks, detections);

                // 将检测到的对象与跟踪对象匹配
                var assignedIds = MatchPredictedObjects(detections, activeTracks, costMatrix);

                // 将未跟踪的新对象添加到跟踪器
                AddUntrackedObjects(detections, assignedIds);
            }
            else
            {
                // 跟踪器为空；为所有当前检测结果创建新轨迹。
                CreateInitialTracks(detections);
            }
            
            RemoveOldTrackedObjects();
        }

        /// <summary>
        /// 为未匹配到任何现有跟踪对象的检测结果添加新轨迹。
        /// 在分配阶段之后调用。
        /// </summary>
        private void AddUntrackedObjects<T>(List<T> detections, HashSet<int> assignedDetections) where T : IDetection
        {
            // 添加新的未跟踪对象     
            for (int i = 0; i < detections.Count; i++)
            {
                if (assignedDetections.Contains(i) is false)
                {
                    _nextId++;

                    detections[i].Id = _nextId;
                    _trackedObjects[_nextId] = new TrackedObject(detections[i], _tailLength);
                }
            }
        }

        /// <summary>
        /// 将新的未跟踪对象添加到跟踪器。
        /// 当跟踪器为空且所有检测结果都被视为新对象时使用。
        /// </summary>
        public void CreateInitialTracks<T>(List<T> detections) where T : IDetection
        {
            foreach (var detection in detections)
            {
                _nextId++;

                detection.Id = _nextId;
                _trackedObjects[_nextId] = new TrackedObject(detection, _tailLength);
            }
        }

        /// <summary>
        /// 移除超过 <c>_maxAge</c> 帧未匹配的跟踪对象。
        /// </summary>
        private void RemoveOldTrackedObjects()
        {
            foreach (var track in _trackedObjects.ToList())
            {
                track.Value.Age++;

                if (track.Value.Age > _maxAge)
                {
                    _trackedObjects.Remove(track.Key);
                }
            }
        }

        /// <summary>
        /// 基于 IoU 和归一化中心点距离，
        /// 计算活动轨迹与当前检测结果之间的成本矩阵。
        /// </summary>
        private static float[,] CalculateCostMatrix<T>(List<KeyValuePair<int, TrackedObject>> activeTracks, List<T> detections) where T : IDetection
        {
            // Define array for storing costMatrix
            var costMatrix = new float[activeTracks.Count, detections.Count];

            for (int i = 0; i < activeTracks.Count; i++)
            {
                var track = activeTracks[i].Value;
                var predictedState = track.GetPredictedState();

                float predictedX = predictedState[0];
                float predictedY = predictedState[1];

                for (int j = 0; j < detections.Count; j++)
                {
                    var detectionBox = detections[j].BoundingBox;
                    var iouCost = 1 - YoloCore.CalculateIoU(track.BoundingBox.BoundingBox, detectionBox);

                    var distance = CalculateDistance(predictedX, predictedY, detectionBox.MidX, detectionBox.MidY);
                    var distanceCost = distance / 100.0f; // Normalize distance

                    costMatrix[i, j] = iouCost + distanceCost;
                }
            }

            return costMatrix;
        }

        /// <summary>
        /// 使用 LAPJV 分配算法根据提供的成本矩阵将当前检测结果匹配到现有的活动轨迹。
        /// 更新匹配检测的跟踪对象，重置其年龄，
        /// 并返回分配的检测索引集合。
        /// </summary>
        private HashSet<int> MatchPredictedObjects<T>(List<T> detections, List<KeyValuePair<int, TrackedObject>> activeTracks, float[,] costMatrix) where T : IDetection
        {
            // Solve assignment using LAPJV algorithm.
            float[,] originalCostMatrix = (float[,])costMatrix.Clone();
            var assignment = LAPJV.Solve(costMatrix);

            // Assign detections to existing tracks
            var assignedDetections = new HashSet<int>();

            for (int i = 0; i < assignment.Length; i++)
            {
                var trackId = activeTracks[i].Key;
                var trackedBox = activeTracks[i].Value;

                int detectionIndex = assignment[i];

                // If a tracked Id is not found, add to age and remove if it's not detected in the last x frames
                if (detectionIndex == -1)
                    continue;

                var cost = originalCostMatrix[i, detectionIndex];

                if (cost < _costThreshold) // Instead of < 0.2f
                {
                    var box = detections[detectionIndex];

                    // Set Id of detected boundingbox
                    box.Id = trackId;

                    // Reset tracked box age counter
                    trackedBox.Age = 0;
                    trackedBox.TrackBoundingBox(box);

                    assignedDetections.Add(detectionIndex);
                }
            }

            return assignedDetections;
        }
        /// <summary>
        /// 计算两个边界框中心之间的距离。
        /// </summary>
        private static float CalculateDistance(float predictedX, float predictedY, float detectionX, float detectionY)
        {
            var dx = predictedX - detectionX;
            var dy = predictedY - detectionY;
            return (float)Math.Sqrt(dx * dx + dy * dy); // Euclidean distance
        }
    }
}
