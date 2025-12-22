using YoloDotNet.Models;

namespace PoseEstimationDemo
{
    /// <summary>
    /// 演示配置自定义关键点标记配置文件，包含自定义颜色和关键点连接说明。
    /// </summary>
    public static class CustomKeyPointColorMap
    {
        /// <summary>
        /// 关键点必须与训练模型中的类别具有完全相同的顺序（！）。
        /// </summary>
        private enum KeyPointType
        {
            Nose,
            LeftEye,
            RightEye,
            LeftEar,
            RightEar,
            LeftShoulder,
            RightShoulder,
            LeftElbow,
            RightElbow,
            LeftWrist,
            RightWrist,
            LeftHip,
            RightHip,
            LeftKnee,
            RightKnee,
            LeftAnkle,
            RightAnkle
        }

        /// <summary>
        /// 用于标识十六进制颜色的颜色名称
        /// </summary>
        private enum KeyPointColor
        {
            Green,
            LightBlue,
            Yellow,
            HotPink
        }

        /// <summary>
        /// 命名的十六进制颜色
        /// </summary>
        private static Dictionary<KeyPointColor, string> Colors => new()
        {
            { KeyPointColor.Green, "#A2FF33" },     // 浅绿色
            { KeyPointColor.LightBlue, "#33ACFF" }, // 浅蓝色
            { KeyPointColor.Yellow, "#FFF633" },    // 黄色
            { KeyPointColor.HotPink, "#FF33AC" }    // 亮粉色
        };

        ///// <summary>
        ///// 关键点选项。
        ///// </summary>
        //public static PoseDrawingOptions KeyPointOptions => new()
        //{
        //    PoseConfidence = 0.65,
        //    KeyPointMarkers = KeyPointMapping
        //};

        #region 配置自定义关键点及其连接的方法
        /// <summary>
        /// 配置关键点连接和要使用的颜色。
        /// </summary>
        public static KeyPointMarker[] KeyPoints =>
        [
            new () // 鼻子
            {
                Color = Colors[KeyPointColor.Green],
                Connections =
                [
                    new ((int)KeyPointType.LeftEye, Colors[KeyPointColor.Green]),
                    new ((int)KeyPointType.RightEye, Colors[KeyPointColor.Green])
                ]
            },
            new () // 左眼
            {
                Color = Colors[KeyPointColor.Green],
                Connections = [ new ((int)KeyPointType.RightEye, Colors[KeyPointColor.Green]) ]
            },
            new () // 右眼
            {
                Color = Colors[KeyPointColor.Green],
            },
            new () // 左耳
            {
                Color = Colors[KeyPointColor.Green],
                Connections =
                [
                    new ((int)KeyPointType.LeftEye, Colors[KeyPointColor.Green]),
                    new ((int)KeyPointType.LeftShoulder, Colors[KeyPointColor.Green]),
                ]
            },
            new () // 右耳
            {
                Color = Colors[KeyPointColor.Green],
                Connections =
                [
                    new ((int)KeyPointType.RightEye, Colors[KeyPointColor.Green]),
                    new ((int)KeyPointType.RightShoulder, Colors[KeyPointColor.Green]),
                ]
            },
            new () // 左肩
            {
                Color = Colors[KeyPointColor.LightBlue],
                Connections =
                [
                    new ((int)KeyPointType.RightShoulder, Colors[KeyPointColor.LightBlue]),
                    new ((int)KeyPointType.LeftElbow, Colors[KeyPointColor.LightBlue]),
                    new ((int)KeyPointType.LeftHip, Colors[KeyPointColor.HotPink])
                ]
            },
            new () // 右肩
            {
                Color = Colors[KeyPointColor.LightBlue],
                Connections =
                [
                    new ((int)KeyPointType.RightElbow, Colors[KeyPointColor.LightBlue]),
                    new ((int)KeyPointType.RightHip, Colors[KeyPointColor.HotPink])
                ]
            },
            new () // 左肘
            {
                Color = Colors[KeyPointColor.LightBlue],
                Connections = [ new ((int)KeyPointType.LeftWrist, Colors[KeyPointColor.LightBlue]) ]
            },
            new () // 右肘
            {
                Color = Colors[KeyPointColor.LightBlue],
                Connections = [ new ((int)KeyPointType.RightWrist, Colors[KeyPointColor.LightBlue]) ]
            },
            new () // 左手腕
            {
                Color = Colors[KeyPointColor.LightBlue]
            },
            new () // 右手腕
            {
                Color = Colors[KeyPointColor.LightBlue]
            },
            new () // 左髋
            {
                Color = Colors[KeyPointColor.Yellow],
                Connections =
                [
                    new ((int)KeyPointType.RightHip, Colors[KeyPointColor.HotPink]),
                    new ((int)KeyPointType.LeftKnee, Colors[KeyPointColor.Yellow])
                ]
            },
            new () // 右髋
            {
                Color = Colors[KeyPointColor.Yellow],
                Connections = [ new ((int)KeyPointType.RightKnee, Colors[KeyPointColor.Yellow]) ]
            },
            new () // 左膝
            {
                Color = Colors[KeyPointColor.Yellow],
                Connections = [ new ((int)KeyPointType.LeftAnkle, Colors[KeyPointColor.Yellow]) ]
            },
            new () // 右膝
            {
                Color = Colors[KeyPointColor.Yellow],
                Connections = [ new ((int)KeyPointType.RightAnkle, Colors[KeyPointColor.Yellow]) ]
            },
            new () // 左踝
            {
                Color = Colors[KeyPointColor.Yellow]
            },
            new () // 右踝
            {
                Color = Colors[KeyPointColor.Yellow]
            }
        ];
        #endregion

    }
}