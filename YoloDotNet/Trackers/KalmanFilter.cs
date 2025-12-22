// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Trackers
{
    public class KalmanFilter(float x, float y)
    {
        // [x, y, dx, dy] → 位置和速度
        private float[] _state = [x, y, 0, 0];

        // 对我们猜测的置信度
        private float[,] _covariance = new float[,]
        {
            { 2, 0, 0, 0 },    // x (更确定)
            { 0, 2, 0, 0 },    // y (更确定)
            { 0, 0, 50, 0 },   // dx (非常不确定)
            { 0, 0, 0, 50 }    // dy (非常不确定)
        };

        // 这些控制我们对运动和测量的信任程度
        private readonly float[,] _measurementNoise = new float[,]
        {
            { 0.05f, 0 },
            { 0, 0.05f }
        };

        private readonly float _sigmaP = 0.5f;  // 位置噪声
        private readonly float _sigmaV = 0.1f;  // 速度噪声
        private readonly float _dt = 1f / 30f;  // 时间步长（30 FPS 下的 1 帧）

        /// <summary>
        /// 基于运动预测下一个状态（位置和速度）。
        /// </summary>
        public void Predict()
        {
            // 1. 使用速度猜测下一个位置
            _state[0] += _state[2] * _dt; // x = x + dx * dt
            _state[1] += _state[3] * _dt; // y = y + dy * dt

            // 2. 创建转移矩阵 F（状态如何随时间变化）
            float[,] f = new float[,]
            {
                { 1, 0, _dt, 0 },
                { 0, 1, 0, _dt },
                { 0, 0, 1, 0 },
                { 0, 0, 0, 1 }
            };

            // 3. 计算新的协方差：P = F * P * F^T + Q
            float[,] newCov = Multiply(Multiply(f, _covariance), Transpose(f));
            float[,] q = GetProcessNoise();
            _covariance = Add(newCov, q);
        }

        /// <summary>
        /// 使用新的测量值更新预测状态。
        /// </summary>
        /// <param name="x">测量的 X 位置。</param>
        /// <param name="y">测量的 Y 位置。</param>
        public void Update(float x, float y)
        {
            // 测量向量 z = [x, y]
            float[] z = new float[] { x, y };

            // 测量矩阵 H（我们只测量 x 和 y）
            float[,] h = new float[,]
            {
                { 1, 0, 0, 0 },
                { 0, 1, 0, 0 }
            };

            // 创新：y = z - H * x
            float[] yVec = Subtract(z, Multiply(h, _state));

            // S = H * P * H^T + R（测量预测）
            float[,] s = Add(Multiply(Multiply(h, _covariance), Transpose(h)), _measurementNoise);

            // 卡尔曼增益 K = P * H^T * S^-1
            float[,] k = Multiply(Multiply(_covariance, Transpose(h)), Inverse2x2(s));

            // 新状态 = 旧状态 + K * 残差
            float[] correction = Multiply(k, yVec);
            _state = Add(_state, correction);

            // 新协方差：(I - K * H) * P
            float[,] kh = Multiply(k, h);
            float[,] i = Identity(4);
            float[,] temp = Subtract(i, kh);
            _covariance = Multiply(temp, _covariance);
        }

        /// <summary>
        /// 返回当前估计状态 [x, y, dx, dy]。
        /// </summary>
        public float[] GetState() => _state;

        /// <summary>
        /// 计算过程噪声矩阵（来自运动的不确定性）。
        /// </summary>
        private float[,] GetProcessNoise()
        {
            float p2 = _sigmaP * _sigmaP;
            float v2 = _sigmaV * _sigmaV;

            return new float[,]
            {
                { p2, 0, _sigmaP * _dt, 0 },
                { 0, p2, 0, _sigmaP * _dt },
                { _sigmaP * _dt, 0, v2, 0 },
                { 0, _sigmaP * _dt, 0, v2 }
            };
        }

        /// <summary>
        /// 矩阵与向量相乘。
        /// </summary>
        private static float[] Multiply(float[,] mat, float[] vec)
        {
            float[] result = new float[mat.GetLength(0)];
            for (int i = 0; i < result.Length; i++)
                for (int j = 0; j < vec.Length; j++)
                    result[i] += mat[i, j] * vec[j];
            return result;
        }

        /// <summary>
        /// 两个矩阵相乘。
        /// </summary>
        private static float[,] Multiply(float[,] a, float[,] b)
        {
            int rows = a.GetLength(0);
            int cols = b.GetLength(1);
            int shared = a.GetLength(1);
            float[,] result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    for (int k = 0; k < shared; k++)
                        result[i, j] += a[i, k] * b[k, j];

            return result;
        }

        /// <summary>
        /// 两个矩阵相加。
        /// </summary>
        private static float[,] Add(float[,] a, float[,] b)
        {
            int rows = a.GetLength(0), cols = a.GetLength(1);
            float[,] result = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = a[i, j] + b[i, j];
            return result;
        }

        /// <summary>
        /// 两个向量相加。
        /// </summary>
        private static float[] Add(float[] a, float[] b)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
                result[i] = a[i] + b[i];
            return result;
        }

        /// <summary>
        /// 向量减法。
        /// </summary>
        private static float[] Subtract(float[] a, float[] b)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
                result[i] = a[i] - b[i];
            return result;
        }

        /// <summary>
        /// 矩阵减法。
        /// </summary>
        private static float[,] Subtract(float[,] a, float[,] b)
        {
            int rows = a.GetLength(0), cols = a.GetLength(1);
            float[,] result = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = a[i, j] - b[i, j];
            return result;
        }

        /// <summary>
        /// 转置矩阵（行变为列）。
        /// </summary>
        private static float[,] Transpose(float[,] mat)
        {
            int rows = mat.GetLength(0), cols = mat.GetLength(1);
            float[,] result = new float[cols, rows];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[j, i] = mat[i, j];
            return result;
        }

        /// <summary>
        /// 计算 2x2 矩阵的逆。
        /// </summary>
        private static float[,] Inverse2x2(float[,] mat)
        {
            float a = mat[0, 0], b = mat[0, 1];
            float c = mat[1, 0], d = mat[1, 1];
            float det = a * d - b * c;
            if (Math.Abs(det) < 1e-6f)
                throw new Exception("矩阵是奇异的，无法求逆。");
            float invDet = 1f / det;

            return new float[,]
            {
                { d * invDet, -b * invDet },
                { -c * invDet, a * invDet }
            };
        }

        /// <summary>
        /// 创建给定大小的单位矩阵。
        /// </summary>
        private static float[,] Identity(int size)
        {
            float[,] result = new float[size, size];
            for (int i = 0; i < size; i++)
                result[i, i] = 1;
            return result;
        }
    }
}
