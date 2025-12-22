// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Utils
{
    /// <summary>
    /// 使用双线性插值和 AVX2 加速调整 Gray8 图像的大小。
    /// 源图像和目标图像必须是 Gray8 格式。
    /// 
    /// 该方法严重依赖 SIMD（单指令，多数据）来使用单条指令
    /// 一次处理多个像素。
    /// 
    /// 使用 AVX2（高级向量扩展 2），使 CPU 能够进行 256 位宽的 SIMD 数学运算
    /// —— 一次对多个值进行超快数学运算。
    /// </summary>
    public static unsafe class Avx2LinearResizer
    {
        public static void ScalePixels(SKBitmap src, SKBitmap dst)
        {
            if (src.ColorType != SKColorType.Gray8 || dst.ColorType != SKColorType.Gray8)
                throw new YoloDotNetException("源图像和目标图像都必须是 Gray8 格式。");

            if (!Avx2.IsSupported)
                throw new PlatformNotSupportedException("需要 AVX2 支持。");

            int srcW = src.Width;
            int srcH = src.Height;
            int dstW = dst.Width;
            int dstH = dst.Height;

            Span<byte> srcSpan = src.GetPixelSpan();
            Span<byte> dstSpan = dst.GetPixelSpan();

            int srcStride = src.RowBytes;
            int dstStride = dst.RowBytes;

            float scaleX = (float)srcW / dstW;
            float scaleY = (float)srcH / dstH;

            // 预计算水平映射
            int[] x0s = new int[dstW];
            int[] x1s = new int[dstW];
            float[] wxs = new float[dstW];

            for (int dx = 0; dx < dstW; dx++)
            {
                float sx = dx * scaleX;
                int x0 = (int)sx;
                int x1 = Math.Min(x0 + 1, srcW - 1);
                x0s[dx] = x0;
                x1s[dx] = x1;
                wxs[dx] = sx - x0;
            }

            fixed (byte* pSrc = srcSpan)
            fixed (byte* pDst = dstSpan)
            fixed (int* px0s = x0s)
            fixed (int* px1s = x1s)
            fixed (float* pwxs = wxs)
            {
                for (int dy = 0; dy < dstH; dy++)
                {
                    float sy = dy * scaleY;
                    int y0 = (int)sy;
                    int y1 = Math.Min(y0 + 1, srcH - 1);
                    float wy = sy - y0;

                    int srcRow0 = y0 * srcStride;
                    int srcRow1 = y1 * srcStride;
                    int dstRow = dy * dstStride;

                    var wyVec = Vector256.Create(wy);

                    int dx = 0;
                    while (dx + 32 <= dstW)
                    {
                        // 检查此 32 像素块的 x0 和 x1 是否连续
                        if (IsContiguous(px0s + dx, 32) && IsContiguous(px1s + dx, 32))
                        {
                            // SIMD 路径
                            byte* topLeft = pSrc + srcRow0 + px0s[dx];
                            byte* topRight = pSrc + srcRow0 + px1s[dx];
                            byte* bottomLeft = pSrc + srcRow1 + px0s[dx];
                            byte* bottomRight = pSrc + srcRow1 + px1s[dx];

                            Vector256<byte> p00 = Avx.LoadVector256(topLeft);
                            Vector256<byte> p10 = Avx.LoadVector256(topRight);
                            Vector256<byte> p01 = Avx.LoadVector256(bottomLeft);
                            Vector256<byte> p11 = Avx.LoadVector256(bottomRight);

                            var wx0 = Avx.LoadVector256(pwxs + dx);
                            var wx1 = Avx.LoadVector256(pwxs + dx + 8);
                            var wx2 = Avx.LoadVector256(pwxs + dx + 16);
                            var wx3 = Avx.LoadVector256(pwxs + dx + 24);

                            var p00f = BytesToFloats(p00);
                            var p10f = BytesToFloats(p10);
                            var p01f = BytesToFloats(p01);
                            var p11f = BytesToFloats(p11);

                            var iTop = new Vector256<float>[4];
                            var iBottom = new Vector256<float>[4];

                            for (int i = 0; i < 4; i++)
                            {
                                var deltaTop = Avx.Subtract(p10f[i], p00f[i]);
                                var deltaBottom = Avx.Subtract(p11f[i], p01f[i]);
                                var wxVec = i switch
                                {
                                    0 => wx0,
                                    1 => wx1,
                                    2 => wx2,
                                    3 => wx3,
                                    _ => wx0
                                };

                                iTop[i] = Fma.IsSupported
                                    ? Fma.MultiplyAdd(deltaTop, wxVec, p00f[i])
                                    : Avx.Add(p00f[i], Avx.Multiply(deltaTop, wxVec));

                                iBottom[i] = Fma.IsSupported
                                    ? Fma.MultiplyAdd(deltaBottom, wxVec, p01f[i])
                                    : Avx.Add(p01f[i], Avx.Multiply(deltaBottom, wxVec));
                            }

                            for (int i = 0; i < 4; i++)
                            {
                                var delta = Avx.Subtract(iBottom[i], iTop[i]);
                                var interp = Fma.IsSupported
                                    ? Fma.MultiplyAdd(delta, wyVec, iTop[i])
                                    : Avx.Add(iTop[i], Avx.Multiply(delta, wyVec));

                                Store8FloatsToBytes(interp, pDst + dstRow + dx + i * 8);
                            }

                            dx += 32;
                        }
                        else
                        {
                            // 非连续：回退到标量处理
                            for (int i = 0; i < 32; i++)
                            {
                                int px = dx + i;
                                int x0 = px0s[px];
                                int x1 = px1s[px];
                                float wx = pwxs[px];

                                byte p00 = pSrc[srcRow0 + x0];
                                byte p10 = pSrc[srcRow0 + x1];
                                byte p01 = pSrc[srcRow1 + x0];
                                byte p11 = pSrc[srcRow1 + x1];

                                float iTop = p00 + (p10 - p00) * wx;
                                float iBottom = p01 + (p11 - p01) * wx;
                                float iValue = iTop + (iBottom - iTop) * wy;

                                pDst[dstRow + px] = (byte)(iValue + 0.5f);
                            }
                            dx += 32;
                        }
                    }

                    // 剩余像素
                    for (; dx < dstW; dx++)
                    {
                        int x0 = px0s[dx];
                        int x1 = px1s[dx];
                        float wx = pwxs[dx];

                        byte p00 = pSrc[srcRow0 + x0];
                        byte p10 = pSrc[srcRow0 + x1];
                        byte p01 = pSrc[srcRow1 + x0];
                        byte p11 = pSrc[srcRow1 + x1];

                        float iTop = p00 + (p10 - p00) * wx;
                        float iBottom = p01 + (p11 - p01) * wx;
                        float iValue = iTop + (iBottom - iTop) * wy;

                        pDst[dstRow + dx] = (byte)(iValue + 0.5f);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsContiguous(int* ptr, int length)
        {
            for (int i = 1; i < length; i++)
                if (ptr[i] != ptr[i - 1] + 1)
                    return false;
            return true;
        }

        private static Vector256<float>[] BytesToFloats(Vector256<byte> vec)
        {
            // 将 256 位向量分割为 2 个 128 位通道
            Vector128<byte> low = vec.GetLower();
            Vector128<byte> high = vec.GetUpper();

            // 将字节解包为 ushort（零扩展）
            Vector128<ushort> lowLo = Sse2.UnpackLow(low, Vector128<byte>.Zero).AsUInt16();
            Vector128<ushort> lowHi = Sse2.UnpackHigh(low, Vector128<byte>.Zero).AsUInt16();
            Vector128<ushort> highLo = Sse2.UnpackLow(high, Vector128<byte>.Zero).AsUInt16();
            Vector128<ushort> highHi = Sse2.UnpackHigh(high, Vector128<byte>.Zero).AsUInt16();

            // 将 ushort 解包为 uint（零扩展）
            Vector128<uint> lowLoLo = Sse2.UnpackLow(lowLo, Vector128<ushort>.Zero).AsUInt32();
            Vector128<uint> lowLoHi = Sse2.UnpackHigh(lowLo, Vector128<ushort>.Zero).AsUInt32();
            Vector128<uint> lowHiLo = Sse2.UnpackLow(lowHi, Vector128<ushort>.Zero).AsUInt32();
            Vector128<uint> lowHiHi = Sse2.UnpackHigh(lowHi, Vector128<ushort>.Zero).AsUInt32();
            Vector128<uint> highLoLo = Sse2.UnpackLow(highLo, Vector128<ushort>.Zero).AsUInt32();
            Vector128<uint> highLoHi = Sse2.UnpackHigh(highLo, Vector128<ushort>.Zero).AsUInt32();
            Vector128<uint> highHiLo = Sse2.UnpackLow(highHi, Vector128<ushort>.Zero).AsUInt32();
            Vector128<uint> highHiHi = Sse2.UnpackHigh(highHi, Vector128<ushort>.Zero).AsUInt32();

            // 将对组合为 256 位向量
            Vector256<uint> vec0 = Vector256.Create(lowLoLo, lowLoHi);
            Vector256<uint> vec1 = Vector256.Create(lowHiLo, lowHiHi);
            Vector256<uint> vec2 = Vector256.Create(highLoLo, highLoHi);
            Vector256<uint> vec3 = Vector256.Create(highHiLo, highHiHi);

            // 将 uint 转换为 float 向量
            Vector256<float> f0 = Avx.ConvertToVector256Single(vec0.AsInt32());
            Vector256<float> f1 = Avx.ConvertToVector256Single(vec1.AsInt32());
            Vector256<float> f2 = Avx.ConvertToVector256Single(vec2.AsInt32());
            Vector256<float> f3 = Avx.ConvertToVector256Single(vec3.AsInt32());

            return [f0, f1, f2, f3];
        }

        private static void Store8FloatsToBytes(Vector256<float> vec, byte* dst)
        {
            var rounded = Avx.Add(vec, Vector256.Create(0.5f));
            var intVec = Avx.ConvertToVector256Int32WithTruncation(rounded);

            Vector128<int> lower = intVec.GetLower();
            Vector128<int> upper = intVec.GetUpper();

            Vector128<short> packed16 = Sse2.PackSignedSaturate(lower, upper);
            Vector128<byte> packed8 = Sse2.PackUnsignedSaturate(packed16, packed16);

            for (int i = 0; i < 8; i++)
                dst[i] = packed8.GetElement(i);
        }
    }
}
