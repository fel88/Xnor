using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;

namespace xnor
{


    public class binop
    {


        public static void THNN_Bin_SpatialConvolutionMM_updateOutput_frame(
                                                                     IntPtr output,//float
                                                                     IntPtr weight,//int
                                                                     IntPtr bias,//float
                                                                     IntPtr ones,//float
                                                                     IntPtr bin_col,//int
                                                                     IntPtr alphas,//float
                                                                     int kW, int kH,
                                                                     int dW, int dH,
                                                                     int padW, int padH,
                                                                     Int64 nInputPlane,
                                                                     Int64 inputWidth, Int64 inputHeight,
                                                                     Int64 nOutputPlane,
                                                                     Int64 outputWidth, Int64 outputHeight,
                                                                     bool quantOutput = false)
        {
            IntPtr output2d;

            //var   output2d = THFloatTensor_newWithStorage2d(output->storage, output->storageOffset, nOutputPlane, -1, outputHeight * outputWidth, -1);
            //var output2d = THFloatTensor_newWithStorage2d(output, (int)nOutputPlane, -1, (int)(outputHeight * outputWidth), -1);



            var strg = THWrapper.THFloatTensor_storage(output);
            var offset = THWrapper.THFloatTensor_storageOffset(output);
            output2d = THWrapper.THFloatTensor_newWithStorage2d(strg, offset, nOutputPlane, (long)-1, outputWidth * outputHeight, (long)-1);
            //InternalArray output2d = new InternalArray(new int[] { });
            THWrapper.THFloatTensor_zero(output2d);

            binary_gemm_cpu(weight, bin_col, output2d, (int)nOutputPlane, (int)(kW * kH * nInputPlane), (int)(outputHeight * outputWidth), 0, 1, 1, alphas, quantOutput);

            if (bias != null && THWrapper.THFloatTensor_nDimension(bias) != 0)
            {
                THWrapper.THFloatTensor_addmm(output2d, 1, output2d, 1, bias, ones);
                //THFloatTensor_addmm(output2d, 1, output2d, 1, bias, ones);
            }
            THWrapper.THFloatTensor_free(output2d);

            //THWrapper.THFloatTensor_free(_ones);
            //THFloatTensor_free(output2d);
        }


        public static void THNN_unfolded_copy(
                                IntPtr columns,
                                IntPtr input,
                                int kW, int kH,
                                int dW, int dH,
                                int padW, int padH,
                                int nInputPlane,
                                int inputWidth, int inputHeight,
                                int outputWidth, int outputHeight)
        {
            // This function assumes that
            // kH*kW does not overflow an int
            // nInputPlane*kH*kW does not overflow a int64_t
            // outputHeight*dH does not overflow a int64_t
            // outputWidth*dW does not overflow a int64_t

            //            int64_t k;
            var input_data = THWrapper.THFloatTensor_data(input);
            var columns_data = THWrapper.THFloatTensor_data(columns);

            for (Int64 k = 0; k < (Int64)nInputPlane * kH * kW; k++)
            {
                /*  float[] dats=new float[30];
                  float[] dats2=new float[30];
                  for (int ii = 0; ii < 30; ii++)
                  {
                      dats[ii]= matmul.GetFloat(columns_data, ii);
                  }
                  for (int ii = 0; ii < 30; ii++)
                  {
                      dats2[ii] = matmul.GetFloat(input_data, ii);
                  }*/
                Int64 nip = k / (kH * kW);
                Int64 rest = k % (kH * kW);
                Int64 kh = rest / kW;
                Int64 kw = rest % kW;
                int x, y;
                Int64 ix, iy;
                var sh1 = (int)(nip * (kH * kW * outputHeight * outputWidth) + kh * (kW * outputHeight * outputWidth) + kw * (outputHeight * outputWidth));
                var sh2 = +(int)(nip * (inputHeight * inputWidth));
                //IntPtr dst = columns_data + (int)(nip * (kH * kW * outputHeight * outputWidth) + kh * (kW * outputHeight * outputWidth) + kw * (outputHeight * outputWidth));
                //IntPtr src = input_data + (int)(nip * (inputHeight * inputWidth));
                IntPtr dst = columns_data + 4 * sh1;
                IntPtr src = input_data + 4 * sh2;
                //                float* dst = columns_data + nip * ((size_t)kH * kW * outputHeight * outputWidth) + kh * ((size_t)kW * outputHeight * outputWidth) + kw * ((size_t)outputHeight * outputWidth);
                //                float* src = input_data + nip * ((size_t)inputHeight * inputWidth);
                if (padW > 0 || padH > 0)
                {
                    Int64 lpad, rpad;
                    for (y = 0; y < outputHeight; y++)
                    {
                        iy = (Int64)y * dH - padH + kh;
                        if (iy < 0 || iy >= inputHeight)
                        {

                            memset(dst + 4 * (y * outputWidth), 0, sizeof(float) * outputWidth);
                        }
                        else
                        {
                            if (dW == 1)
                            {
                                ix = 0 - padW + kw;
                                lpad = Math.Max(0, padW - kw);
                                rpad = Math.Max(0, padW - (kW - kw - 1));
                                if (outputWidth - rpad - lpad <= 0)
                                {
                                    memset(dst + 4 * (y * outputWidth), 0, sizeof(float) * outputWidth);
                                }
                                else
                                {
                                    if (lpad > 0) memset(dst + 4 * (y * outputWidth), 0, sizeof(float) * lpad);
                                    memcpy(dst + 4 * ((int)(y * outputWidth + lpad)), src + 4 * ((int)(iy * inputWidth + ix + lpad)), sizeof(float) * (outputWidth - rpad - lpad));
                                    if (rpad > 0) memset(dst + 4 * (int)(y * outputWidth + outputWidth - rpad), 0, sizeof(float) * rpad);
                                }
                            }
                            else
                            {
                                for (x = 0; x < outputWidth; x++)
                                {
                                    ix = (Int64)x * dW - padW + kw;
                                    if (ix < 0 || ix >= inputWidth)
                                        memset(dst + 4 * (y * outputWidth + x), 0, sizeof(float) * 1);
                                    else
                                        memcpy(dst + 4 * (y * outputWidth + x), src + 4 * (int)(iy * inputWidth + ix), sizeof(float) * (1));
                                }
                            }
                        }
                    }
                }
                else
                {
                    for (y = 0; y < outputHeight; y++)
                    {
                        iy = (Int64)y * dH + kh;
                        ix = 0 + kw;
                        if (dW == 1)
                        {
                            memcpy(dst + 4 * ((int)y * outputWidth), src + 4 * ((int)(iy * inputWidth + ix)), sizeof(float) * outputWidth);
                            /* for (int ii = 0; ii < 30; ii++)
                             {
                                 dats[ii] = matmul.GetFloat(columns_data, ii);
                             }*/
                        }
                        else
                        {
                            for (x = 0; x < outputWidth; x++)
                            {
                                memcpy(dst + 4 * (y * outputWidth + x), src + 4 * ((int)(iy * inputWidth + ix + x * dW)), sizeof(float) * (1));
                            }
                        }
                    }
                }
            }
        }

        public static void memset(IntPtr dst, byte val, long size)
        {
            byte[] temp = new byte[size];
            Marshal.Copy(dst, temp, 0, temp.Length);
            for (int i = 0; i < temp.Length; i++)
            {
                temp[i] = val;
            }
            Marshal.Copy(temp, 0, dst, (int)size);
        }

        public static void memcpy(IntPtr dst, IntPtr src, long size)
        {
            byte[] temp = new byte[size];
            Marshal.Copy(src, temp, 0, temp.Length);
            Marshal.Copy(temp, 0, dst, (int)size);
        }

        public static void memset(float[] a1, long dst, byte val, long size)
        {
            for (int i = 0; i < size / 4; i++)
            {
                a1[i + dst] = val;
            }
        }
        /*public static void memcpy(float[] a1, long dst, float[] a2, long dst2, long size)
        {
            for (int i = 0; i < size / 4; i++)
            {
                a1[i + dst] = a2[i + dst2];
            }
        }*/
        /*
            THFloatTensor *input,
                                                THFloatTensor *output,
                                                THIntTensor *weight,
                                                THFloatTensor *bias,
                                                THFloatTensor *columns,
                                                THFloatTensor *alphas,
                                                int kH, int kW,
                                                int dH, int dW,
                                                int padH, int padW)*/
        public static void THNN_Bin_SpatialConvolutionMM_updateOutput(
            IntPtr input,
            IntPtr output,
            IntPtr weight,
            IntPtr bias,
            IntPtr columns,
            IntPtr alphas,
               int kH, int kW,
                                                int dH, int dW,
                                                int padH, int padW
            , bool quantOutput = false)
        {
            int ndim = THWrapper.THFloatTensor_nDimension(input);
            int dimf = 0;
            int dimh = 1;
            int dimw = 2;

            if (ndim == 4)
            {
                dimf++;
                dimh++;
                dimw++;
            }
            var nInputPlane = THWrapper.THFloatTensor_size(input, dimf);
            var inputHeight = THWrapper.THFloatTensor_size(input, dimh);
            var inputWidth = THWrapper.THFloatTensor_size(input, dimw);

            var nOutputPlane = THWrapper.THFloatTensor_size(weight, 0);
            var outputHeight = (inputHeight + 2 * padH - kH) / dH + 1;
            var outputWidth = (inputWidth + 2 * padW - kW) / dW + 1;

            //InternalArray ones = new InternalArray(new int[] { 1 });
            IntPtr ones = THWrapper.THFloatTensor_new();
            if (bias != null && THWrapper.THFloatTensor_nDimension(bias) == 1)
            {
                THWrapper.THFloatTensor_resize2d(bias, THWrapper.THFloatTensor_size(bias, 0), 1);
                //THFloatTensor_resize2d(bias, bias.Shape[0], 1);
            }
            THWrapper.THFloatTensor_resize2d(ones, 1, outputHeight * outputWidth);
            //THFloatTensor_resize2d(ones, 1, outputHeight * outputWidth);
            THWrapper.THFloatTensor_fill(ones, 1);
            //THFloatTensor_fill(ones, 1);

            var T = THWrapper.THFloatTensor_size(input, 0);
            //InternalArray bin_col = new InternalArray(new int[] { 1 });
            var bin_col = THWrapper.THIntTensor_new();

            THWrapper.THFloatTensor_resize4d(output, T, (int)nOutputPlane, outputHeight, outputWidth);
            //THFloatTensor_resize4d(output, T, (int)nOutputPlane, outputHeight, outputWidth);
            THWrapper.THFloatTensor_resize3d(columns, T, kW * kH * nInputPlane, outputHeight * outputWidth);
            //THFloatTensor_resize3d(columns, T, kW * kH * nInputPlane, outputHeight * outputWidth);
            THWrapper.THIntTensor_resize3d(bin_col, T, (int)nOutputPlane, outputHeight * outputWidth);
            //THIntTensor_resize3d(bin_col, T, (int)nOutputPlane, outputHeight * outputWidth);

            for (int t = 0; t < T; t++)
            {
                /*var input_t = input.Get2DImageFrom4DArray(0, t);
                var columns_t = columns.Get2DImageFrom4DArray(0, t);
                var bin_col_t = bin_col.Get2DImageFrom4DArray(0, t);*/


                //var _bin_col = bin_col.ToTHTensor();
                var input_t = THWrapper.THFloatTensor_newSelect(input, 0, t);
                var columns_t = THWrapper.THFloatTensor_newSelect(columns, 0, t);
                var bin_col_t = THWrapper.THIntTensor_newSelect(bin_col, 0, t);
                /* var bbb = InternalArray.FromTHFloatTensor(columns_t);
                 for (int i = 0; i < bbb.Data.Length; i++)
                 {
                     var res = bbb.Data[i];
                     if (res != 0)
                     {

                     }

                 }*/
                THNN_unfolded_copy(
                    columns_t, input_t, kW, kH, dW, dH, padW, padH,
                    (int)nInputPlane, (int)inputWidth, (int)inputHeight, (int)outputWidth, (int)outputHeight
                );
                //  bbb = InternalArray.FromTHFloatTensor(columns_t);

                //Debug.Assert(bbb.Data[1800] == -0.053401);
                //Debug.Assert(bbb.Data[4200] == -0.852360);

                /* for (int i = 0; i < bbb.Data.Length; i++)
                 {
                     var res = bbb.Data[i];
                     if (res != 0)
                     {

                     }
                 }*/
                encode_cols_cpu(columns_t, bin_col_t);
                //  var bb = InternalArray.FromTHIntTensor(bin_col_t);
                /* for (int i = 0; i < bb.IntData.Length; i++)
                 {
                     var res = bb.IntData[i];
                     if (res != 0)
                     {

                     }
                 }*/

            }
            for (int t = 0; t < T; t++)
            {
                /* THFloatTensor* output_t = THFloatTensor_newSelect(output, 0, t);
                 THIntTensor* bin_col_t = THIntTensor_newSelect(bin_col, 0, t);
                */
                /*var output_t = output.Get2DImageFrom4DArray(0, t);
                var bin_col_t = bin_col.Get2DImageFrom4DArray(0, t);*/
                //var _output = output.ToTHTensor();
                //var _bin_col = bin_col.ToTHTensor();
                var output_t = THWrapper.THFloatTensor_newSelect(output, 0, t);
                var bin_col_t = THWrapper.THIntTensor_newSelect(bin_col, 0, t);
                THNN_Bin_SpatialConvolutionMM_updateOutput_frame(
                    output_t, weight, bias, ones, bin_col_t, alphas, kW, kH, dW, dH, padW, padH,
                    nInputPlane, inputWidth, inputHeight, nOutputPlane, outputWidth, outputHeight, quantOutput
                );

            }

        }

        public static void THIntTensor_resize3d(InternalArray bin_col, int size1, int size2, int size3)
        {
            bin_col.IntData = new UInt32[size1 * size2 * size3];
            bin_col.Shape = new int[] { size1, size2, size3 };
        }
        public static void THFloatTensor_resize3d(InternalArray bin_col, int size1, int size2, int size3)
        {
            bin_col.Data = new float[size1 * size2 * size3];
            bin_col.Shape = new int[] { size1, size2, size3 };
        }
        public static void THFloatTensor_resize4d(InternalArray bin_col, int size1, int size2, int size3, int size4)
        {
            bin_col.Data = new float[size1 * size2 * size3 * size4];
            bin_col.Shape = new int[] { size1, size2, size3, size4 };
        }
        public static void THFloatTensor_fill(InternalArray ar, float val)
        {
            for (int i = 0; i < ar.Data.Length; i++)
            {
                ar.Data[i] = val;
            }
        }
        public static void THFloatTensor_zero(InternalArray ar)
        {
            for (int i = 0; i < ar.Data.Length; i++)
            {
                ar.Data[i] = 0;
            }
        }
        internal static InternalArray bin_conv2d(InternalArray input, InternalArray weight, InternalArray bias, InternalArray alpha, int[] kernel_size, int[] stride, int[] padding)
        {


            var col_tensor = THWrapper.THFloatTensor_new();
            var output = THWrapper.THFloatTensor_new();
            var _alpha = alpha.ToTHTensor();
            var _input = input.ToTHTensor();
            var _weight = weight.ToTHTensor();
            IntPtr _bias;
            if (bias == null)
            {
                _bias = THWrapper.THFloatTensor_new();

            }
            else
            {
                _bias = bias.ToTHTensor();
            }
            binop.THNN_Bin_SpatialConvolutionMM_updateOutput(_input, output, _weight, _bias, col_tensor, _alpha,
           kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0], padding[1]);
            THWrapper.THFloatTensor_free(col_tensor);
            var ret = InternalArray.FromTHFloatTensor(output);
            THWrapper.THFloatTensor_free(output);
            THWrapper.THFloatTensor_free(_bias);
            THWrapper.THFloatTensor_free(_input);
            THWrapper.THFloatTensor_free(_alpha);
            return ret;
        }
        internal static InternalArray fpbin_conv2d(InternalArray input, InternalArray weight, InternalArray bias, InternalArray alpha, int[] kernel_size, int[] stride, int[] padding)
        {


            var col_tensor = THWrapper.THFloatTensor_new();
            var output = THWrapper.THFloatTensor_new();
            var _alpha = alpha.ToTHTensor();
            var cln = new InternalArray(input.Shape);
            for (int i = 0; i < cln.Data.Length; i++)
            {
                cln.Data[i] = input.QIntData[i];
            }
            var _input = cln.ToTHTensor();
            var _weight = weight.ToTHTensor();
            IntPtr _bias;
            if (bias == null)
            {
                _bias = THWrapper.THFloatTensor_new();

            }
            else
            {
                _bias = bias.ToTHTensor();
            }
            binop.THNN_Bin_SpatialConvolutionMM_updateOutput(_input, output, _weight, _bias, col_tensor, _alpha,
           kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0], padding[1], true);
            THWrapper.THFloatTensor_free(col_tensor);
            var ret = InternalArray.FromTHFloatTensor(output);
            ret.QIntData = new short[ret.Data.Length];
            for (int i = 0; i < ret.Data.Length; i++)
            {
                ret.QIntData[i] = (short)ret.Data[i];
            }
            ret.Data = null;
            THWrapper.THFloatTensor_free(output);
            THWrapper.THFloatTensor_free(_bias);
            THWrapper.THFloatTensor_free(_input);
            THWrapper.THFloatTensor_free(_alpha);
            return ret;
        }
        public static UInt32 encode_val(IntPtr array, int n)
        {
            UInt32 sign, r = 0;
            for (int i = 0; i < matmul.ENCODE_BIT && i < n; i++)
            {

                sign = (uint)((matmul.GetFloat(array, i) > 0) ? 1 : 0);
                r |= (sign << i);
            }
            return r;
        }
        public static void encode_rows_cpu_kernel(IntPtr columns, IntPtr columns_binary, int m, int n)
        {
            int i, l = 1 + (n - 1) / matmul.ENCODE_BIT;
            //#pragma omp parallel for
            for (i = 0; i < m * l; i++)
            {
                int p = n * (i / l) + matmul.ENCODE_BIT * (i % l);
                matmul.SetUint32(columns_binary, i, encode_val(columns + 4 * p, n - matmul.ENCODE_BIT * (i % l)));
            }
        }
        public static void THIntTensor_resize2d(InternalArray ar, int m, int l)
        {
            ar.IntData = new UInt32[m * l];
            ar.Shape = new int[] { 1, m * l };
        }
        public static void THFloatTensor_resize2d(InternalArray ar, int m, int l)
        {
            ar.Data = new float[m * l];
            ar.Shape = new int[] { 1, m * l };
        }
        public static void encode_cols_cpu(IntPtr input, IntPtr output)//input float, output int
        {
            /*   int n = input->size[0];
               int k = input->size[1];
               int l = 1 + (n - 1) / ENCODE_BIT;

               THIntTensor_resize2d(output, l, k);
               float* a = THFloatTensor_data(input);
               uint32_t* b = (uint32_t*)THIntTensor_data(output);

               encode_cols_cpu_kernel(a, b, n, k);*/

            var n = (int)THWrapper.THFloatTensor_size(input, 0);
            var k = (int)THWrapper.THFloatTensor_size(input, 1);

            int l = 1 + (n - 1) / matmul.ENCODE_BIT;
            THWrapper.THIntTensor_resize2d(output, l, k);

            var a = THWrapper.THFloatTensor_data(input);
            var b = THWrapper.THIntTensor_data(output);
            encode_cols_cpu_kernel(a, b, n, k);

            //var res1 = BitConverter.ToUInt32(BitConverter.GetBytes(THWrapper.THIntTensor_get2d(output, 0, 7)), 0);
            //var res2 = THWrapper.THIntTensor_get2d(output, 7, 0);
        }
        public static void encode_cols_cpu_kernel(IntPtr columns, IntPtr columns_binary, int m, int n)
        {
            int col_bin_m = 1 + (m - 1) / matmul.ENCODE_BIT;
            int i, j, k;
            //#pragma omp parallel for
            for (i = 0; i < col_bin_m; i++)
            {
                int i64 = i * matmul.ENCODE_BIT;
                for (j = 0; j < n && i64 < m; j++)
                {

                    uint sign, rvalue = 0;

                    for (k = 0; j + n * (i64 + k) < m * n && k < matmul.ENCODE_BIT; k++)
                    {
                        sign = (matmul.GetFloat(columns, j + n * (i64 + k)) > 0) ? 1u : 0u;
                        rvalue |= (sign << k);
                    }

                    matmul.SetUint32(columns_binary, j + n * i, rvalue);
                }
            }
        }
        public static void encode_rows_cpu(IntPtr input, IntPtr output)//input float, output int
        {
            var m = (int)THWrapper.THFloatTensor_size(input, 0);
            var n = (int)THWrapper.THFloatTensor_size(input, 1);

            int l = 1 + (n - 1) / matmul.ENCODE_BIT;
            THWrapper.THIntTensor_resize2d(output, m, l);
            var test1 = THWrapper.THIntTensor_nDimension(output);
            var dim0 = THWrapper.THIntTensor_size(output, 0);
            var dim1 = THWrapper.THIntTensor_size(output, 1);
            //THIntTensor_resize2d(output, m, l);
            /*int m = input->size[0];
            int n = input->size[1];
            int l = 1 + (n - 1) / ENCODE_BIT;

            THIntTensor_resize2d(output, m, l);
            float* a = THFloatTensor_data(input);
            uint32_t* b = (uint32_t*)THIntTensor_data(output);
            */

            var a = THWrapper.THFloatTensor_data(input);
            var b = THWrapper.THIntTensor_data(output);
            encode_rows_cpu_kernel(a, b, m, n);
            //var temp = InternalArray.FromTHIntTensor(output);

        }


        /*
         * a -int
         * b- int
         * c- float\alphas-float
         */
        public static void binary_gemm_cpu(IntPtr a, IntPtr b, IntPtr c, int m, int nn, int k, int transb, int beta, int alpha, IntPtr alphas, bool quantOutput = false)
        {

            if (THWrapper.THFloatTensor_nDimension(c) != 2 || THWrapper.THFloatTensor_size(c, 0) * THWrapper.THFloatTensor_size(c, 1) < m * k)
            {
                THWrapper.THFloatTensor_resize2d(c, m, k);
                //THFloatTensor_resize2d(c, m, k);
            }
            /*
            uint32_t* A = (uint32_t*)THIntTensor_data(a);
            uint32_t* B = (uint32_t*)THIntTensor_data(b);
            float* C = THFloatTensor_data(c);
            float* D = THFloatTensor_data(alphas);
            */
            var A = THWrapper.THIntTensor_data(a);
            var B = THWrapper.THIntTensor_data(b);
            var C = THWrapper.THFloatTensor_data(c);
            var D = THWrapper.THFloatTensor_data(alphas);

            var aa = InternalArray.FromTHIntTensor(a);
            var bb = InternalArray.FromTHIntTensor(b);
            var cc = InternalArray.FromTHFloatTensor(c);
            var dd = InternalArray.FromTHFloatTensor(alphas);

            int n = 1 + (nn - 1) / matmul.ENCODE_BIT;
            int brow = transb != 0 ? 1 : k;
            int bcol = transb != 0 ? n : 1;

            //matmul.dgemm_nn(m, k, nn, A, n, 1, B, brow, bcol, C, k, 1, beta, alpha, D);
            //matmul.dgemm_nn(m, k, nn, A, n, 1, B, brow, bcol, C, k, 1, beta, alpha, D);
            matmul.fpdgemm_nn(m, k, nn, A, n, 1, B, brow, bcol, C, k, 1, beta);

            if (alpha != 0)
            {
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < k; j++)
                    {
                        //C[i * n + j] *= alphas[i];
                        var aa1 = matmul.GetFloat(C, i * k + j);
                        short aq1 = (short)(aa1 * 256);

                        var aa2 = matmul.GetFloat(D, i);
                        var aq2 = (short)(aa2 * 256);

                        var val4 = (short)((int)(aq1 * aq2) >> 8);
                        var quant_res = val4 / 256f;
                        var orig = aa1 * aa2;
                        //matmul.SetFloat(C, i * k + j, aa1 * aa2);
                        if (quantOutput)
                        {
                            matmul.SetFloat(C, i * k + j, val4);
                        }
                        else
                        {
                            matmul.SetFloat(C, i * k + j, val4 / 256f);
                        }
                        //C[i * n + j] = (float)(C[i * n + j] * alphas[i]);
                    }
                }
            }

        }
        public static void fpbinary_gemm_cpu(IntPtr a, IntPtr b, IntPtr c, int m, int nn, int k, int transb, int beta, int alpha, IntPtr alphas)
        {

            if (THWrapper.THFloatTensor_nDimension(c) != 2 || THWrapper.THFloatTensor_size(c, 0) * THWrapper.THFloatTensor_size(c, 1) < m * k)
            {
                THWrapper.THFloatTensor_resize2d(c, m, k);
                //THFloatTensor_resize2d(c, m, k);
            }
            /*
            uint32_t* A = (uint32_t*)THIntTensor_data(a);
            uint32_t* B = (uint32_t*)THIntTensor_data(b);
            float* C = THFloatTensor_data(c);
            float* D = THFloatTensor_data(alphas);
            */
            var A = THWrapper.THIntTensor_data(a);
            var B = THWrapper.THIntTensor_data(b);
            var C = THWrapper.THFloatTensor_data(c);
            var D = THWrapper.THFloatTensor_data(alphas);

            var aa = InternalArray.FromTHIntTensor(a);
            var bb = InternalArray.FromTHIntTensor(b);
            var cc = InternalArray.FromTHFloatTensor(c);
            var dd = InternalArray.FromTHFloatTensor(alphas);

            int n = 1 + (nn - 1) / matmul.ENCODE_BIT;
            int brow = transb != 0 ? 1 : k;
            int bcol = transb != 0 ? n : 1;

            //matmul.dgemm_nn(m, k, nn, A, n, 1, B, brow, bcol, C, k, 1, beta, alpha, D);
            //matmul.dgemm_nn(m, k, nn, A, n, 1, B, brow, bcol, C, k, 1, beta, alpha, D);
            matmul.fpdgemm_nn(m, k, nn, A, n, 1, B, brow, bcol, C, k, 1, beta);

            if (alpha != 0)
            {
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < k; j++)
                    {
                        //C[i * n + j] *= alphas[i];
                        var aa1 = matmul.GetFloat(C, i * k + j);
                        short aq1 = (short)(aa1 * 256);

                        var aa2 = matmul.GetFloat(D, i);
                        var aq2 = (short)(aa2 * 256);

                        var val4 = (short)((int)(aq1 * aq2) >> 8);
                        var quant_res = val4 / 256f;
                        var orig = aa1 * aa2;
                        //matmul.SetFloat(C, i * k + j, aa1 * aa2);

                        matmul.SetFloat(C, i * k + j, val4);

                        //C[i * n + j] = (float)(C[i * n + j] * alphas[i]);
                    }
                }
            }

        }
        internal static InternalArray bin_linear(InternalArray input, InternalArray weight, InternalArray bias, InternalArray alpha)
        {
            var m = input.Shape[0];
            var n = input.Shape[1];
            var k = weight.Shape[0];


            /**
             *
             * m = input.data.shape[0]
    n = input.data.shape[1]
    k = weight.data.shape[0]
    out_tensor = torch.FloatTensor()
    bin_input = torch.IntTensor()
    use_cuda = input.is_cuda
               binop.encode_rows_cpu(input.data, bin_input)
                binop.binary_gemm_cpu(bin_input, weight.data, output.data, m, n, k, 1, 0, 0, alpha.data)
    output.data.mul_(alpha.data.t().expand(output.shape))
    if bias is not None:
        output.data.add_(bias.data.expand(output.shape))
    return output

             */
            //InternalArray output = new InternalArray(new int[] { });

            var _input = input.ToTHTensor();
            var bin_input = THWrapper.THIntTensor_new();
            encode_rows_cpu(_input, bin_input);
            var temp = InternalArray.FromTHIntTensor(bin_input);

            var _alpha = alpha.ToTHTensor();
            //var _bin_input = bin_input.ToTHTensor();
            var _weight = weight.ToTHTensor();
            var _output = THWrapper.THFloatTensor_new();
            binop.binary_gemm_cpu(bin_input, _weight, _output, m, n, k, 1, 0, 0, _alpha);

            var temp2 = InternalArray.FromTHFloatTensor(_output);
            THWrapper.THFloatTensor_free(_input);
            THWrapper.THIntTensor_free(bin_input);
            //var tt = alpha.ToTHTensor();
            var ttt = alpha.Transpose2D();

            //var newt=THWrapper.THFloatTensor_newTranspose(tt, 0, 1);

            /*output.data.mul_(alpha.data.t().expand(output.shape))
  */
            if (bias != null)
            {/*
  if bias is not None:
        output.data.add_(bias.data.expand(output.shape))*/
            }
            var output = InternalArray.FromTHFloatTensor(_output);
            for (int i = 0; i < ttt.Data.Length; i++)
            {
                output.Data[i] *= ttt.Data[i];
            }
            THWrapper.THFloatTensor_free(_output);


            return output;
        }

        internal static InternalArray fpbin_linear(InternalArray input, InternalArray weight, InternalArray bias, InternalArray alpha)
        {
            var m = input.Shape[0];
            var n = input.Shape[1];
            var k = weight.Shape[0];


            /**
             *
             * m = input.data.shape[0]
    n = input.data.shape[1]
    k = weight.data.shape[0]
    out_tensor = torch.FloatTensor()
    bin_input = torch.IntTensor()
    use_cuda = input.is_cuda
               binop.encode_rows_cpu(input.data, bin_input)
                binop.binary_gemm_cpu(bin_input, weight.data, output.data, m, n, k, 1, 0, 0, alpha.data)
    output.data.mul_(alpha.data.t().expand(output.shape))
    if bias is not None:
        output.data.add_(bias.data.expand(output.shape))
    return output

             */
            //InternalArray output = new InternalArray(new int[] { });

            var cln = new InternalArray(input.Shape);
            for (int i = 0; i < cln.Data.Length; i++)
            {
                cln.Data[i] = input.QIntData[i];
            }

            var _input = cln.ToTHTensor();
            var bin_input = THWrapper.THIntTensor_new();
            encode_rows_cpu(_input, bin_input);
            //var temp = InternalArray.FromTHIntTensor(bin_input);

            var _alpha = alpha.ToTHTensor();
            //var _bin_input = bin_input.ToTHTensor();
            var _weight = weight.ToTHTensor();
            var _output = THWrapper.THFloatTensor_new();
            binop.fpbinary_gemm_cpu(bin_input, _weight, _output, m, n, k, 1, 0, 0, _alpha);

            var temp2 = InternalArray.FromTHFloatTensor(_output);
            THWrapper.THFloatTensor_free(_input);
            THWrapper.THIntTensor_free(bin_input);
            //var tt = alpha.ToTHTensor();
            var ttt = alpha.Transpose2D();
            ttt.QIntData = new short[ttt.Data.Length];
            for (int i = 0; i < ttt.Data.Length; i++)
            {
                ttt.QIntData[i] = (short)(ttt.Data[i] * 256);
            }
            ttt.Data = null;
            //var newt=THWrapper.THFloatTensor_newTranspose(tt, 0, 1);

            /*output.data.mul_(alpha.data.t().expand(output.shape))
  */
            if (bias != null)
            {
                throw new NotImplementedException();
                /*
                  if bias is not None:
                        output.data.add_(bias.data.expand(output.shape))*/
            }

            var output = InternalArray.FromTHFloatTensor(_output);
            output.QIntData = new short[output.Data.Length];
            for (int i = 0; i < output.QIntData.Length; i++)
            {
                output.QIntData[i] = (short)(output.Data[i] * 256);
            }
            output.Data = null;
            for (int i = 0; i < ttt.QIntData.Length; i++)
            {
                var val4 = (short)((int)(output.QIntData[i] * ttt.QIntData[i]) >> 8);
                //output.QIntData[i] = output.Data[i] * ttt.Data[i];
                output.QIntData[i] = val4;
            }
            THWrapper.THFloatTensor_free(_output);


            return output;
        }


    }
}

