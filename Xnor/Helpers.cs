using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace xnor
{
    public static class Helpers
    {

        public static InternalArray Pad2d(InternalArray ar)
        {
            InternalArray ret = new InternalArray(new int[] { ar.Shape[0] + 2, ar.Shape[1] + 2 });
            int pos = 0;
            int target = ret.Shape[1];
            for (int i = 0; i < ar.Shape[0]; i++)
            {
                //copy row                    
                target++;
                Array.Copy(ar.Data, pos, ret.Data, target, ar.Shape[1]);
                target += ar.Shape[1];
                pos += ar.Shape[1];
                target++;
            }

            return ret;
        }

        public static InternalArray Unpad2d(InternalArray ar)
        {
            InternalArray ret = new InternalArray(new int[] { ar.Shape[0] - 2, ar.Shape[1] - 2 });
            int pos = 0;
            int target = ar.Shape[1];
            for (int i = 0; i < ret.Shape[0]; i++)
            {
                //copy row                    
                target++;
                Array.Copy(ar.Data, target, ret.Data, pos, ret.Shape[1]);
                target += ret.Shape[1];
                pos += ret.Shape[1];
                target++;
            }

            return ret;
        }

        public static InternalArray Unpad3d(InternalArray ar, int padCnt = 1)
        {
            InternalArray ret = new InternalArray(new int[] { ar.Shape[0], ar.Shape[1] - padCnt * 2, ar.Shape[2] - padCnt * 2 });
            int pos = 0;
            int target = 0;
            for (int j = 0; j < ret.Shape[0]; j++)
            {
                target += ar.Shape[1] * padCnt;
                for (int i = 0; i < ret.Shape[1]; i++)
                {
                    //copy row                    
                    target += padCnt;
                    Array.Copy(ar.Data, target, ret.Data, pos, ret.Shape[2]);
                    target += ret.Shape[2];
                    pos += ret.Shape[2];
                    target += padCnt;
                }
                target += ar.Shape[1] * padCnt;
            }

            return ret;
        }

        public static InternalArray Pad3d(InternalArray ar, int padCnt = 1)
        {
            InternalArray ret = new InternalArray(new int[] { ar.Shape[0], ar.Shape[1] + padCnt * 2, ar.Shape[2] + padCnt * 2 });
            int pos = 0;
            int target = 0;
            for (int j = 0; j < ar.Shape[0]; j++)
            {
                target += ret.Shape[1] * padCnt;
                for (int i = 0; i < ar.Shape[1]; i++)
                {
                    //copy row                    
                    target += padCnt;
                    Array.Copy(ar.Data, pos, ret.Data, target, ar.Shape[2]);
                    target += ar.Shape[2];
                    pos += ar.Shape[2];
                    target += padCnt;
                }
                target += ret.Shape[1] * padCnt;
            }

            return ret;
        }

        public static bool AllowParallelProcessing = false;
        public static InternalArray[] ParallelProcess(NeuralItem[] items, InternalArray input)
        {
            InternalArray[] res = new InternalArray[items.Length];
            if (!AllowParallelProcessing)
            {
                for (int i = 0; i < items.Length; i++)
                {
                    res[i] = items[i].Forward(input);
                }
            }
            else
            {
                Parallel.For(0, items.Length, (i) =>
                {
                    res[i] = items[i].Forward(input);
                });
            }
            return res;
        }

        public static InternalArray Cat(InternalArray[] items, int dim = 0)
        {
            var n = items[0].Shape[0];
            var sumch = items.Sum(z => z.Shape[1]);
            var h = items[0].Shape[2];
            var w = items[0].Shape[3];

            InternalArray ret = new InternalArray(new int[] { n, sumch, h, w });

            List<float> data = new List<float>();
            foreach (var item in items)
            {
                for (int i = 0; i < n; i++)
                {
                    var img = Get3DImageFrom4DArray(item, i);
                    data.AddRange(img.Data);
                }
            }

            ret.Data = data.ToArray();
            return ret;
        }

#if NET461
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
        public static InternalArray Get3DImageFrom4DArray(this InternalArray array, int ind1)
        {
            int pos = ind1 * array.offsets[0];
            InternalArray ret = new InternalArray(new int[] { array.Shape[1], array.Shape[2], array.Shape[3] });
            Array.Copy(array.Data, pos, ret.Data, 0, ret.Data.Length);
            return ret;
        }

#if NET461
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
        public static InternalArray GetNext3DImageFrom4DArray(this InternalArray array, ref int pos)
        {

            InternalArray ret = new InternalArray(new int[] { array.Shape[1], array.Shape[2], array.Shape[3] });
            Array.Copy(array.Data, pos, ret.Data, 0, ret.Data.Length);
            pos += array.offsets[0];
            return ret;
        }
        public static InternalArray GetNext3DImageFrom4DArrayQuant(this InternalArray array, ref int pos)
        {

            InternalArray ret = new InternalArray(new int[] { array.Shape[1], array.Shape[2], array.Shape[3] });
            ret.QIntData = new short[ret.Data.Length];
            ret.Data = null;
            Array.Copy(array.QIntData, pos, ret.QIntData, 0, ret.QIntData.Length);
            pos += array.offsets[0];
            return ret;
        }
        public static List<NeuralItem> GetAllChilds(NeuralItem item, List<NeuralItem> ret = null)
        {
            if (ret == null)
            {
                ret = new List<NeuralItem>();
            }
            if (item.Childs == null)
            {
                ret.Add(item);
            }
            else
            {
                foreach (var citem in item.Childs)
                {
                    GetWeightedAllChilds(citem, ret);
                }
            }
            return ret;
        }
        public static List<NeuralItem> GetWeightedAllChilds(NeuralItem item, List<NeuralItem> ret = null)
        {
            if (ret == null)
            {
                ret = new List<NeuralItem>();
            }
            if (item.Childs == null)
            {
                ret.Add(item);
            }
            else
            {
                foreach (var citem in item.Childs)
                {
                    if (citem is AvgPool2d) continue;
                    GetWeightedAllChilds(citem, ret);
                }
            }
            return ret;
        }

        public static InternalArray ParseFromString(string str)
        {
            List<int> dims = new List<int>();

            int dim = 0;
            int maxdim = 0;
            StringBuilder sb = new StringBuilder();
            List<int> cnt = new List<int>();
            List<float> data = new List<float>();
            str = str.Substring(str.IndexOf('['));
            for (int i = 0; i < str.Length; i++)
            {
                if (str[i] == '[')
                {

                    dim++; maxdim = Math.Max(dim, maxdim);

                    if (dims.Count < maxdim)
                    {
                        dims.Add(0);
                        cnt.Add(0);
                    }
                    cnt[dim - 1] = 1;
                    sb.Clear();
                    continue;
                }
                if (str[i] == ']')
                {
                    if (dim == maxdim)
                    {
                        data.Add(float.Parse(sb.ToString()));
                    }
                    dims[dim - 1] = Math.Max(dims[dim - 1], cnt[dim - 1]);
                    dim--;
                    sb.Clear();
                    if (dim == 0) { break; }
                    continue;
                }
                if (str[i] == ',')
                {
                    if (dim == maxdim)
                    {
                        data.Add(float.Parse(sb.ToString()));
                    }
                    sb.Clear();
                    cnt[dim - 1]++;
                    continue;
                }
                sb.Append(str[i]);
            }

            InternalArray ret = new InternalArray(dims.ToArray());
            ret.Data = data.ToArray();
            return ret;
        }
        public static InternalArray ParseFromString2(string str)
        {

            var ar = str.Split(new char[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries).ToArray();
            var a1 = ar[0].Split(new char[] { ' ', ',', '(', ')' }, StringSplitOptions.RemoveEmptyEntries).ToArray();
            var dims = a1.Select(int.Parse).ToArray();
            InternalArray ret = new InternalArray(dims.ToArray());
            foreach (var ss in ar.Skip(1))
            {
                var s1 = ss.Split(new char[] { ',', '(', ')' }, StringSplitOptions.RemoveEmptyEntries).ToArray();
                int[] index = new int[dims.Length];
                for (int i = 0; i < dims.Length; i++)
                {
                    index[i] = int.Parse(s1[i]);
                }
                var val = float.Parse(s1.Last());
                ret.Set4D(index[0], index[1], index[2], index[3], val);
            }



            return ret;
        }
#if NET461
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
        public static InternalArray GetNext2dImageFrom4dArray(this InternalArray array, ref int pos0)
        {

            InternalArray ret = new InternalArray(new int[] { array.Shape[2], array.Shape[3] });
            Array.Copy(array.Data, pos0, ret.Data, 0, ret.Data.Length);
            pos0 += array.offsets[1];
            return ret;
        }
        public static InternalArray GetNext2dImageFrom4dArrayQuant(this InternalArray array, ref int pos0)
        {

            InternalArray ret = new InternalArray(new int[] { array.Shape[2], array.Shape[3] });
            ret.QIntData = new short[ret.Data.Length];
            ret.Data = null;
            Array.Copy(array.QIntData, pos0, ret.QIntData, 0, ret.QIntData.Length);
            pos0 += array.offsets[1];
            return ret;
        }
#if NET461
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
        public static InternalArray Get2DImageFrom4DArray(this InternalArray array, int ind1, int ind2)
        {
            var pos = ind1 * array.offsets[0] + ind2 * array.offsets[1];
            InternalArray ret = new InternalArray(new int[] { array.Shape[2], array.Shape[3] });
            if (array.Data != null)
            {
                Array.Copy(array.Data, pos, ret.Data, 0, ret.Data.Length);
            }
            if (array.QIntData != null)
            {
                ret.QIntData = new short[ret.Data.Length];
                ret.Data = null;
                Array.Copy(array.QIntData, pos, ret.QIntData, 0, ret.QIntData.Length);

            }
            return ret;
        }


        public static InternalArray Randn(int[] dims)
        {
            Random r = new Random();
            var ar = new InternalArray(dims);
            for (int i = 0; i < ar.Data.Length; i++)
            {
                ar.Data[i] = (float)r.NextGaussian();
            }
            return ar;
        }

        public static InternalArray Randn(this Random r, int[] dims)
        {
            var ar = new InternalArray(dims);
            for (int i = 0; i < ar.Data.Length; i++)
            {
                ar.Data[i] = (float)r.NextGaussian();
            }
            return ar;
        }

        public static double NextGaussian(this Random rand, double mean = 0, double stdDev = 1)
        {
            //Random rand = new Random(); //reuse this if you are generating many
            double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal =
                         mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)

            return randNormal;
        }

        public static long[] VectorsTest(int k = 1000)
        {

            float[] data = new float[k];
            float[] data2 = new float[k];

            Random r = new Random();
            for (int i = 0; i < k; i++)
            {
                data[i] = ((float)r.NextDouble());
                data2[i] = ((float)r.NextDouble());
            }

            float[] res = new float[data.Length];
            float[] res2 = new float[data.Length];
            var sw1 = Stopwatch.StartNew();
            for (int i = 0; i < data.Length; i++)
            {
                res2[i] = data[i] * data2[i];
            }

            sw1.Stop();
            var e1 = sw1.ElapsedMilliseconds;

            var sw2 = Stopwatch.StartNew();
            for (int i = 0; i < data.Length; i += 4)
            {
                /*Vector4 v1 = new Vector4((float)data[i], (float)data[i + 1], (float)data[i + 2], data[i + 3]);
                Vector4 v2 = new Vector4((float)data2[i], (float)data2[i + 1], (float)data2[i + 2], data2[i + 3]);

                var rr = Vector4.Multiply(v1, v2);

                res[i] = rr.X;
                res[i + 1] = rr.Y;
                res[i + 2] = rr.Z;
                res[i + 3] = rr.W;*/
            }

            sw2.Stop();
            var e2 = sw2.ElapsedMilliseconds;

            for (int i = 0; i < res2.Length; i++)
            {
                if (res2[i] != res[i])
                {
                    throw new ArgumentException();
                }
            }
            return new[] { e1, e2 };
        }
    }
}
