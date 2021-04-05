using System;
using System.Collections.Generic;
using System.Data;
using System.Globalization;
using System.Linq;
using System.Xml.Linq;

namespace xnor
{
    public class InternalArray
    {
        public IntPtr ToTHTensor()
        {
            IntPtr ret = (IntPtr)0;
           
            if (IntData != null)
            {
                ret = THWrapper.THIntTensor_new();
                CreatedTensors.Add(ret);

                if (Shape.Length == 2)
                {
                    THWrapper.THIntTensor_resize2d(ret, Shape[0], Shape[1]);
                    for (int i = 0; i < Shape[0]; i++)
                    {
                        for (int j = 0; j < Shape[1]; j++)
                        {
                            THWrapper.THIntTensor_set2d(ret, i, j, Get2DInt(i, j));
                        }
                    }

                }
                if (Shape.Length == 3)
                {
                    THWrapper.THIntTensor_resize3d(ret, Shape[0], Shape[1], Shape[2]);
                    for (int k = 0; k < Shape[0]; k++)
                        for (int i = 0; i < Shape[1]; i++)
                        {
                            for (int j = 0; j < Shape[2]; j++)
                            {
                                THWrapper.THIntTensor_set3d(ret, k, i, j, Get3DInt(k, i, j));
                            }
                        }

                }
                if (Shape.Length == 4)
                {
                    THWrapper.THIntTensor_resize4d(ret, Shape[0], Shape[1], Shape[2], Shape[3]);
                    for (int k1 = 0; k1 < Shape[0]; k1++)
                        for (int k = 0; k < Shape[1]; k++)
                            for (int i = 0; i < Shape[2]; i++)
                            {
                                for (int j = 0; j < Shape[3]; j++)
                                {
                                    THWrapper.THIntTensor_set4d(ret, k1, k, i, j, Get4DInt(k1, k, i, j));
                                }
                            }

                }
                return ret;
            }
            if (Data != null)
            {
                ret = THWrapper.THFloatTensor_new();
                CreatedTensors.Add(ret);

                if (Shape.Length == 2)
                {
                    THWrapper.THFloatTensor_resize2d(ret, Shape[0], Shape[1]);
                    for (int i = 0; i < Shape[0]; i++)
                    {
                        for (int j = 0; j < Shape[1]; j++)
                        {
                            THWrapper.THFloatTensor_set2d(ret, i, j, Get2D(i, j));
                        }
                    }

                }
                if (Shape.Length == 3)
                {
                    THWrapper.THFloatTensor_resize3d(ret, Shape[0], Shape[1], Shape[2]);
                    for (int k = 0; k < Shape[0]; k++)
                        for (int i = 0; i < Shape[1]; i++)
                        {
                            for (int j = 0; j < Shape[2]; j++)
                            {
                                THWrapper.THFloatTensor_set3d(ret, k, i, j, Get3D(k, i, j));
                            }
                        }

                }
                if (Shape.Length == 4)
                {
                    THWrapper.THFloatTensor_resize4d(ret, Shape[0], Shape[1], Shape[2], Shape[3]);
                    for (int k1 = 0; k1 < Shape[0]; k1++)
                        for (int k = 0; k < Shape[1]; k++)
                            for (int i = 0; i < Shape[2]; i++)
                            {
                                for (int j = 0; j < Shape[3]; j++)
                                {
                                    THWrapper.THFloatTensor_set4d(ret, k1, k, i, j, Get4D(k1, k, i, j));
                                }
                            }

                }
            }
            return ret;
        }
        public static List<IntPtr> CreatedTensors = new List<IntPtr>();
        public static InternalArray FromTHFloatTensor(IntPtr tensor)
        {

            var dims = THWrapper.THFloatTensor_nDimension(tensor);
            int[] d = new int[dims];
            for (int i = 0; i < dims; i++)
            {
                d[i] = (int)THWrapper.THFloatTensor_size(tensor, i);
            }
            InternalArray ret = new InternalArray(d);
            if (dims == 2)
            {
                for (int i = 0; i < d[0]; i++)
                {
                    for (int j = 0; j < d[1]; j++)
                    {
                        ret.Set2D(i, j, THWrapper.THFloatTensor_get2d(tensor, i, j));
                    }
                }
            }
            if (dims == 3)
            {
                for (int k = 0; k < d[0]; k++)

                    for (int i = 0; i < d[1]; i++)
                    {
                        for (int j = 0; j < d[2]; j++)
                        {
                            ret.Set3D(k, i, j, THWrapper.THFloatTensor_get3d(tensor, k, i, j));
                        }
                    }
            }
            if (dims == 4)
            {
                for (int k1 = 0; k1 < d[0]; k1++)
                    for (int k = 0; k < d[1]; k++)

                        for (int i = 0; i < d[2]; i++)
                        {
                            for (int j = 0; j < d[3]; j++)
                            {
                                ret.Set4D(k1, k, i, j, THWrapper.THFloatTensor_get4d(tensor, k1, k, i, j));
                            }
                        }
            }

            ret.UpdateOffsets();
            return ret;
        }
        public static InternalArray FromTHIntTensor(IntPtr tensor)
        {

            var dims = THWrapper.THIntTensor_nDimension(tensor);
            int[] d = new int[dims];
            for (int i = 0; i < dims; i++)
            {
                d[i] = (int)THWrapper.THIntTensor_size(tensor, i);
            }
            InternalArray ret = new InternalArray(d, false);
            if (dims == 2)
            {
                for (int i = 0; i < d[0]; i++)
                {
                    for (int j = 0; j < d[1]; j++)
                    {
                        var val = THWrapper.THIntTensor_get2d(tensor, i, j);
                        if (val != 0)
                        {

                        }
                        ret.Set2DInt(i, j, (uint)val);
                    }
                }
            }
            if (dims == 3)
            {
                for (int k = 0; k < d[0]; k++)

                    for (int i = 0; i < d[1]; i++)
                    {
                        for (int j = 0; j < d[2]; j++)
                        {
                            ret.Set3DInt(k, i, j, (uint)THWrapper.THIntTensor_get3d(tensor, k, i, j));
                        }
                    }
            }
            if (dims == 4)
            {
                for (int k1 = 0; k1 < d[0]; k1++)
                    for (int k = 0; k < d[1]; k++)

                        for (int i = 0; i < d[2]; i++)
                        {
                            for (int j = 0; j < d[3]; j++)
                            {
                                ret.Set4DInt(k1, k, i, j, (uint)THWrapper.THIntTensor_get4d(tensor, k1, k, i, j));
                            }
                        }
            }

            ret.UpdateOffsets();
            return ret;
        }
        public override string ToString()
        {
            if (shp == null)
            {
                string shp1 = "(";
                for (int i = 0; i < Shape.Length; i++)
                {
                    if (i > 0) { shp1 += ", "; }
                    shp1 += Shape[i];
                }
                shp1 += ")";
                shp = $"{Name}: {shp1}";
            }
            return shp;
        }
        string shp = null;
        #region ctors

        public int QIntScale = 256;
        public float Unquant(int pos)
        {
            return QIntData[pos] / (float)QIntScale;
        }

        public InternalArray GetQInt(int scale)
        {

            InternalArray ret = new InternalArray(Shape);
            ret.QIntScale = scale;
            ret.QIntData = new short[Data.Length];
            for (int i = 0; i < Data.Length; i++)
            {
                ret.QIntData[i] = (short)(Data[i] * scale);
            }
            ret.Data = null;
            return ret;
        }

        public InternalArray(int[] dims, bool isfloat = true)
        {
            Shape = (int[])dims.Clone();
            long l = 1;
            for (int i = 0; i < dims.Length; i++)
            {
                l *= dims[i];
            }
            if (isfloat)
            {
                Data = new float[l];
            }
            else
            {
                IntData = new uint[l];

            }


            offsets = new int[dims.Length - 1];
            for (int i = 0; i < dims.Length - 1; i++)
            {
                int val = 1;

                for (int j = 0; j < (i + 1); j++)
                {
                    val *= Shape[Shape.Length - 1 - j];
                }
                offsets[offsets.Length - 1 - i] = val;
            }
            //offsets = offsets.Reverse().ToArray();
            //= new int[] { Shape.Dims[1] * Shape.Dims[2] * Shape.Dims[3], Shape.Dims[2] * Shape.Dims[3], Shape.Dims[3] };
            //= new int[] { Shape.Dims[1] * Shape.Dims[2], Shape.Dims[2] };
        }

        internal void Mult(float v)
        {
            for (int i = 0; i < Data.Length; i++)
            {
                Data[i] *= v;
            }
        }
        #endregion



        #region fields

        public string Name { get; set; }
        public int[] offsets = null;
        public int GetOffset(int dim)
        {
            return offsets[dim];
        }

        public void UpdateOffsets()
        {
            var dims = Shape;
            offsets = new int[dims.Length - 1];
            for (int i = 0; i < dims.Length - 1; i++)
            {
                int val = 1;

                for (int j = 0; j < (i + 1); j++)
                {
                    val *= Shape[Shape.Length - 1 - j];
                }
                offsets[offsets.Length - 1 - i] = val;
            }
        }
        public float[] Data;
        public UInt32[] IntData;
        public short[] QIntData;
        public int[] Shape;

        #endregion


        public static InternalArray FromXml(string path)
        {
            XDocument doc = XDocument.Load(path);

            var item = doc.Descendants("data").First().Value;
            var sz = doc.Descendants("size").First().Value;

            var dims = sz.Split(new char[] { ',', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries).ToArray();
            InternalArray ret = new InternalArray(dims.Select(int.Parse).ToArray());

            var data = item.Split(new char[] { ',', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries).ToArray();
            ret.Data = data.Select(z => float.Parse(z.Replace(",", "."), CultureInfo.InvariantCulture)).ToArray();

            return ret;
        }

        public void Add(InternalArray ar)
        {
            for (int i = 0; i < Data.Length; i++)
            {
                Data[i] += ar.Data[i];
            }
        }

        public void Add(float ar)
        {
            for (int i = 0; i < Data.Length; i++)
            {
                Data[i] += ar;
            }
        }

        public static InternalArray Add(InternalArray a1, InternalArray a2)
        {
            if (a1.Shape.Length != a2.Shape.Length) throw new ArgumentException("dim1.len!=dim2.len");
            int[] outs = new int[a1.Shape.Length];
            bool rightBigger = false;
            for (int i = 0; i < a1.Shape.Length; i++)
            {
                outs[i] = Math.Max(a1.Shape[i], a2.Shape[i]);
                if (a2.Shape[i] > a1.Shape[i])
                {
                    rightBigger = true;
                }
            }
            if (rightBigger)
            {
                var temp = a2;
                a2 = a1;
                a1 = temp;
            }
            InternalArray ret = new InternalArray(outs);
            if (outs.Length == 4)
            {
                //iterate over all values of bigger one and add smallest one.
                int index = 0;
                int index2 = 0;
                for (int i = 0; i < a1.Shape[0]; i++)
                {
                    if (!a2.WithIn(i, 0, 0, 0)) continue;
                    for (int i1 = 0; i1 < a1.Shape[1]; i1++)
                    {
                        if (!a2.WithIn(i, i1, 0, 0)) continue;
                        for (int i2 = 0; i2 < a1.Shape[2]; i2++)
                        {
                            if (!a2.WithIn(i, i1, i2, 0)) continue;
                            for (int i3 = 0; i3 < a1.Shape[3]; i3++)
                            {
                                if (!a2.WithIn(i, i1, i2, i3)) continue;
                                //a1.Data[index++] = a2.Get4D(i, i1, i2, i3);
                                ret.Data[index] = a1.Data[index] + a2.Data[index2];
                                index++;
                                index2++;
                            }
                        }
                    }
                }
            }
            else
            {
                throw new NotImplementedException();
            }

            return ret;
        }
        public float Get4D(int v1, int v2, int v3, int v4)
        {
            //int[] ar = new int[] { Shape.Dims[1] * Shape.Dims[2] * Shape.Dims[3], Shape.Dims[2] * Shape.Dims[3], Shape.Dims[3] };

            int[] dat = new int[] { v1, v2, v3 };

            int pos = v4;
            for (int i = 0; i < 3; i++)
            {
                pos += dat[i] * offsets[i];
            }
            return Data[pos];
        }
        public UInt32 Get4DInt(int v1, int v2, int v3, int v4)
        {


            int[] dat = new int[] { v1, v2, v3 };

            int pos = v4;
            for (int i = 0; i < 3; i++)
            {
                pos += dat[i] * offsets[i];
            }
            return IntData[pos];
        }
        public void Set4D(int v1, int v2, int v3, int v4, float val)
        {
            //int[] ar = new int[] { Shape.Dims[1] * Shape.Dims[2] * Shape.Dims[3], Shape.Dims[2] * Shape.Dims[3], Shape.Dims[3] };
            int[] dat = new int[] { v1, v2, v3 };

            int pos = v4;
            for (int i = 0; i < 3; i++)
            {
                pos += dat[i] * offsets[i];
            }
            Data[pos] = val;
        }
        public void Set4DInt(int v1, int v2, int v3, int v4, uint val)
        {
            //int[] ar = new int[] { Shape.Dims[1] * Shape.Dims[2] * Shape.Dims[3], Shape.Dims[2] * Shape.Dims[3], Shape.Dims[3] };
            int[] dat = new int[] { v1, v2, v3 };

            int pos = v4;
            for (int i = 0; i < 3; i++)
            {
                pos += dat[i] * offsets[i];
            }
            IntData[pos] = val;
        }
#if NET461
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
        public void Set3D(int v1, int v2, int v3, float val)
        {
            Data[v3 + v1 * offsets[0] + v2 * offsets[1]] = val;
        }
        public void Set3DQuant(int v1, int v2, int v3, short val)
        {
            QIntData[v3 + v1 * offsets[0] + v2 * offsets[1]] = val;
        }
        public void Set3DInt(int v1, int v2, int v3, uint val)
        {
            IntData[v3 + v1 * offsets[0] + v2 * offsets[1]] = val;
        }
#if NET461
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
        public float Get3D(int v1, int v2, int v3)
        {
            return Data[v1 * offsets[0] + v2 * offsets[1] + v3];
        }
        public UInt32 Get3DInt(int v1, int v2, int v3)
        {
            return IntData[v1 * offsets[0] + v2 * offsets[1] + v3];
        }
        public void Sub(InternalArray ar)
        {
            for (int i = 0; i < Data.Length; i++)
            {
                Data[i] -= ar.Data[i];
            }
        }

        public void Sub(float bias)
        {
            for (int i = 0; i < Data.Length; i++)
            {
                Data[i] -= bias;
            }
        }

        public double GetItem(int[] index)
        {
            int pos = 0;
            for (int i = 0; i < index.Length; i++)
            {
                pos += index[i] * Shape[i];
            }
            return Data[pos];
        }

        public InternalArray Clone()
        {
            InternalArray ret = new InternalArray(Shape);
            ret.Data = null;
            ret.Shape = (int[])ret.Shape.Clone();
            if (Data != null)
            {
                ret.Data = new float[Data.Length];
                Array.Copy(Data, ret.Data, Data.Length);

            }
            if (QIntData != null)
            {
                ret.QIntData = new short[QIntData.Length];
                Array.Copy(QIntData, ret.QIntData, QIntData.Length);
            }
            if (IntData != null)
            {
                ret.IntData = new uint[IntData.Length];
                Array.Copy(IntData, ret.IntData, IntData.Length);
            }
            return ret;
        }

        internal InternalArray Unsqueeze(int v)
        {
            InternalArray ret = new InternalArray(new int[] { 1, Shape[0], Shape[1], Shape[2] });
            ret.Data = Data.ToArray();
            return ret;
        }

        internal InternalArray Transpose(int[] v)
        {

            InternalArray ret = new InternalArray(v.Select(z => Shape[z]).ToArray());

            for (int i = 0; i < Shape[0]; i++)
            {
                for (int j = 0; j < Shape[1]; j++)
                {
                    for (int k = 0; k < Shape[2]; k++)
                    {
                        var val = Get3D(i, j, k);
                        var ar1 = new int[] { i, j, k };
                        ret.Set3D(ar1[v[0]], ar1[v[1]], ar1[v[2]], val);
                    }
                }
            }

            return ret;

        }
        internal InternalArray Transpose2D()
        {
            if (Shape.Length != 2) throw new Exception();
            
            InternalArray ret = new InternalArray(new int[] { Shape[1], Shape[0] });
            if (QIntData != null)
            {
                ret.Data = null;
                ret.QIntData = new short[QIntData.Length];
                for (int i = 0; i < Shape[0]; i++)
                {
                    for (int j = 0; j < Shape[1]; j++)
                    {
                        var val = Get2DQuant(i, j);
                        ret.Set2DQuant(j, i, val);
                    }
                }
                return ret;
            }

            for (int i = 0; i < Shape[0]; i++)
            {
                for (int j = 0; j < Shape[1]; j++)
                {
                    var val = Get2D(i, j);
                    ret.Set2D(j, i, val);
                }
            }

            return ret;

        }
        internal void Set2DInt(int i, int j, UInt32 val)
        {
            int pos = i * Shape[1] + j;
            IntData[pos] = val;
        }
        internal void Set2D(int i, int j, float val)
        {
            int pos = i * Shape[1] + j;
            Data[pos] = val;
        }
        internal void Set2DQuant(int i, int j, short val)
        {
            int pos = i * Shape[1] + j;
            QIntData[pos] = val;
        }

#if NET461
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
        internal bool WithIn(int x, int y)
        {
            return x >= 0 && y >= 0 && x < Shape[0] && y < Shape[1];
        }
#if NET461
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
        internal bool WithIn(int x, int y, int z)
        {
            return x >= 0 && y >= 0 && z >= 0 && x < Shape[0] && y < Shape[1] && z < Shape[2];
        }

#if NET461
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
        internal bool WithIn(int x, int y, int z, int k)
        {
            return x >= 0 && y >= 0 && z >= 0 && k >= 0 && x < Shape[0] && y < Shape[1] && z < Shape[2] && k < Shape[3];
        }

#if NET461
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
        public float Get2D(int i, int j)
        {
            int pos = i * Shape[1] + j;
            return Data[pos];
        }
        public short Get2DQuant(int i, int j)
        {
            int pos = i * Shape[1] + j;
            return QIntData[pos];
        }

        public UInt32 Get2DInt(int i, int j)
        {
            int pos = i * Shape[1] + j;
            return IntData[pos];
        }

        public static double tolerance = 10e-6;

        public bool IsEqual(InternalArray resss)
        {
            if (Shape.Length != resss.Shape.Length) return false;
            for (int i = 0; i < Shape.Length; i++)
            {
                if (Shape[i] != resss.Shape[i]) return false;
            }
            for (int i = 0; i < Data.Length; i++)
            {
                if (Math.Abs(Data[i] - resss.Data[i]) > tolerance) return false;
            }
            return true;
        }
    }
}
