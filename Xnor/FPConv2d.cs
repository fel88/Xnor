using System;
using System.Text;
using System.Threading.Tasks;

namespace xnor
{
    public class FPConv2d : Conv2d
    {
        public FPConv2d(int inChannels, int outChannels, int kSize, int stride, int padding, bool bias = false, int dilation = 1)
            : base(inChannels, outChannels, kSize, stride, padding, bias, dilation)
        {

        }

        public void PrepareData()
        {
            if (Bias != null)
            {
                InternalArray qint3 = Bias.GetQInt(256);
                Bias = qint3;
            }
            InternalArray qint4 = Weight.GetQInt(256);
            Weight = qint4;
        }


        public override InternalArray ProcessImageOptimized2(InternalArray ar, int hout, int wout, int c, int hin, int win)
        {


            InternalArray ret = new InternalArray(new int[] { outChannels, hout, wout });
            ret.QIntData = new short[ret.Data.Length];
            ret.Data = null;
            InternalArray[,] filters = new InternalArray[outChannels, c];

            int pos0 = 0;
            for (int ch = 0; ch < outChannels; ch++)
            {
                for (int zz = 0; zz < c; zz++)
                {
                    var kernel = Weight.GetNext2dImageFrom4dArrayQuant(ref pos0);                    
                    filters[ch, zz] = kernel;
                }
            }

            int shiftx = padding[0] - kernelSize[0] / 2;
            int shifty = padding[1] - kernelSize[1] / 2;

            Parallel.For(0, hout, (i) =>
            {
                var imul = (i) * stride[0] - kernelSize[0] / 2 - shiftx;
                var maxi1 = Math.Min((ar.Shape[1] - imul) / dilation[0], kernelSize[0]);
                var mini1 = Math.Max((int)Math.Ceiling(-(double)imul / dilation[0]), 0);
                Parallel.For(0, wout, (j) =>
                {
                    var jmul = (j) * stride[1] - kernelSize[1] / 2 - shifty;
                    var minj1 = Math.Max((int)Math.Ceiling(-(double)jmul / dilation[1]), 0);
                    var maxj1 = Math.Min((ar.Shape[2] - jmul) / dilation[1], kernelSize[1]);

                    for (int ch = 0; ch < outChannels; ch++)
                    {
                        short val = 0;

                        for (int zz = 0; zz < c; zz++)
                        {
                            var kernel = filters[ch, zz];
                            var offset1 = zz * ar.offsets[0];
                            int kindex = 0;

                            for (int i1 = mini1; i1 < maxi1; i1++)
                            {
                                var x = imul + i1 * dilation[0];

                                for (int j1 = minj1; j1 < maxj1; j1++)
                                {
                                    var y = jmul + j1 * dilation[1];
                                    var index = offset1 + x * ar.offsets[1] + y;

                                    var val4 = (short)((int)(kernel.QIntData[kindex] * ar.QIntData[index]) >> 8);
                                    val += val4;
                                    kindex++;
                                }
                            }
                        }
                        ret.Set3DQuant(ch, i, j, val);
                    }
                });
            });
            //for (int i = 0; i < hout; i++)
            {

            }

            return ret;
        }





        public override InternalArray Forward(InternalArray ar)
        {
            //Profiler.PushCurrent(new CalcLogItem(this, "conv2d"));


            var hin = ar.Shape[2];
            var win = ar.Shape[3];
            var n = ar.Shape[0];
            var c = ar.Shape[1];

            var hout = ((hin + 2 * padding[0] - dilation[0] * (kernelSize[0] - 1) - 1) / stride[0]) + 1;
            var wout = ((win + 2 * padding[1] - dilation[1] * (kernelSize[1] - 1) - 1) / stride[1]) + 1;



            var cout = Weight.Shape[0];
            InternalArray ret = new InternalArray(new int[] { n, cout, hout, wout });
            ret.QIntData = new short[ret.Data.Length];
            ret.Data = null;
            //get all 3d images            
            int pos = 0;
            int pos0 = 0;
            for (int i = 0; i < n; i++)
            {
                var img = ar.GetNext3DImageFrom4DArrayQuant(ref pos0);
                InternalArray img2 = null;
                img2 = ProcessImageOptimized2(img, hout, wout, c, hin, win);


                Array.Copy(img2.QIntData, 0, ret.QIntData, pos, img2.QIntData.Length);
                pos += img2.QIntData.Length;
            }

            return ret;
        }

    }
}
