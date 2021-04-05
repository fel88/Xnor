using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;

namespace xnor
{
    public class FPBatchNorm2d : BatchNorm2d
    {
        public FPBatchNorm2d(int nOut, double eps = 1e-5) : base(nOut, eps)
        {

        }


        public void PrepareData()
        {
            InternalArray qint = RunningMean.GetQInt(256);
            RunningMean = qint;
            var cln = RunningVar.Clone();

            for (int i = 0; i < cln.Data.Length; i++)
            {
                cln.Data[i] = (float)Math.Sqrt(cln.Data[i] + eps);
            }
            InternalArray qint2 = cln.GetQInt(256);

            for (int i = 0; i < qint2.QIntData.Length; i++)
            {
                if (qint2.QIntData[i] == 0) qint2.QIntData[i] = 1;
                var q1 = qint2.Unquant(i);
                var err = Math.Abs(q1 - Math.Sqrt(RunningVar.Data[i] + eps));
                if (err > 0.1)
                {

                }
            }
            RunningVar = qint2;

            InternalArray qint3 = Bias.GetQInt(256);
            Bias = qint3;
            InternalArray qint4 = Weight.GetQInt(256);
            Weight = qint4;
        }


        public override InternalArray Forward(InternalArray ar)
        {

            InternalArray ret = new InternalArray(ar.Shape);

            var n = ar.Shape[0];
            var c = ar.Shape[1];
            List<short> data = new List<short>();

            InternalArray ret2 = new InternalArray(ar.Shape);
            ret2.Data = null;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    var img = Helpers.Get2DImageFrom4DArray(ar, i, j);
                    int len = 0;
                    if (img.QIntData != null)
                    {
                        len = img.QIntData.Length;
                    }
                    else
                    {
                        len = img.Data.Length;
                    }
                    for (int zi = 0; zi < len; zi++)
                    {
                        //  var orig = (float)(((img.Data[zi] - RunningMean.Data[j]) / Math.Sqrt(RunningVar.Data[j] + eps)) * Weight.Data[j] + Bias.Data[j]);
                        //img.Data[zi] = (float)(((img.Data[zi] - RunningMean.Data[j]) / Math.Sqrt(RunningVar.Data[j] + eps)) * Weight.Data[j] + Bias.Data[j]);
                        short val = 0;
                        if (img.QIntData != null)
                        {
                            val = img.QIntData[zi];
                        }
                        else
                        {
                            val = (short)(img.Data[zi] * 256);
                        }
                        var val2 = val - RunningMean.QIntData[j];
                        var val3 = (short)((short)((int)(val2 << 8) / RunningVar.QIntData[j]));
                        var val4 = (short)((int)(val3 * Weight.QIntData[j]) >> 8);
                        var val5 = (short)((val4 + Bias.QIntData[j]));
                        //var res = val5 / 256f;
                        var res = val5;

                        //var input = img.Data[zi];
                        //var res2 = (float)(((input - (qint.QIntData[j] / 256f)) / (qint2.QIntData[j] / 256f)) * qint4.QIntData[j] / 256f + qint3.QIntData[j] / 256f);
                        //img.Data[zi] = res;
                        data.Add(res);
                        //img.Data[zi] = res;

                    }

                    //data.AddRange(img.Data);
                }
            }

            ret2.QIntData = data.ToArray();

            return ret2;
        }
    }
}
