using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;

namespace xnor
{
    public class FPBatchNorm1d : BatchNorm1d
    {


        public FPBatchNorm1d(int nOut, double eps = 1e-5) : base(nOut, eps)
        {

        }




        public void PrepareData()
        {
            InternalArray qint = RunningMean.GetQInt(256);
            qint.Data = null;
            RunningMean = qint;

            var cln = RunningVar.Clone();

            for (int i = 0; i < cln.Data.Length; i++)
            {
                cln.Data[i] = (float)Math.Sqrt(cln.Data[i] + eps);
            }
            InternalArray qint2 = cln.GetQInt(256);
            qint2.Data = null;
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
            qint3.Data = null;

            InternalArray qint4 = Weight.GetQInt(256);
            Weight = qint4;
            qint4.Data = null;

        }


        public override InternalArray Forward(InternalArray ar)
        {
#if PROFILER
            LogItem = new CalcLogItem(this, "batchNorm2d");
            var sw = Stopwatch.StartNew();
#endif
            InternalArray ret = new InternalArray(ar.Shape);

            //  if (ar.QIntData != null)
            {



                var n = ar.Shape[0];
                var c = ar.Shape[1];
                List<short> data = new List<short>();
                ret.Data = null;
                int len = 0;
                if (ar.QIntData != null)
                {
                    len = ar.QIntData.Length;
                }
                else
                {
                    len = ar.Data.Length;
                }

                for (int i = 0; i < len; i++)
                {
                    short val = 0;
                    if (ar.QIntData != null)
                    {
                        val = ar.QIntData[i];
                    }
                    else
                    {
                        val = (short)(ar.Data[i] * 256);
                    }
                    var val2 = val - RunningMean.QIntData[i];
                    var val3 = (short)((short)((int)(val2 << 8) / RunningVar.QIntData[i]));
                    var val4 = (short)((int)(val3 * Weight.QIntData[i]) >> 8);
                    var val5 = (short)((val4 + Bias.QIntData[i]));
                    //var res = val5 / 256f;
                    var res = val5;



                    //var tt = (float)(((ar.Data[i] - RunningMean.Data[i]) / Math.Sqrt(RunningVar.Data[i] + eps)) * Weight.Data[i] + Bias.Data[i]);
                    data.Add(res);
                }

                ret.QIntData = data.ToArray();

            }

            return ret;
        }
    }
}
