using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;

namespace xnor
{
    public class BatchNorm2d : NeuralItem
    {
        public override void LoadFromZip(ZipArchiveEntry[] ww)
        {
            foreach (var item in ww)
            {
                if (item.Name.Contains("_var"))
                {
                    using (var stream = item.Open())
                    {
                        using (var memoryStream = new MemoryStream())
                        {
                            stream.CopyTo(memoryStream);
                            RunningVar = NpyLoader.Load(memoryStream.ToArray());
                        }
                    }
                }
                if (item.Name.Contains("_mean"))
                {
                    using (var stream = item.Open())
                    {
                        using (var memoryStream = new MemoryStream())
                        {
                            stream.CopyTo(memoryStream);
                            RunningMean = NpyLoader.Load(memoryStream.ToArray());
                        }
                    }
                }
                if (item.Name.Contains("_bias"))
                {
                    using (var stream = item.Open())
                    {
                        using (var memoryStream = new MemoryStream())
                        {
                            stream.CopyTo(memoryStream);
                            Bias = NpyLoader.Load(memoryStream.ToArray());
                        }
                    }
                }
                if (item.Name.Contains("_weight"))
                {
                    using (var stream = item.Open())
                    {
                        using (var memoryStream = new MemoryStream())
                        {
                            stream.CopyTo(memoryStream);
                            Weight = NpyLoader.Load(memoryStream.ToArray());
                        }
                    }
                }
            }
        }
        protected double eps;
        public BatchNorm2d(int nOut, double eps = 1e-5)
        {
            this.eps = eps;
            Bias = new InternalArray(new int[] { nOut });
            Weight = new InternalArray(new int[] { nOut });
            for (int i = 0; i < Weight.Data.Length; i++)
            {
                Weight.Data[i] = 1;
            }
        }


        public override int SetData(List<InternalArray> arrays)
        {
            //if (!arrays[0].Name.Contains("we")) throw new ArgumentException("not bn.weight detected");
            //if (!arrays[1].Name.Contains("bias")) throw new ArgumentException("not bn.bias detected");
            Weight = arrays[0];
            Bias = arrays[1];
            RunningMean = arrays[2];
            RunningVar = arrays[3];
            return 4;
        }


        public InternalArray RunningMean;
        public InternalArray RunningVar;
        public InternalArray Bias;
        public InternalArray Weight;

        public override InternalArray Forward(InternalArray ar)
        {
#if PROFILER
            LogItem = new CalcLogItem(this, "batchNorm2d");
            var sw = Stopwatch.StartNew();
#endif

            InternalArray ret = new InternalArray(ar.Shape);

            var n = ar.Shape[0];
            var c = ar.Shape[1];
            List<float> data = new List<float>();


            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    var img = Helpers.Get2DImageFrom4DArray(ar, i, j);
                    for (int zi = 0; zi < img.Data.Length; zi++)
                    {                        
                        img.Data[zi] = (float)(((img.Data[zi] - RunningMean.Data[j]) / Math.Sqrt(RunningVar.Data[j] + eps)) * Weight.Data[j] + Bias.Data[j]); ;                      
                    }

                    data.AddRange(img.Data);
                }
            }

            ret.Data = data.ToArray();
#if PROFILER

            sw.Stop();
            if (Parent != null)
            {
                Profiler.AddLog(Parent.LogItem, LogItem);
            }
#endif
            return ret;
        }
    }
}
