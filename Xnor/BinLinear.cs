using System.IO;
using System.IO.Compression;
using System.Linq;

namespace xnor
{
    public class BinLinear : NeuralItem
    {
        public BinLinear(int inFeatures, int outFeatures, bool bias, bool fpBn = false)
        {
            bn = fpBn ? new FPBatchNorm1d(inFeatures) : new BatchNorm1d(inFeatures);
            if (bias)
            {
                Bias = new InternalArray(new int[] { outFeatures });
            }
        }
        public virtual void PrepareData()
        {
            if (bn is FPBatchNorm1d fp)
            {
                fp.PrepareData();
            }
        }
        public override void LoadFromZip(ZipArchiveEntry[] ww)
        {
            bn.LoadFromZip(ww.Where(z => z.Name.Contains("_bn_")).ToArray());

            foreach (var item in ww.Where(z => !z.Name.Contains("_bn_")))
            {
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
                if (item.Name.Contains("_alpha"))
                {
                    using (var stream = item.Open())
                    {
                        using (var memoryStream = new MemoryStream())
                        {
                            stream.CopyTo(memoryStream);
                            Alpha = NpyLoader.Load(memoryStream.ToArray());
                        }
                    }
                }
            }
        }

        public InternalArray Weight;
        public InternalArray Bias;
        public InternalArray Alpha;
        public BatchNorm1d bn;
        public override InternalArray Forward(InternalArray ar)
        {
            ar = bn.Forward(ar);
            if (ar.QIntData != null)
            {
                ar.Data = new float[ar.QIntData.Length];
                for (int a = 0; a < ar.Data.Length; a++)
                {
                    ar.Data[a] = ar.QIntData[a]/256f;
                }
                ar.QIntData = null;
            }
            ar = binop.bin_linear(ar, Weight, Bias, Alpha);
            return ar;

        }
    }
}
