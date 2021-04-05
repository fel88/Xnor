using System.IO;
using System.IO.Compression;
using System.Linq;

namespace xnor
{
    public class Linear : NeuralItem
    {
        public Linear(int inFeatures, int outFeratures, bool bias)
        {
            Weight = new InternalArray(new int[] { outFeratures, inFeatures  });
            if (bias)
            {
                Bias = new InternalArray(new int[] { inFeatures });
            }
        }

        public override void LoadFromZip(ZipArchiveEntry[] ww)
        {           
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
                
            }
        }

        public InternalArray Weight;
        public InternalArray Bias;
        public override InternalArray Forward(InternalArray ar)
        {
            if (ar.QIntData != null)
            {
                ar.Data = new float[ar.QIntData.Length];
                for (int i = 0; i < ar.QIntData.Length; i++)
                {
                    ar.Data[i] = ar.QIntData[i] / 256f;
                }
                ar.QIntData = null;
            }
            InternalArray ret = new InternalArray(new int[] { 1, Weight.Shape[0] });
            for (int j = 0; j < Weight.Shape[0]; j++)
            {
                float acc = 0;
                for (int i = 0; i < Weight.Shape[1]; i++)
                {
                    acc += Weight.Get2D(j, i) * ar.Data[i];
                }
                ret.Data[j] = acc;
                if (Bias != null)
                {
                    ret.Data[j] += Bias.Data[j];
                }
            }
            return ret;
        }
    }
}
