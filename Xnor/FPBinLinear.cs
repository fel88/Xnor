using System.IO;
using System.IO.Compression;
using System.Linq;

namespace xnor
{
    public class FPBinLinear : BinLinear
    {
        public FPBinLinear(int inFeatures, int outFeatures, bool bias) : base(inFeatures, outFeatures, bias, true)
        {

        }
        public override void PrepareData()
        {
            (bn as FPBatchNorm1d).PrepareData();
            // InternalArray qint3 = Alpha.GetQInt(256);
            // Alpha = qint3;             
        }

        public override InternalArray Forward(InternalArray ar)
        {
            ar = bn.Forward(ar);
           // var orig = ar.Clone();

            ar = binop.fpbin_linear(ar, Weight, Bias, Alpha);
/*
            orig.Data = new float[orig.QIntData.Length];
            for (int a = 0; a < orig.Data.Length; a++)
            {
                orig.Data[a] = orig.QIntData[a] / 256f;
            }
            orig.QIntData = null;

            var ar2 = binop.bin_linear(orig, Weight, Bias, Alpha);*/
            return ar;


        }
    }
}
