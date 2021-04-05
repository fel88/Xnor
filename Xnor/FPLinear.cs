using System.IO;
using System.IO.Compression;
using System.Linq;

namespace xnor
{
    public class FPLinear : Linear
    {
        public FPLinear(int inFeatures, int outFeratures, bool bias) : base(inFeatures, outFeratures, bias)
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


        public override InternalArray Forward(InternalArray ar)
        {

            InternalArray ret = new InternalArray(new int[] { 1, Weight.Shape[0] });
            for (int j = 0; j < Weight.Shape[0]; j++)
            {
                float acc = 0;
                for (int i = 0; i < Weight.Shape[1]; i++)
                {
                    var a1 = Weight.Get2DQuant(j, i);
                    var a2 = ar.QIntData[i];
                    var val4 = (short)((int)(a1 * a2) >> 8);
                    acc += val4;
                }
                ret.Data[j] = acc;
                if (Bias != null)
                {
                    ret.Data[j] += Bias.QIntData[j];
                }
            }
            return ret;
        }
    }
}
