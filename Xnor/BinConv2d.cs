using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;

namespace xnor
{
    public class BinConv2d : NeuralItem
    {
        public BinConv2d(int inChannels, int outChannels, int kSize, int stride, int padding, bool bias = false, int dilation = 1, bool fpBn = false)
        {
            this.inChannels = inChannels;
            this.outChannels = outChannels;
            Weight = new InternalArray(new int[] { outChannels, inChannels, kSize, kSize });

            this.padding = new int[] { padding, padding };
            this.stride = new[] { stride, stride };
            this.kernelSize = new[] { kSize, kSize };
            this.dilation = new[] { dilation, dilation };
            if (fpBn)
            {
                bn = new FPBatchNorm2d(inChannels);
            }
            else
            {
                bn = new BatchNorm2d(inChannels);
            }
        }

        public void PrepareData()
        {
            if (bn is FPBatchNorm2d fp)
            {
                fp.PrepareData();
            }
        }


        public BatchNorm2d bn;

        public InternalArray Weight;
        public InternalArray Bias;
        public InternalArray Alpha;
        int[] kernelSize;
        int[] padding;
        int[] stride;
        int[] dilation;

        int inChannels;
        int outChannels;

        public override int SetData(List<InternalArray> arrays)
        {
            //if (!arrays[0].Name.Contains("conv")) throw new ArgumentException("not conv weight detected");
            Weight = arrays[0];
            return 1;
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

        public override InternalArray Forward(InternalArray ar)
        {
            ar = bn.Forward(ar);

            if (ar.Data != null && ar.Data.Any(z => float.IsInfinity(z) || float.IsNaN(z)))
            {

            }
            if (ar.QIntData != null)
            {
                
                return binop.fpbin_conv2d(ar, Weight, Bias, Alpha, kernelSize, stride, padding); ;
                
            }
            var ret = binop.bin_conv2d(ar, Weight, Bias, Alpha, kernelSize, stride, padding);
            
            if (ar.Data != null && ret.Data.Any(z => float.IsInfinity(z) || float.IsNaN(z)))
            {

            }
            return ret;
        }
    }
}
