using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace xnor
{
    public class Stuff
    {
        public static int ReadInt(byte[] bytes, int start)
        {
            int res = 0;
            for (int i = 0; i < 4; i++)
            {
                res ^= (bytes[start + (3 - i)] << (i * 8));
            }
            return res;

        }
        public static MnistItem ReadImage(byte[] bb, int start, int w, int h)
        {
            byte[,] data = new byte[w, h];
            int cntr = 0;
            for (int j = 0; j < h; j++)
            {
                for (int i = 0; i < w; i++)
                {
                    data[i, j] = bb[start + cntr];
                    cntr++;
                }
            }
            return new MnistItem() { Data = data };
        }
        public static MnistItem[] LoadImages(string imgPath, string labelsPath, bool withBitmap = false)
        {
            List<MnistItem> bmps = new List<MnistItem>();

            Stopwatch sw = new Stopwatch();
            sw.Start();
            var bytes = File.ReadAllBytes(imgPath);
            int imagesCnt = ReadInt(bytes, 4);
            int indexer = 8;
            var w = ReadInt(bytes, indexer);
            indexer += 4;
            var h = ReadInt(bytes, indexer);
            indexer += 4;
            for (int i = 0; i < imagesCnt; i++)
            {
                bmps.Add(ReadImage(bytes, indexer, w, h));
                indexer += w * h;

            }
            sw.Stop();
            var ms = sw.ElapsedMilliseconds;
            sw.Stop(); sw.Reset();
            sw.Start();
            if (withBitmap)
            {
                foreach (var mnistItem in bmps)
                {
                    mnistItem.GetBitmap();
                }
            }
            sw.Stop();
            var ms2 = sw.ElapsedMilliseconds;
            sw.Stop(); sw.Reset();
            sw.Start();

            #region load lables
            bytes = File.ReadAllBytes(labelsPath);
            imagesCnt = ReadInt(bytes, 4);
            indexer = 8;
            for (int i = 0; i < imagesCnt; i++)
            {
                bmps[i].Label = bytes[indexer];
                indexer++;
            }
            sw.Stop();
            var ms3 = sw.ElapsedMilliseconds;


            #endregion

            return bmps.ToArray();

        }
    }
}
