using System;
using System.Collections.Generic;
using System.IO.Compression;
using System.Linq;
using System.Xml.Linq;

namespace xnor
{
    public class BinVGG13 : NeuralItem
    {
        public List<LogInfo> results = new List<LogInfo>();

        public BinVGG13()
        {
            int[] cfg = new int[] { 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1 };
            //  'VGG13': [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            int cnt = 1;
            int in_channels = 64;
            items.Add(new Conv2d(3, 64, 3, 1, 1) { Name = "conv0" });
            items.Add(new BatchNorm2d(64) { Name = "bn0" });
            items.Add(new Relu() { Name = "relu0" });
            foreach (var x in cfg)
            {
                if (x == -1)
                {
                    items.Add(new MaxPool2d(2, 2) { Name = "pool" + cnt });
                    cnt++;
                }
                else
                {
                    items.Add(new BinConv2d(in_channels, x, 3, 1, 1) { Name = "conv" + cnt });
                    cnt++;
                    items.Add(new BatchNorm2d(x) { Name = "bn" + cnt });
                    cnt++;
                    items.Add(new Relu() { Name = "relu" + cnt });
                    cnt++;
                    in_channels = x;
                }
            }
            items.Add(new AvgPool2d(1, 1) { Name = "pool" + cnt });

            classifier = new Linear(512, 10, false) { Name = "classifier" };
            items.Add(classifier);


        }
        Linear classifier;
        public override InternalArray Forward(InternalArray ar)
        {
            results.Add(new LogInfo(null, ar.Clone()) { Label = "input" });
            foreach (var item in items)
            {
                ar = item.Forward(ar);
                results.Add(new LogInfo(item, ar.Clone()));
            }
            results.Add(new LogInfo(null, ar.Clone()) { Label = "output" });

            return ar;
        }
        public List<NeuralItem> items = new List<NeuralItem>();

        public void LoadFromZip(string path)
        {
            using (ZipArchive zip = ZipFile.Open(path, ZipArchiveMode.Read))
            {
                foreach (var item in items)
                {
                    var ww = zip.Entries.Where(z => z.Name.Contains(item.Name + "_")).ToArray();
                    item.LoadFromZip(ww);
                }
                /* foreach (ZipArchiveEntry entry in zip.Entries)
                 {
                     if (!entry.Name.EndsWith(".xml")) continue;

                     using (var stream = entry.Open())
                     {
                         var doc = XDocument.Load(stream);
                         var root = doc.Descendants("root").First();
                         foreach (var item in root.Elements())
                         {
                             if (item.Attribute("name") != null)
                             {
                                 var name = item.Attribute("name").Value;
                                 var fr = items.FirstOrDefault(z => z.Name == name);
                                 if (fr == null) continue;
                                 var ww = zip.Entries.Where(z => z.Name.Contains($"{fr.Name}_")).ToArray();
                                 fr.LoadFromZip(ww);
                             }
                         }
                     }
                     break;
                 }*/
            }
        }
    }

    public class Relu : NeuralItem
    {


        public override InternalArray Forward(InternalArray ar1)
        {
            InternalArray ar = ar1.Clone();
            for (int i = 0; i < ar.Data.Length; i++)
            {
                ar.Data[i] = Math.Max(0, ar.Data[i]);
            }
            return ar;
        }
    }
}
