using System;
using System.Collections.Generic;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Xml.Linq;

namespace xnor
{
    public class FPBinLeNetTest : NeuralItem
    {
        public FPBinLeNetTest()
        {
            conv1 = new FPConv2d(1, 20, 5, 1, 0);
            pool1 = new MaxPool2d(2, 2);

            conv2 = new BinConv2d(20, 50, 5, 1, 0, fpBn: true);
            pool2 = new MaxPool2d(2, 2);
            //fc1 = new BinLinear(50 * 4 * 4, 500, false, true);
            fc1 = new FPBinLinear(50 * 4 * 4, 500, false);
            fc2 = new FPLinear(500, 10, true);
            bn1 = new FPBatchNorm2d(20);
            bn2 = new FPBatchNorm2d(50);
            bn3 = new FPBatchNorm1d(500);



            conv1.Name = "conv1";
            pool1.Name = "pool1";
            pool2.Name = "pool2";
            conv2.Name = "conv2";
            fc1.Name = "fc1";
            fc2.Name = "fc2";
            bn1.Name = "bn1";
            bn2.Name = "bn2";
            bn3.Name = "bn3";
            items.Add(bn1);
            items.Add(bn2);
            items.Add(bn3);
            items.Add(fc2);
            items.Add(fc1);
            items.Add(conv1);
            items.Add(conv2);
            items.Add(pool1);
            items.Add(pool2);
        }

        public void PrepareData()
        {
            if (conv1 is FPConv2d ff) ff.PrepareData();
            bn1.PrepareData();
            bn2.PrepareData();
            bn3.PrepareData();
            fc1.PrepareData();
            if (fc2 is FPLinear fpl)
                fpl.PrepareData();
            conv2.PrepareData();


        }
        public List<NeuralItem> items = new List<NeuralItem>();

        public Conv2d conv1;
        public FPBatchNorm2d bn1;
        public MaxPool2d pool1;
        public MaxPool2d pool2;
        public BinConv2d conv2;
        public Linear fc2;
        public BinLinear fc1;
        public FPBatchNorm2d bn2;
        public FPBatchNorm1d bn3;

        public void LoadFromZip(string path)
        {
            using (ZipArchive zip = ZipFile.Open(path, ZipArchiveMode.Read))
                foreach (ZipArchiveEntry entry in zip.Entries)
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
                }
        }

        public static InternalArray Relu(InternalArray ar)
        {
            InternalArray ar1 = new InternalArray(ar.Shape);

            if (ar.Data != null)
            {
                for (int i = 0; i < ar.Data.Length; i++)
                {
                    ar1.Data[i] = Math.Max(0, ar.Data[i]);
                }
            }
            else if (ar.QIntData != null)
            {

                ar1.QIntData = new short[ar1.Data.Length];
                ar1.Data = null;
                for (int i = 0; i < ar.QIntData.Length; i++)
                {
                    ar1.QIntData[i] = Math.Max((short)0, ar.QIntData[i]);
                }
            }

            return ar1;
        }


        public void SaveXml()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("<?xml version=\"1.0\"?>");
            sb.AppendLine("<root>");
            conv1.AppendToXml(sb);
            sb.AppendLine("</root>");


        }



        public List<LogInfo> results = new List<LogInfo>();
        public override InternalArray Forward(InternalArray ar)
        {
            results.Clear();
            results.Add(new LogInfo(null, ar.Clone(), "input"));
            ar.QIntData = new short[ar.Data.Length];
            for (int i = 0; i < ar.QIntData.Length; i++)
            {
                ar.QIntData[i] = (short)(ar.Data[i]*256);
            }
            ar.Data = null;
            ar = conv1.Forward(ar);
            results.Add(new LogInfo(conv1, ar.Clone()));


            ar = bn1.Forward(ar);
            results.Add(new LogInfo(bn1, ar.Clone()));

            ar = Relu(ar);
            results.Add(new LogInfo(null, ar.Clone(), "relu1"));

            ar = pool1.Forward(ar);
            results.Add(new LogInfo(pool1, ar.Clone()));


            ar = conv2.Forward(ar);
            results.Add(new LogInfo(conv2, ar.Clone(), "BinConv2D conv2"));


            ar = bn2.Forward(ar);
            results.Add(new LogInfo(bn2, ar.Clone()));


            ar = Relu(ar);
            results.Add(new LogInfo(null, ar.Clone(), "relu2"));
            ar = pool2.Forward(ar);
            results.Add(new LogInfo(pool2, ar.Clone()));

            var ar2 = new InternalArray(new int[] { 1, 4 * 4 * 50 });
            if (ar.QIntData != null)
            {
                ar2.QIntData = new short[ar2.Data.Length];
                ar2.Data = null;
                for (int i = 0; i < ar.QIntData.Length; i++)
                {
                    ar2.QIntData[i] = ar.QIntData[i];
                }
            }
            else
            {
                for (int i = 0; i < ar.Data.Length; i++)
                {
                    ar2.Data[i] = ar.Data[i];
                }
            }

            results.Add(new LogInfo(null, ar2.Clone(), "view"));


            ar = fc1.Forward(ar2);
            results.Add(new LogInfo(fc1, ar.Clone(), "BinLinear fc1"));

            ar = bn3.Forward(ar);
            results.Add(new LogInfo(bn3, ar.Clone()));

            ar = Relu(ar);
            results.Add(new LogInfo(null, ar.Clone(), "relu3"));
            //input 500
            ar = fc2.Forward(ar);
            results.Add(new LogInfo(fc2, ar.Clone()));


            //output 10
            //output size should be [1,10]
            results.Add(new LogInfo(null, ar.Clone(), "output"));
            return ar;
        }
    }
}
