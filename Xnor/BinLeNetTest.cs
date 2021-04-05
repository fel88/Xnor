using System;
using System.Collections.Generic;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Xml.Linq;

namespace xnor
{
    public class BinLeNetTest : NeuralItem
    {
        public BinLeNetTest()
        {
            conv1 = new Conv2d(1, 20, 5, 1, 0);
            pool1 = new MaxPool2d(2, 2);
            conv2 = new BinConv2d(20, 50, 5, 1, 0);
            pool2 = new MaxPool2d(2, 2);
            fc1 = new BinLinear(50 * 4 * 4, 500, false);
            fc2 = new Linear(500, 10, true);
            bn1 = new BatchNorm2d(20);
            bn2 = new BatchNorm2d(50);
            bn3 = new BatchNorm1d(500);

            
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
        public List<NeuralItem> items = new List<NeuralItem>();

        public Conv2d conv1;
        public BatchNorm2d bn1;
        public MaxPool2d pool1;
        public MaxPool2d pool2;
        public BinConv2d conv2;
        public Linear fc2;
        public BinLinear fc1;
        public BatchNorm2d bn2;
        public BatchNorm1d bn3;

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
            for (int i = 0; i < ar.Data.Length; i++)
            {
                ar1.Data[i] = Math.Max(0, ar.Data[i]);
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
            results.Add(new LogInfo(null, ar.Clone(),"input"));
            ar = conv1.Forward(ar);
            results.Add(new LogInfo(conv1, ar.Clone()));


            ar = bn1.Forward(ar);
            results.Add(new LogInfo(bn1, ar.Clone()));

            ar = Relu(ar);
            results.Add(new LogInfo(null, ar.Clone(),"relu1"));

            ar = pool1.Forward(ar);
            results.Add(new LogInfo(pool1, ar.Clone()));

            if (ar.Data.Any(z => float.IsInfinity(z) || float.IsNaN(z)))
            {

            }
            ar = conv2.Forward(ar);
            results.Add(new LogInfo(conv2, ar.Clone(), "BinConv2D conv2"));

            
            if (ar.Data.Any(z => float.IsInfinity(z) || float.IsNaN(z)))
            {

            }
            ar = bn2.Forward(ar);
            results.Add(new LogInfo(bn2, ar.Clone()));

            
            ar = Relu(ar);
            results.Add(new LogInfo(null, ar.Clone(), "relu2"));
            ar = pool2.Forward(ar);
            results.Add(new LogInfo(pool2, ar.Clone()));

            var ar2 = new InternalArray(new int[] { 1, 4 * 4 * 50 });
            for (int i = 0; i < ar.Data.Length; i++)
            {
                ar2.Data[i] = ar.Data[i];
            }
            results.Add(new LogInfo(null, ar.Clone(), "view"));

            
            ar = fc1.Forward(ar2);
            results.Add(new LogInfo(fc1, ar.Clone(), "BinLinear fc1"));
            
            ar = bn3.Forward(ar);
            results.Add(new LogInfo(bn3, ar.Clone()));

            ar = Relu(ar);
            results.Add(new LogInfo(null, ar.Clone(), "relu3"));
            //input 500
            ar = fc2.Forward(ar);
            results.Add(new LogInfo(fc1, ar.Clone()));

            
            //output 10
            //output size should be [1,10]
            results.Add(new LogInfo(null, ar.Clone(), "output"));
            return ar;
        }
    }
    public class LogInfo
    {
        public LogInfo(NeuralItem item, InternalArray outp, string label = null)
        {
            Label = label;
            Source = item;
            Output = outp;
        }
        public string Label;
        public NeuralItem Source;
        public InternalArray Output;
    }

    public class Fix
    {

        public const int FIXED_POINT = 16;
        public const int ONE = 1 << FIXED_POINT;

        public static int mul(int a, int b)
        {
            return (int)((long)a * (long)b >> FIXED_POINT);
        }

        public static int toFix(double val)
        {
            return (int)(val * ONE);
        }

        public static int intVal(int fix)
        {
            return fix >> FIXED_POINT;
        }

        public static double doubleVal(int fix)
        {
            return ((double)fix) / ONE;
        }

        public static void main(String[] args)
        {
            var a = 0.023f;
            var b = 0.34f;
            var aa = (int)(a * 256);
            var sh1 = aa >> 8;
            var aab = (aa / 256f);
            var bb = (int)(b * 256);
            var cc = (aa * bb) / 256;
            var cc2 = cc / 256f;
            var real = a * b;
            var error = Math.Abs(real - cc2);
            int f1 = toFix(Math.PI);
            int f3 = toFix(-Math.PI);
            int f2 = toFix(2);

            int result = mul(f1, f2);
            int result2 = mul(f2, f3);
            var res = doubleVal(result);
            var res2 = doubleVal(result2);

            var a1=("f1:" + f1 + "," + intVal(f1));
            var a2=("f2:" + f2 + "," + intVal(f2));
            var a3=("r:" + result + "," + intVal(result));
            var a4=("double: " + doubleVal(result));

        }
    }
}
