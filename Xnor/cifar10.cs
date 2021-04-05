using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace xnor
{
    public partial class cifar10 : Form
    {
        public cifar10()
        {
            InitializeComponent();
            Shown += Cifar10_Shown;


            labels = File.ReadAllLines(Path.Combine("batches.meta.txt")).ToArray();
            net = new BinVGG13();
            net.LoadFromZip("BinVGG13.zip");
        }
        BinVGG13 net;
        public void progressReport(float t)
        {
            statusStrip1.Invoke((Action)(() =>
            {
                toolStripProgressBar1.Value = (int)Math.Round(t * 100);
            }));
        }
        private void Cifar10_Shown(object sender, EventArgs e)
        {
            Thread th = new Thread(() =>
            {
                SetStatusInfo("Loading images..");
                items = LoadCifarImages(Path.Combine("test_batch.bin"), progressReport);
                SetStatusInfo("Done.");
                statusStrip1.Invoke((Action)(() =>
                {
                    toolStripProgressBar1.Visible = false;

                }));
            });
            th.IsBackground = true;
            th.Start();
        }

        string[] labels;
        CifarItem[] items;
        public class CifarItem
        {
            public InternalArray x;
            public int label;
            public bool isVal;
            public object[] raw;
            public bool isval;
            public Bitmap Bmp;
        }

        Random r = new Random();
        private void button1_Click(object sender, EventArgs e)
        {
            if (items != null && items.Any())
            {
                var item = items[r.Next(items.Length)];


                var res = net.Forward(Normalize(item.x));
                int maxi = -1;

                for (int i = 0; i < res.Data.Length; i++)
                {
                    if (maxi == -1 || res.Data[i] > res.Data[maxi])
                    {
                        maxi = i;
                    }
                }
                label2.Text = labels[maxi];
                label1.Text = labels[item.label];
                pictureBox1.Image = item.Bmp;

                listView1.Items.Clear();
                foreach (var rr in net.results)
                {


                    var fr = rr.Source;
                    string weights = "";
                    if (fr is Conv2d c2)
                    {
                        weights = c2.Weight.Data.Length.ToString("N0");
                    }
                    if (fr is BinConv2d c3)
                    {
                        weights = c3.Weight.IntData.Length.ToString("N0");
                    }
                    if (fr is Linear lin)
                    {
                        weights = lin.Weight.Data.Length.ToString("N0");
                    }
                    if (fr is BinLinear blin)
                    {
                        weights = blin.Weight.IntData.Length.ToString("N0");
                    }
                    listView1.Items.Add(new ListViewItem(new string[] {
                        string.IsNullOrEmpty(rr.Label)?(rr.Source.Name):rr.Label,
                        string.IsNullOrEmpty(rr.Label)?rr.Source.GetType().Name:"",
                        string.Join(" , ", rr.Output.Shape),
                        (rr.Output.Data != null ? rr.Output.Data.Length : rr.Output.IntData.Length) + "",
                    weights
                    })
                    { Tag = rr.Output });
                }
            }

        }

        private InternalArray Normalize(InternalArray x)
        {
            float[] mean = new float[] { 0.485f, 0.456f, 0.406f };
            float[] std = new float[] { 0.229f, 0.224f, 0.225f };
            
            
            InternalArray ret = new InternalArray(x.Shape);
            
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < x.Shape[2]; j++)
                {
                    for (int k = 0; k < x.Shape[3]; k++)
                    {
                        var val = x.Get4D(0, i, j, k);                        
                        val -= mean[i];
                        val /= std[i];
                        ret.Set4D(0, i, j, k, val);
                    }
                }
            }
            return ret;
        }

        public static CifarItem ReadCifarImage(byte[] bb, long start)
        {
            var w = 32;
            var h = 32;
            Bitmap bmp = new Bitmap(32, 32);
            var label = bb[start];
            CifarItem ret = new CifarItem();
            ret.label = label;
            start++;

            NativeBitmap b = new NativeBitmap(bmp);


            for (int j = 0; j < h; j++)
            {
                for (int i = 0; i < w; i++)
                {
                    var _ind = (i * w + j);
                    b.SetPixel(j, i, new byte[] { bb[start + _ind], bb[start + 1024 + _ind], bb[start + 2048 + _ind], 0xff });
                }
            }

            ret.Bmp = b.GetBitmap();
            var x = new InternalArray(new int[] { 1, 3, 32, 32 });

            for (var dc = 0; dc < 3; dc++)
            {
                var i = 0;
                for (var xc = 0; xc < 32; xc++)
                {
                    for (var yc = 0; yc < 32; yc++)
                    {
                        var px = ret.Bmp.GetPixel(xc, yc);
                        var bt = (byte)((px.ToArgb() & (dc << 8)) >> 8);
                        x.Set4D(0, dc, xc, yc, bt / 255.0f - 0.5f);
                        i++;
                    }
                }
            }
            ret.x = x;

            return ret;
        }
        public void SetStatusInfo(string text)
        {
            statusStrip1.Invoke((Action)(() =>
            {
                toolStripStatusLabel1.Text = text;
            }));

        }
        public static CifarItem[] LoadCifarImages(string imgPath, Action<float> progressReport, bool withBitmap = false)
        {
            List<CifarItem> bmps = new List<CifarItem>();

            Stopwatch sw = new Stopwatch();
            sw.Start();
            var bytes = File.ReadAllBytes(imgPath);
            int imagesCnt = 1000;//10000
            long indexer = 0;
            for (int i = 0; i < imagesCnt; i++)
            {
                bmps.Add(ReadCifarImage(bytes, indexer));
                indexer += 3073;
                progressReport?.Invoke(i / (float)imagesCnt);
            }

            sw.Stop();
            var ms = sw.ElapsedMilliseconds;
            sw.Stop(); sw.Reset();
            sw.Start();
            /*if (withBitmap)
            {
                foreach (var mnistItem in bmps)
                {
                    mnistItem.GetBitmap();
                }
            }*/
            sw.Stop();
            var ms2 = sw.ElapsedMilliseconds;
            sw.Stop();



            return bmps.ToArray();

        }

        private void listView1_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (listView1.SelectedItems.Count == 0) return;
            var ar = listView1.SelectedItems[0].Tag as InternalArray;
            listView2.Items.Clear();
            int cntr = 0;
            if (checkBox1.Checked)
            {
                var cnt = int.Parse(textBox1.Text);
                for (int i = 0; i < cnt; i++)
                {
                    var item = ar.Data[ar.Data.Length - cnt + i];
                    listView2.Items.Add(new ListViewItem(new string[] { (ar.Data.Length - cnt + i) + "", item + "" }));
                }
            }
            else
            {
                foreach (var item in ar.Data)
                {
                    listView2.Items.Add(new ListViewItem(new string[] { cntr + "", item + "" }));
                    cntr++;
                    if (cntr > int.Parse(textBox1.Text)) break;
                }
            }
        }
    }
}
