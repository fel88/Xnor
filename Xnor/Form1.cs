using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;

namespace xnor
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();

            Fix.main(null);
            var n1 = THWrapper.THIntTensor_new();
            THWrapper.THIntTensor_resize2d(n1, 10, 20);

            var bts = BitConverter.GetBytes(-422);
            var res = BitConverter.ToUInt32(bts, 0);
            THWrapper.THIntTensor_set2d(n1, 0, 0, res);
            var v1 = THWrapper.THIntTensor_get2d(n1, 0, 0);
            var bts2 = BitConverter.GetBytes(v1);
            var restore = BitConverter.ToInt32(bts2, 0);

            //var n1 = THWrapper.THFloatTensor_new();
            //THWrapper.THFloatTensor_resize2d(n1, 10, 20);
            //var v1 = THWrapper.THFloatTensor_get2d(n1, 0, 0);
            //THWrapper.THFloatTensor_fill(n1, 11);
            //var dims=THWrapper.THFloatTensor_nDimension(n1);
            //for (int i = 0; i < dims; i++)
            //{
            //    var a1 = THWrapper.THFloatTensor_size(n1, i);
            //}



            //var v2 = THWrapper.THFloatTensor_get2d(n1, 0, 0);
            //THWrapper.THFloatTensor_set2d(n1, 0, 0,-4.4f);
            //var v3 = THWrapper.THFloatTensor_get2d(n1, 0, 0);

            //var di = THWrapper.THFloatTensor_data(n1);
            //byte[] dest = new byte[4 * 4];
            //Marshal.Copy(di, dest, 0, 16);

            //var aaaa= BitConverter.ToSingle(dest, 0);
            //var aaaa2 = BitConverter.ToSingle(dest, 4);

            //var int1=InternalArray.FromTHTensor(n1);

            //var bts = BitConverter.GetBytes(-12.143f);
            //for (int i = 0; i < bts.Length; i++)
            //{
            //    dest[i] = bts[i];
            //}

            //Marshal.Copy(dest, 0, di, 16);
            //v3 = THWrapper.THFloatTensor_get2d(n1, 0, 0);
            //THWrapper.THFloatTensor_free(n1);

            net = new FPBinLeNetTest();
            imgs = Stuff.LoadImages("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
            sample = NpyLoader.Load("1.npy");

            net.LoadFromZip("BinLenet.zip");
            net.PrepareData();

        }
        InternalArray sample;
        MnistItem[] imgs;

        FPBinLeNetTest net;
        Random rand;

        int run(int sampleId, bool withUi = true)
        {


            var item = imgs[sampleId];
            if (withUi)
            {
                pictureBox1.Image = item.GetBitmap().GetBitmap();

                label1.Text = item.Label + "";
            }
            var inp = ToArray(item);
            InternalArray inp2 = new InternalArray(inp.Shape);
            if (withUi)
            {
                if (checkBox1.Checked)
                {
                    inp = inp2;
                }
                if (checkBox3.Checked)
                {
                    inp = new InternalArray(inp.Shape);
                    for (int ii = 0; ii < inp.Data.Length; ii++)
                    {
                        inp.Data[ii] = 1;
                    }
                }
                if (checkBox4.Checked)
                {
                    inp = sample;

                }
            }
            var res = net.Forward(inp);
            int maxi = -1;

            for (int i = 0; i < res.Data.Length; i++)
            {
                if (maxi == -1 || res.Data[i] > res.Data[maxi])
                {
                    maxi = i;
                }
            }
            if (withUi)
            {


                listView1.Items.Clear();
                foreach (var rr in net.results)
                {
                    /* bool hasNonValue = false;
                     if (rr.Value.Data != null)
                     {
                         hasNonValue = rr.Value.Data.Any(z => float.IsNaN(z) || float.IsInfinity(z));

                     }*/

                    var fr = rr.Source;
                    string weights = "";
                    if (fr is Conv2d c2)
                    {
                        if (c2.Weight.Data != null)
                        {
                            weights = c2.Weight.Data.Length.ToString("N0");
                        }
                        if (c2.Weight.QIntData != null)
                        {
                            weights = c2.Weight.QIntData.Length.ToString("N0");
                        }
                        
                    }
                    if (fr is BinConv2d c3)
                    {
                        weights = c3.Weight.IntData.Length.ToString("N0");
                    }
                    if (fr is Linear lin)
                    {
                        if (lin.Weight.Data != null)
                        {
                            weights = lin.Weight.Data.Length.ToString("N0");
                        }
                        if (lin.Weight.QIntData != null)
                        {
                            weights = lin.Weight.QIntData.Length.ToString("N0");
                        }
                    }
                    if (fr is BinLinear blin)
                    {
                        weights = blin.Weight.IntData.Length.ToString("N0");
                    }
                    if (fr is BatchNorm2d bn && !(fr is FPBatchNorm2d))
                    {
                        weights = (bn.Weight.Data.Length + bn.Bias.Data.Length + bn.RunningMean.Data.Length + bn.RunningVar.Data.Length).ToString("N0");
                    }
                    int len = 0;
                    if (rr.Output.Data != null)
                    {
                        len = rr.Output.Data.Length;
                    }
                    if (rr.Output.IntData != null)
                    {
                        len = rr.Output.IntData.Length;
                    }
                    if (rr.Output.QIntData != null)
                    {
                        len = rr.Output.QIntData.Length;
                    }
                    listView1.Items.Add(new ListViewItem(new string[]
                    {
                        string.IsNullOrEmpty(rr.Label)?rr.Source.Name:rr.Label,
                        rr.Source!=null?rr.Source.GetType().Name:"",
                        string.Join(" , ", rr.Output.Shape),
                        len.ToString(),
                        weights
                    })
                    { Tag = rr.Output });
                }
            }
            return maxi;
        }
        private void button1_Click(object sender, EventArgs e)
        {
            Stopwatch sw = Stopwatch.StartNew();
            rand = new Random();

            //for (int i = 0; i < 128; i++)
            {


                //while (true)

                var index = rand.Next(imgs.Length);

                var res = run(index);

                label2.Text = res + "";
                label2.BackColor = res == imgs[index].Label ? Color.Green : Color.Yellow;
                label2.ForeColor = res == imgs[index].Label ? Color.White : Color.Blue;
                //  if (res != imgs[index].Label) break;
            }

            sw.Stop();
            var ms = sw.ElapsedMilliseconds;
            toolStripStatusLabel1.Text = ms + "ms";



        }

        public InternalArray ToArray(MnistItem item)
        {
            InternalArray ar = new InternalArray(new int[] { 1, 1, 28, 28 });
            for (int i = 0; i < item.Data.GetLength(0); i++)
            {
                for (int j = 0; j < item.Data.GetLength(1); j++)
                {
                    float v = item.Data[i, j] / 255f;
                    v -= 0.1307f;
                    v /= 0.3081f;
                    ar.Set4D(0, 0, j, i, v);
                }
            }

            return ar;
        }

        private void listView1_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (listView1.SelectedItems.Count == 0) return;
            var ar = listView1.SelectedItems[0].Tag as InternalArray;
            listView2.Items.Clear();
            int cntr = 0;
            if (checkBox2.Checked)
            {
                var cnt = int.Parse(textBox1.Text);
                for (int i = 0; i < cnt; i++)
                {
                    if (ar.Data != null)
                    {
                        var item = ar.Data[ar.Data.Length - cnt + i];
                        listView2.Items.Add(new ListViewItem(new string[] { (ar.Data.Length - cnt + i) + "", item + "" }));
                    }
                    else
                    {
                        var item = ar.QIntData[ar.Data.Length - cnt + i];
                        listView2.Items.Add(new ListViewItem(new string[] { (ar.QIntData.Length - cnt + i) + "", item + "" }));
                    }
                }
            }
            else
            {
                if (ar.QIntData != null)
                {
                    foreach (var item in ar.QIntData)
                    {

                        listView2.Items.Add(new ListViewItem(new string[] { cntr + "", item + "" }));
                        cntr++;
                        if (cntr > int.Parse(textBox1.Text)) break;

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

        private void button2_Click(object sender, EventArgs e)
        {
            run(0);
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {

        }

        private void listView2_SelectedIndexChanged(object sender, EventArgs e)
        {

        }
        Thread th;
        CancellationToken token;
        CancellationTokenSource cancelTokenSource;
        private void button3_Click(object sender, EventArgs e)
        {
            if (th != null)
            {
                cancelTokenSource.Cancel();
                button3.Text = "run";
                return;
            }
            button3.Text = "stop";
            cancelTokenSource = new CancellationTokenSource();
            token = cancelTokenSource.Token; int good = 0;
            int total = 0;
            th = new Thread(() =>
           {
               for (int i = 0; i < imgs.Length; i++)
               {
                   if (token.IsCancellationRequested)
                   {
                       break;
                   }
                   var ans = run(i, false);
                   if (imgs[i].Label == ans)
                   {
                       good++;
                   }
                   total++;
                   float perc2 = (i / (float)imgs.Length) * 100f;
                   float perc3 = (good / (float)total) * 100f;
                   progressBar1.Invoke((Action)(() =>
                   {
                       progressBar1.Value = (int)Math.Round(perc2);
                       label4.Text = i + " / " + imgs.Length + ": " + Math.Round(perc3, 2) + "%";
                   }));
               }
               th = null;
           });
            th.IsBackground = true;
            th.Start();

            float perc = (good / (float)imgs.Length) * 100f;
        }

        private void maxPoolDebugToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (listView1.SelectedItems.Count == 0) return;
            var ar = listView1.SelectedItems[0].Tag as InternalArray;
            maxPoolDebugger d = new maxPoolDebugger();
            InternalArray arr = new InternalArray(new int[] { 1, 1, 6, 6 });
            Random r = new Random();
            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    arr.Set4D(0, 0, i, j, r.Next(10));
                }
            }
            d.Init(arr);
            d.Show();

        }
    }
    public class BinVGG16 : NeuralItem
    {
        public override InternalArray Forward(InternalArray ar)
        {
            throw new NotImplementedException();
        }
    }
}
