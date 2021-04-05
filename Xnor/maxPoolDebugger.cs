using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace xnor
{
    public partial class maxPoolDebugger : Form
    {
        public maxPoolDebugger()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Redraw();
        }

        InternalArray input;

        public void Redraw( )
        {
            var img = input.Get2DImageFrom4DArray(0, 0);
            Bitmap bmp = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            var gr = Graphics.FromImage(bmp);
            gr.Clear(Color.White);
            int ww = int.Parse(textBox1.Text);
            for (int i = 0; i < img.Shape[0]; i++)
            {
                for (int j = 0; j < img.Shape[1]; j++)
                {
                    gr.DrawRectangle(Pens.Black, j * ww, i * ww, ww, ww);
                    gr.DrawString(Math.Round(img.Get2D(i, j), 3) + "", new Font("Consolas",10), Brushes.Black, j * ww, i * ww);
                }
            }
            pictureBox1.Image = bmp;
            MaxPool2d pool = new MaxPool2d(2, 2);
            var res = pool.Forward(input);
            var img2 = res.Get2DImageFrom4DArray(0, 0);


            Bitmap bmp2 = new Bitmap(pictureBox2.Width, pictureBox2.Height);
            var gr2 = Graphics.FromImage(bmp2);
            gr2.Clear(Color.White);
            
            for (int i = 0; i < img2.Shape[0]; i++)
            {
                for (int j = 0; j < img2.Shape[1]; j++)
                {
                    gr2.DrawRectangle(Pens.Black, j * ww, i * ww, ww, ww);
                    gr2.DrawString(Math.Round(img2.Get2D(i, j), 3) + "", new Font("Consolas", 10), Brushes.Black, j * ww, i * ww);
                }
            }
            pictureBox2.Image = bmp2;
        }

        internal void Init(InternalArray ar)
        {
            input = ar;
        }
    }
}
