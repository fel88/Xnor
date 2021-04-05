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
    public partial class mdi : Form
    {
        public mdi()
        {
            InitializeComponent();
        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            Form1 f = new Form1();
            f.MdiParent = this;
            f.Show();
        }

        private void toolStripButton2_Click(object sender, EventArgs e)
        {
            cifar10 f = new cifar10();
            f.MdiParent = this;
            f.Show();
        }
    }
}
