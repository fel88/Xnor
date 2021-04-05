using System.Drawing;

namespace xnor
{
    public class MnistItem
    {
        public static int NewId;
        public MnistItem()
        {

            Id = NewId++;
            if (Id == 834)
            {

            }
        }
        public int Id;
        public byte[,] Data;
        public int Label;
        private NativeBitmap _bitmap = null;

        public NativeBitmap Bitmap
        {
            get
            {
                if (_bitmap == null)
                {
                    _bitmap = GetBitmap();
                }
                return _bitmap;
            }
            set
            {
                _bitmap = value;
            }
        }
        public NativeBitmap GetBitmap()
        {
            if (_bitmap != null) return _bitmap;
            Bitmap bmp = new Bitmap(Data.GetLength(0), Data.GetLength(1));
            NativeBitmap rom = new NativeBitmap(bmp);
            for (int i = 0; i < Data.GetLength(0); i++)
            {
                for (int j = 0; j < Data.GetLength(1); j++)
                {
                    rom.SetPixel(i, j, new[] { Data[i, j], Data[i, j], Data[i, j], (byte)255 });
                }
            }
            _bitmap = rom;
            return _bitmap;
        }
    }
}
