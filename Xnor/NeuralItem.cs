using System;
using System.Collections.Generic;
using System.IO.Compression;

namespace xnor
{
    public abstract class NeuralItem
    {

        public string Name { get; set; }
        public NeuralItem Parent;

        public abstract InternalArray Forward(InternalArray ar);

        public void LoadStateDict(string path)
        {
            throw new NotImplementedException();

        }
        public long LastMs;
        public virtual NeuralItem[] Childs { get => null; }

        public virtual int SetData(List<InternalArray> arrays)
        {
            return 0;
        }

        public virtual void LoadFromZip(ZipArchiveEntry[] ww)
        {
            
        }
    }
}
