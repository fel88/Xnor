using System;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;

namespace xnor
{
    public static class NpyLoader
    {
        public static string GetProp(string str, string key)
        {
            var ind1 = str.IndexOf(key);
            var last = str.Substring(ind1);

            int cnt = 0;
            var ind2 = last.IndexOf(',');
            for (int i = 0; i < last.Length; i++)
            {
                if (last[i] == '(') { cnt++; }
                if (last[i] == ')')
                {
                    cnt--;
                    if (cnt == 0) { ind2 = i; break; }
                }
                if (last[i] == ',' && cnt == 0) { ind2 = i; break; }
            }

            //var ind3 = last.LastIndexOf(')');
            //var shp = last.Substring(ind2, ind3 - ind2);
            var shp = last.Substring(0, ind2);
            var ind3 = shp.IndexOf(':');
            shp = shp.Substring(ind3 + 1);
            return shp;
        }
        public static InternalArray Load(string filename)
        {
            var bts = File.ReadAllBytes(filename);
            return Load(bts);
        }
        public static InternalArray Load(byte[] bts)
        {

            ushort len = (ushort)((bts[0x6 + 0x2]) + (bts[0x6 + 0x2 + 1] << 8));
            var str = Encoding.UTF8.GetString(bts, 10, len);

            var shp = GetProp(str, "shape");
            var arr3 = shp.Split(new char[] { ',', '(', ')', ' ', }, StringSplitOptions.RemoveEmptyEntries).ToArray();
            var dims = arr3.Select(z => int.Parse(z)).ToArray();

            var descr = GetProp(str, "descr");

            int bytesPerItem = 4;
            bool isFloat = descr.Contains("<f4");

            InternalArray ret = new InternalArray(dims, isFloat);

            int cnt = 0;
            for (int i = 10 + len; i < bts.Length; i += bytesPerItem)
            {
                if (isFloat)
                {
                    var val = BitConverter.ToSingle(bts, i);
                    ret.Data[cnt] = val;
                }
                else
                {
                    var val = BitConverter.ToUInt32(bts, i);
                    ret.IntData[cnt] = val;
                }

                cnt++;
            }
            return ret;
        }
    }
}
