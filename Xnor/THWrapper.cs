using System;
using System.Runtime.InteropServices;

namespace xnor
{
    public class THWrapper
    {
        [DllImport("th.dll")]
        public static extern IntPtr THFloatTensor_new();
        [DllImport("th.dll")]
        public static extern IntPtr THIntTensor_new();
        [DllImport("th.dll")]
        public static extern void THFloatTensor_free(IntPtr tensor);
        [DllImport("th.dll")]
        public static extern void THIntTensor_free(IntPtr tensor);

        [DllImport("th.dll")]
        public static extern void THFloatTensor_transpose(IntPtr tensor, IntPtr tensor2, int dimension1_, int dimension2_);
        [DllImport("th.dll")]
        public static extern IntPtr THFloatTensor_newTranspose(IntPtr tensor, int dimension1_, int dimension2_);
        [DllImport("th.dll")]
        public static extern void THIntTensor_transpose(IntPtr tensor, IntPtr tensor2, int dimension1_, int dimension2_);


        [DllImport("th.dll")]
        public static extern void THFloatTensor_resize2d(IntPtr tensor, long sz1, long sz2);
        [DllImport("th.dll")]
        public static extern void THIntTensor_resize2d(IntPtr tensor, long sz1, long sz2);

        [DllImport("th.dll")]
        public static extern void THFloatTensor_resize3d(IntPtr tensor, long sz1, long sz2, long sz3);
        [DllImport("th.dll")]
        public static extern void THIntTensor_resize3d(IntPtr tensor, long sz1, long sz2, long sz3);
        [DllImport("th.dll")]
        public static extern void THFloatTensor_resize4d(IntPtr tensor, long sz1, long sz2, long sz3, long sz4);
        [DllImport("th.dll")]
        public static extern void THIntTensor_resize4d(IntPtr tensor, long sz1, long sz2, long sz3, long sz4);
        [DllImport("th.dll")]

        public static extern float THFloatTensor_get2d(IntPtr tensor, long sz1, long sz2);
        [DllImport("th.dll")]

        public static extern uint THIntTensor_get2d(IntPtr tensor, long sz1, long sz2);
        [DllImport("th.dll")]

        public static extern float THFloatTensor_get3d(IntPtr tensor, long sz1, long sz2, long sz3);
        [DllImport("th.dll")]

        public static extern uint THIntTensor_get3d(IntPtr tensor, long sz1, long sz2, long sz3);
        [DllImport("th.dll")]

        public static extern float THFloatTensor_get4d(IntPtr tensor, long sz1, long sz2, long sz3, long sz4);
        [DllImport("th.dll")]

        public static extern uint THIntTensor_get4d(IntPtr tensor, long sz1, long sz2, long sz3, long sz4);
        [DllImport("th.dll")]

        public static extern void THFloatTensor_set2d(IntPtr tensor, long sz1, long sz2, float val);
        [DllImport("th.dll")]

        public static extern void THFloatTensor_set3d(IntPtr tensor, long sz1, long sz2, long sz3, float val);
        [DllImport("th.dll")]

        public static extern void THFloatTensor_set4d(IntPtr tensor, long sz1, long sz2, long sz3, long sz4, float val);
        [DllImport("th.dll")]

        public static extern void THIntTensor_set4d(IntPtr tensor, long sz1, long sz2, long sz3, long sz4, uint val);
        [DllImport("th.dll")]

        public static extern void THIntTensor_set3d(IntPtr tensor, long sz1, long sz2, long sz3, uint val);
        [DllImport("th.dll")]

        public static extern void THIntTensor_set2d(IntPtr tensor, long sz1, long sz2, uint val);
        [DllImport("th.dll")]

        public static extern void THFloatTensor_fill(IntPtr tensor, float sz1);
        [DllImport("th.dll")]

        public static extern void THFloatTensor_zero(IntPtr tensor);

        [DllImport("th.dll")]

        public static extern long THFloatTensor_size(IntPtr tensor, int dim);
        [DllImport("th.dll")]

        public static extern long THIntTensor_size(IntPtr tensor, int dim);

        [DllImport("th.dll")]

        public static extern int THFloatTensor_nDimension(IntPtr tensor);
        [DllImport("th.dll")]

        public static extern int THIntTensor_nDimension(IntPtr tensor);


        [DllImport("th.dll")]

        public static extern long THFloatTensor_storageOffset(IntPtr tensor);
        [DllImport("th.dll")]

        public static extern IntPtr THFloatTensor_storage(IntPtr tensor);
        [DllImport("th.dll")]

        public static extern void THFloatTensor_addmm(IntPtr tensor, float beta, IntPtr t, float alpha, IntPtr mat1, IntPtr mat2);

        [DllImport("th.dll")]

        public static extern IntPtr THFloatTensor_newWithStorage2d(IntPtr tensor, long storageOffset_, long size0_, long stride0_,
                                long size1_, long stride1_);
        [DllImport("th.dll")]

        public static extern IntPtr THFloatTensor_data(IntPtr tensor);
        [DllImport("th.dll")]

        public static extern IntPtr THIntTensor_data(IntPtr tensor);
        [DllImport("th.dll")]

        public static extern IntPtr THFloatTensor_newSelect(IntPtr tensor, int dimension_, long sliceIndex_);
        [DllImport("th.dll")]

        public static extern IntPtr THIntTensor_newSelect(IntPtr tensor, int dimension_, long sliceIndex_);


    }
}

