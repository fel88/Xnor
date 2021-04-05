using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace xnor
{
    public class matmul
    {
        public const int ENCODE_BIT = 32;
        public static UInt32 MASK(UInt32 a)
        {
            var ret = ((a) + (-(a) & -(((0) > (a)) ? 1 : 0)));
            return (UInt32)ret;
        }

        public static int popcnt32(UInt32 a)
        {
            int ret = 0;
            for (int i = 0; i < 32; i++)
            {
                ret += (a & (1 << i)) > 0 ? 1 : 0;
            }
            return ret;
        }

        public const int MR = 4;
        public const int NR = 4;
        public const int MC = 256;
        public const int NC = 256;
        public const int KC = 64;

        //
        //  Packing complete panels from A (i.e. without padding)
        //
        //pack_MRxk(int k, uint32_t *A, int incRowA, int incColA, uint32_t *buffer){
        public static void
        pack_MRxk(int k, IntPtr A, int incRowA, int incColA, IntPtr buffer)
        {
            int i, j;

            for (j = 0; j < k; ++j)
            {
                for (i = 0; i < MR; ++i)
                {
                    SetUint32(buffer, i, GetUint32(A, i * incRowA));
                    //buffer[i] = A[i * incRowA];
                }
                buffer += 4 * MR;
                A += 4 * incColA;

            }
        }

        static matmul()
        {
            _A = Marshal.AllocHGlobal(sizeof(uint) * MC * KC);
            _B = Marshal.AllocHGlobal(sizeof(uint) * KC * NC);
        }
        //
        //  Packing panels from A with padding if required
        //
        //pack_A(int mc, int kc, uint32_t *A, int incRowA, int incColA, uint32_t *buffer){
        public static void pack_A(int mc, int kc, IntPtr A, int incRowA, int incColA, IntPtr buffer)
        {
            IntPtr origBuf = buffer;
            int i, j, mp = mc / MR, _mr = mc % MR;

            /*for (int ii = 0; ii < 10; ii++)
            {
                var ress = GetUint32(A, ii);
            }*/
            for (i = 0; i < mp; ++i)
            {
                pack_MRxk(kc, A, incRowA, incColA, buffer);

                buffer += 4 * kc * MR;
                A += 4 * MR * incRowA;
            }
            /* for (int ii = 0; ii < 10; ii++)
             {
                 var ress = GetUint32(origBuf, ii);
             }*/
            if (_mr > 0)
            {
                for (j = 0; j < kc; ++j)
                {
                    for (i = 0; i < _mr; ++i)
                    {
                        SetUint32(buffer, i, GetUint32(A, i * incRowA));
                        //buffer[i] = A[i * incRowA];
                    }
                    for (i = _mr; i < MR; ++i)
                    {
                        SetUint32(buffer, i, UBIT);

                        //buffer[i] = UBIT;
                    }

                    buffer += 4 * MR;
                    A += 4 * incColA;
                }
            }
            /*for (int ii = 0; ii < 10; ii++)
            {
                var ress = GetUint32(origBuf, ii);
            }
            for (int ii = 0; ii < 10; ii++)
            {
                var aaz = GetUint32(_A, ii);

            }*/
        }
        const UInt32 UBIT = unchecked((uint)(~0));
        //
        //  Packing complete panels from B (i.e. without padding)
        //
        //pack_kxNR(int k, uint32_t *B, int incRowB, int incColB, uint32_t *buffer){
        public static void pack_kxNR(int k, IntPtr B, int incRowB, int incColB, IntPtr buffer)
        {
            int i, j;

            for (i = 0; i < k; ++i)
            {
                for (j = 0; j < NR; ++j)
                {
                    SetUint32(buffer, j, GetUint32(B, j * incColB));
                    //buffer[j] = B[j * incColB];
                }
                buffer += 4 * NR;
                B += 4 * incRowB;

            }
        }

        //
        //  Packing panels from B with padding if required
        //
        //pack_B(int kc, int nc, uint32_t *B, int incRowB, int incColB, uint32_t *buffer){
        public static void pack_B(int kc, int nc, IntPtr B, int incRowB, int incColB, IntPtr buffer)
        {
            int i, j, np = nc / NR, _nr = nc % NR;



            for (j = 0; j < np; ++j)
            {
                pack_kxNR(kc, B, incRowB, incColB, buffer);
                buffer += 4 * kc * NR;
                B += 4 * NR * incColB;

            }
            if (_nr > 0)
            {
                for (i = 0; i < kc; ++i)
                {
                    for (j = 0; j < _nr; ++j)
                    {
                        SetUint32(buffer, j, GetUint32(B, j * incColB));
                        //buffer[j] = B[j * incColB];
                    }
                    for (j = _nr; j < NR; ++j)
                    {
                        SetUint32(buffer, j, UBIT);
                        //buffer[j] = UBIT;
                    }

                    buffer += 4 * NR;
                    B += 4 * incRowB;
                }
            }
        }

        static IntPtr _A;
        static IntPtr _B;

        //
        //  Macro Kernel for the multiplication of blocks of A and B.  We assume that
        //  these blocks were previously packed to buffers _A and _B.
        //
        public static void
        dgemm_macro_kernel(int mc,
                           int nc,
                           int kc,
                           int beta,
                           IntPtr C,
                           int incRowC,
                           int incColC)
        {
            int mp = (mc + MR - 1) / MR;
            int np = (nc + NR - 1) / NR;

            int _mr = mc % MR;
            int _nr = nc % NR;

            int mr, nr;
            int i, j;
            IntPtr origC = C;
            //#pragma omp parallel shared(C) private(i,j,nr,mr)
            {
                //#pragma omp for schedule(dynamic)
                for (j = 0; j < np; ++j)
                {
                    nr = (j != np - 1 || _nr == 0) ? NR : _nr;
                    for (i = 0; i < mp; ++i)
                    {
                        mr = (i != mp - 1 || _mr == 0) ? MR : _mr;

                        //List<float> ll2 = new List<float>();
                        if (origC != C)
                        {
                            throw new Exception();
                        }
                        /*  for (int ii = 0; ii < 10; ii++)
                          {
                              var val = GetFloat(C, ii);
                              ll2.Add(val);
                          }*/

                        if (mr == MR && nr == NR)
                        {
                            dgemm_micro_kernel(mr, nr, kc, _A + 4 * i * kc * MR, _B + 4 * j * kc * NR, beta, C + 4 * (i * MR * incRowC + j * NR * incColC), incRowC, incColC);
                        }
                        else
                        {
                            dgescal(mr, nr, beta, C + 4 * (i * MR * incRowC + j * NR * incColC), incRowC, incColC);
                            dgemm_micro_kernel(mr, nr, kc, _A + 4 * i * kc * MR, _B + 4 * j * kc * NR, 0, C + 4 * (i * MR * incRowC + j * NR * incColC), incRowC, incColC);

                        }
                    }
                }
            }
        }

        //
        //  Compute C <- beta*C + A*B, beta = 0 or 1
        //
        /*
         dgemm_nn(int            m,
         int            n,
         int            kk,
         uint32_t      *A,
         int            incRowA,
         int            incColA,
         uint32_t      *B,
         int            incRowB,
         int            incColB,
         float         *C,
         int            incRowC,
         int            incColC,
         int            beta,
         int            alpha,
         float         *alphas)
{*/
        public static void dgemm_nn(int m,
                 int n,
                 int kk,
                 IntPtr A,
                 int incRowA,
                 int incColA,
                 IntPtr B,
                 int incRowB,
                 int incColB,
                 IntPtr C,
                 int incRowC,
                 int incColC,
                 int beta,
                 int alpha,
                 IntPtr alphas)
        {
            int i, j, l, k = 1 + (kk - 1) / ENCODE_BIT;
            int mb = (m + MC - 1) / MC;
            int nb = (n + NC - 1) / NC;
            int kb = (k + KC - 1) / KC;

            int _mc = m % MC;
            int _nc = n % NC;
            int _kc = k % KC;

            int mc, nc, kc;

            int _beta;

            for (j = 0; j < nb; ++j)
            {
                nc = (j != nb - 1 || _nc == 0) ? NC : _nc;

                for (l = 0; l < kb; ++l)
                {
                    kc = (l != kb - 1 || _kc == 0) ? KC : _kc;
                    _beta = (l == 0) ? beta : 1;

                    pack_B(kc, nc,
                           B + 4 * (l * KC * incRowB + j * NC * incColB), incRowB, incColB,
                           _B);

                    for (i = 0; i < mb; ++i)
                    {
                        mc = (i != mb - 1 || _mc == 0) ? MC : _mc;

                        pack_A(mc, kc,
                               A + 4 * (i * MC * incRowA + l * KC * incColA), incRowA, incColA,
                               _A);

                        /*List<uint> afterPack = new List<uint>();
                        for (int ii = 0; ii < 10; ii++)
                        {
                            var aaz = GetUint32(_A, ii);
                            afterPack.Add(aaz);
                        }*/


                        dgemm_macro_kernel(mc, nc, kc, _beta,
                                           C + 4 * (i * MC * incRowC + j * NC * incColC),
                                           incRowC, incColC);
                    }
                }
            }

            /*List<float> afterLoop1 = new List<float>();
            for (int ii = 0; ii < 100; ii++)
            {
                var ress = GetFloat(C, ii);
                afterLoop1.Add(ress);
            }
            */

            //#pragma omp parallel for schedule(dynamic,1) collapse(2)
            var fl0 = GetFloat(C, 0);
            for (i = 0; i < m; i++)
            {
                for (j = 0; j < n; j++)
                {
                    SetFloat(C, i * n + j, GetFloat(C, i * n + j) + kk);
                    //C[i * n + j] += kk;
                }
            }
            if (alpha != 0)
            {
                for (i = 0; i < m; i++)
                {
                    for (j = 0; j < n; j++)
                    {
                        //C[i * n + j] *= alphas[i];
                        var aa1 = GetFloat(C, i * n + j);
                        var aa2 = GetFloat(alphas, i);
                        SetFloat(C, i * n + j, aa1 * aa2);
                        //C[i * n + j] = (float)(C[i * n + j] * alphas[i]);
                    }
                }
            }
        }
        public static void fpdgemm_nn(int m,
                 int n,
                 int kk,
                 IntPtr A,
                 int incRowA,
                 int incColA,
                 IntPtr B,
                 int incRowB,
                 int incColB,
                 IntPtr C,
                 int incRowC,
                 int incColC,
                 int beta
               )
        {
            int i, j, l, k = 1 + (kk - 1) / ENCODE_BIT;
            int mb = (m + MC - 1) / MC;
            int nb = (n + NC - 1) / NC;
            int kb = (k + KC - 1) / KC;

            int _mc = m % MC;
            int _nc = n % NC;
            int _kc = k % KC;

            int mc, nc, kc;

            int _beta;

            for (j = 0; j < nb; ++j)
            {
                nc = (j != nb - 1 || _nc == 0) ? NC : _nc;

                for (l = 0; l < kb; ++l)
                {
                    kc = (l != kb - 1 || _kc == 0) ? KC : _kc;
                    _beta = (l == 0) ? beta : 1;

                    pack_B(kc, nc,
                           B + 4 * (l * KC * incRowB + j * NC * incColB), incRowB, incColB,
                           _B);

                    for (i = 0; i < mb; ++i)
                    {
                        mc = (i != mb - 1 || _mc == 0) ? MC : _mc;

                        pack_A(mc, kc,
                               A + 4 * (i * MC * incRowA + l * KC * incColA), incRowA, incColA,
                               _A);

                        /*List<uint> afterPack = new List<uint>();
                        for (int ii = 0; ii < 10; ii++)
                        {
                            var aaz = GetUint32(_A, ii);
                            afterPack.Add(aaz);
                        }*/


                        dgemm_macro_kernel(mc, nc, kc, _beta,
                                           C + 4 * (i * MC * incRowC + j * NC * incColC),
                                           incRowC, incColC);
                    }
                }
            }

            /*List<float> afterLoop1 = new List<float>();
            for (int ii = 0; ii < 100; ii++)
            {
                var ress = GetFloat(C, ii);
                afterLoop1.Add(ress);
            }
            */

            //#pragma omp parallel for schedule(dynamic,1) collapse(2)
            var fl0 = GetFloat(C, 0);
            for (i = 0; i < m; i++)
            {
                for (j = 0; j < n; j++)
                {
                    SetFloat(C, i * n + j, GetFloat(C, i * n + j) + kk);
                    //C[i * n + j] += kk;
                }
            }

        }

        //
        //  Compute X *= alpha
        //
        public static void dgescal(int m,
                 int n,
                 int beta,
                 IntPtr X,
                 int incRowX,
                 int incColX)
        {
            int i, j;

            if (beta == 0)
            {
                //#pragma omp parallel for schedule(dynamic,1) collapse(2)
                for (j = 0; j < n; ++j)
                {
                    for (i = 0; i < m; ++i)
                    {
                        SetFloat(X, i * incRowX + j * incColX, 0);
                    }
                }
            }
        }

        public static float GetFloat(IntPtr p, int pos, int elemSize = 4)
        {
            byte[] temp = new byte[4];
            Marshal.Copy(p + pos * elemSize, temp, 0, 4);
            return BitConverter.ToSingle(temp, 0);
        }
        public static void SetFloat(IntPtr p, int pos, float val, int elemSize = 4)
        {
            var temp = BitConverter.GetBytes(val);
            Marshal.Copy(temp, 0, p + pos * elemSize, 4);
        }
        public static void SetUint32(IntPtr p, int pos, UInt32 val, int elemSize = 4)
        {
            var temp = BitConverter.GetBytes(val);
            Marshal.Copy(temp, 0, p + pos * elemSize, 4);
        }
        public static UInt32 GetUint32(IntPtr p, int pos, int elemSize = 4)
        {
            byte[] temp = new byte[4];
            Marshal.Copy(p + pos * elemSize, temp, 0, 4);
            return BitConverter.ToUInt32(temp, 0);
        }
        //
        //  Micro kernel for multiplying panels from A and B.
        //
        //dgemm_micro_kernel(int m, int n, int kc, uint32_t *A, uint32_t *B, int beta, float *C, int incRowC, int incColC)
        public static void dgemm_micro_kernel(int m, int n, int kc, IntPtr A, IntPtr B, int beta, IntPtr C, int incRowC, int incColC)
        {
            int[] AB = new int[MR * NR];
            int i, j, l;

            //
            //  Compute AB = A*B
            //
            //#pragma omp parallel for
            for (l = 0; l < MR * NR; ++l)
            {
                AB[l] = 0;
            }

            for (l = 0; l < kc; ++l)
            {
                for (j = 0; j < NR; ++j)
                {
                    for (i = 0; i < MR; ++i)
                    {
                        AB[i + j * MR] -= popcnt32(MASK(GetUint32(A, i) ^ GetUint32(B, j))) << 1;
                    }
                }

                A += 4 * MR;
                B += 4 * NR;

            }

            //
            //  Update C <- beta*C
            //
            if (beta == 0)
            {
                //#pragma omp for collapse(2)
                for (j = 0; j < n; ++j)
                {
                    for (i = 0; i < m; ++i)
                    {
                        SetFloat(C, i * incRowC + j * incColC, GetFloat(C, i * incRowC + j * incColC) + 0);
                        //C[i * incRowC + j * incColC] += 0;
                    }
                }
            }
            //#pragma omp for collapse(2)
            for (j = 0; j < n; ++j)
            {
                for (i = 0; i < m; ++i)
                {
                    var bef = GetFloat(C, i * incRowC + j * incColC);
                    SetFloat(C, i * incRowC + j * incColC, bef + AB[i + j * MR]);

                    //C[i * incRowC + j * incColC] += AB[i + j * MR];
                }
            }

            List<float> l1 = new List<float>();
            for (int ii = 0; ii < 10; ii++)
            {
                var fl = GetFloat(C, ii);
                l1.Add(fl);
            }
        }
    }
}

