using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using DeepMagic;

namespace DeepLearning
{
    class Program
    {
        static void Main(string[] args)
        {
            double[][][] d = IO.LoadAllText(@"C:\DNN\test.txt");
            double[][] o = new DNN(@"C:\DNN\a.dnn").Compute(d[0]);

            ShowText(d[0], d[1], o);
            Console.ReadLine();
        }

        public static void ShowText(double[][] r, double[][] t, double[][] o)
        {
            int[] rightCount = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            int[] errorCount = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            for (int i = 0; i < t.Length; i++)
            {
                int a = IO.double2int(o[i]);
                int b = IO.double2int(t[i]);
                if (a == b)
                {
                    rightCount[b] = rightCount[b] + 1;
                }
                else
                {
                    errorCount[b] = errorCount[b] + 1;
                    for (int j = 0; j < r[i].Length; j++)
                    {
                        Console.Write(r[i][j]);
                        if (j % 32 == 0)
                            Console.WriteLine("");
                    }
                    Console.WriteLine(b + " is not " + a);
                }
            }
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine("\n" + i + " : " + (100.0 * rightCount[i] / (errorCount[i] + rightCount[i])) + "%");
            }

            Console.WriteLine("\n" + (100.0 * rightCount.Sum() / t.Length).ToString("0.00") + "%");
        }
        public static void ComputeImg(string s)
        {
            double[] d = IO.LoadImg(s);
            double[][] o = new DNN(@"C:\DNN\a.dnn").Compute(new double[][] { d });
            Console.WriteLine(IO.double2int(o[0]));
        }
    }
}