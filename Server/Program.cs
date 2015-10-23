using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeepMagic;
using System.Threading;

namespace Server
{
    class Program
    {
        static void Main(string[] args)
        {
            double[][][] io = IO.LoadAllText(@"C:\DNN\train.txt");
            DNN d = new DNN(@"C:\DNN\a.dnn");
            new Task(() => { d.Train(io[0], io[1]); }).Start();

            while (true)
            {
                Thread.Sleep(1000 * 10);
                Console.WriteLine(d.e);
                d.Save();
            }
        }
    }
}
