using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeepMagic;

namespace Server
{
    class Program
    {
        static void Main(string[] args)
        {
            double[][][] io = IO.LoadAllText(@"C:\0\Lab\DNN\train.txt");
            new DNN(@"C:\0\Lab\DNN\a.dnn").TrainSupervised(io[0], io[1]);
        }
    }
}
