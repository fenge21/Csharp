using Accord.Neuro;
using Accord.Neuro.ActivationFunctions;
using Accord.Neuro.Learning;
using Accord.Neuro.Networks;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace DeepMagic
{
    public class DNN
    {
        private string dl { get; set; }
        private DeepBeliefNetwork n { get; set; }
        public DNN(string dl) { this.dl = dl; }
        public void TrainSupervised(double[][] inputs, double[][] outputs)
        {
            if (n == null)
            {
                if (File.Exists(dl))
                {
                    while (n == null)
                    {
                        try
                        {
                            n = DeepBeliefNetwork.Load(dl);
                        }
                        catch (Exception)
                        {
                            Thread.Sleep(500);
                        }
                    }
                }
                else
                {
                    n = new DeepBeliefNetwork(new BernoulliFunction(), inputs[0].Length, inputs[0].Length, inputs[0].Length, outputs[0].Length);
                    new GaussianWeights(n).Randomize();
                }
            }

            var t = new DeepNeuralNetworkLearning(n) { Algorithm = (ann, i) => new ParallelResilientBackpropagationLearning(ann), LayerIndex = n.Machines.Count - 1, };
            double e = 100;
            new Task(() => { while (true) { e = t.RunEpoch(t.GetLayerInput(inputs), outputs); } }).Start();
            while (true)
            {
                Console.WriteLine(e);
                n.UpdateVisibleWeights();
                try
                {
                    n.Save(dl);
                }
                catch (Exception)
                {

                }
                Thread.Sleep(1000*10);
            }
        }
        //public void TrainUnSupervised(double[][] inputs)
        //{
        //    if (n == null)
        //    {
        //        if (File.Exists(dl))
        //        {
        //            while (n == null)
        //            {
        //                try
        //                {
        //                    n = DeepBeliefNetwork.Load(dl);
        //                }
        //                catch (Exception)
        //                {
        //                    Thread.Sleep(500);
        //                }
        //            }
        //        }
        //        else
        //        {
        //            n = new DeepBeliefNetwork(new BernoulliFunction(), inputs[0].Length, 1024, 500, 100, 10);
        //            new GaussianWeights(n).Randomize();
        //        }
        //    }

        //    var t = new DeepBeliefNetworkLearning(n) { Algorithm = (h, v, i) => new ContrastiveDivergenceLearning(h, v), LayerIndex = n.Machines.Count - 1, };
        //    double e = 100;
        //    new Task(() => { while (true) e = t.RunEpoch(t.GetLayerInput(inputs)); }).Start();
        //    while (true)
        //    {
        //        Console.WriteLine(e);
        //        n.UpdateVisibleWeights();
        //        try
        //        {
        //            n.Save(dl);
        //        }
        //        catch (Exception)
        //        {

        //        }
        //        Thread.Sleep(2000);
        //    }
        //} //怎么用？还不知道
        public double[][] Compute(double[][] inputs)
        {
            if (n == null)
            {
                if (File.Exists(dl))
                {
                    while (n == null)
                    {
                        try
                        {
                            n = DeepBeliefNetwork.Load(dl);
                        }
                        catch (Exception)
                        {
                            Thread.Sleep(500);
                        }
                    }
                }
                else
                {
                    throw new Exception("No DNN Found!");
                }
            }

            List<double[]> d = new List<double[]>();
            for (int i = 0; i < inputs.Length; i++)
            {
                d.Add(n.Compute(inputs[i]));
            }
            return d.ToArray();
        }
    }
}
