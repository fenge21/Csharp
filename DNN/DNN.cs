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
        private string p { get; set; }
        private DeepBeliefNetwork n { get; set; }
        public DNN(string path) { p = path; }
        public void Train(double[][] i, double[][] o = null)
        {
            if (n == null)
            {
                if (File.Exists(p)) n = DeepBeliefNetwork.Load(p);
                else
                {
                    n = new DeepBeliefNetwork(new BernoulliFunction(), i[0].Length, i[0].Length, i[0].Length, o == null ? i[0].Length : o[0].Length);
                    new GaussianWeights(n).Randomize();
                }
            }

            double e = 100;
            dynamic t;
            if (o == null)
            {
                t = new DeepBeliefNetworkLearning(n) { Algorithm = (h, v, j) => new ContrastiveDivergenceLearning(h, v), LayerIndex = n.Machines.Count - 1, };
                new Task(() => { while (true) e = t.RunEpoch(t.GetLayerInput(i)); }).Start();
            }
            else
            {
                t = new DeepNeuralNetworkLearning(n) { Algorithm = (ann, j) => new ParallelResilientBackpropagationLearning(ann), LayerIndex = n.Machines.Count - 1, };
                new Task(() => { while (true) { e = t.RunEpoch(t.GetLayerInput(i), o); } }).Start();
            }

            while (true)
            {
                Console.WriteLine(e);
                n.UpdateVisibleWeights();
                n.Save(p);
                Thread.Sleep(1000 * 10);
            }
        }
        public double[][] Compute(double[][] i)
        {
            if (n == null)
            {
                if (File.Exists(p)) n = DeepBeliefNetwork.Load(p);
                else throw new Exception("No DNN Found!");
            }

            List<double[]> d = new List<double[]>();
            for (int j = 0; j < i.Length; j++)
            {
                d.Add(n.Compute(i[j]));
            }
            return d.ToArray();
        }
    }
}
