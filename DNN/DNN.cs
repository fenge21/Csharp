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
        public DeepBeliefNetwork n { get; private set; }
        public string p { get; private set; }
        public double e { get; private set; }
        public DNN(string path) { p = path; }

        public void Train(double[][] i, double[][] o = null, int outputLength = 10, int hiddenLayer = -1)
        {
            if (n == null)
            {
                if (File.Exists(p)) n = DeepBeliefNetwork.Load(p);
                else
                {
                    outputLength = (o == null) ? outputLength : o[0].Length;
                    hiddenLayer = (hiddenLayer == -1) ? (int)Math.Log(i[0].Length, outputLength) : hiddenLayer;
                    List<int> layers = new List<int>();
                    for (int j = 0; j < hiddenLayer; j++) layers.Add(i[0].Length);
                    layers.Add(outputLength);
                    n = new DeepBeliefNetwork(new BernoulliFunction(), i[0].Length, layers.ToArray());
                    new GaussianWeights(n).Randomize();
                }
            }

            dynamic t;
            if (o == null)
            {
                t = new DeepBeliefNetworkLearning(n) { Algorithm = (h, v, j) => new ContrastiveDivergenceLearning(h, v), LayerIndex = n.Machines.Count - 1, };
                while (true) e = t.RunEpoch(t.GetLayerInput(i));
            }
            else
            {
                t = new DeepNeuralNetworkLearning(n) { Algorithm = (ann, j) => new ParallelResilientBackpropagationLearning(ann), LayerIndex = n.Machines.Count - 1, };
                while (true) e = t.RunEpoch(t.GetLayerInput(i), o);
            }
        }
        public double[][] Compute(double[][] i)
        {
            if (n == null) n = DeepBeliefNetwork.Load(p);

            List<double[]> d = new List<double[]>();
            for (int j = 0; j < i.Length; j++) d.Add(n.Compute(i[j]));
            return d.ToArray();
        }
        public void Save()
        {
            n.UpdateVisibleWeights();
            n.Save(p);
        }
    }
}
