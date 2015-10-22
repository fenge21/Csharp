using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepMagic
{
    public class IO
    {
        public static double[] LoadImg(string imgPath)
        {
            Image img = Bitmap.FromFile(imgPath);
            Bitmap b = new Bitmap(img);
            double[] features = new double[32 * 32];
            for (int i = 0; i < 32; i++)
                for (int j = 0; j < 32; j++)
                    features[i * 32 + j] = (b.GetPixel(j, i).R > 0) ? 0 : 1;

            return features;
        }
        public static double[] LoadText(string text)
        {
            Bitmap bitmap = new Bitmap(32, 32, PixelFormat.Format32bppRgb);
            string[] lines = text.Split(new String[] { "\n" }, StringSplitOptions.RemoveEmptyEntries);
            for (int i = 0; i < 32; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    if (lines[i][j] == '0')
                        bitmap.SetPixel(j, i, Color.White);
                    else bitmap.SetPixel(j, i, Color.Black);
                }
            }
            double[] features = new double[32 * 32];
            for (int i = 0; i < 32; i++)
                for (int j = 0; j < 32; j++)
                    features[i * 32 + j] = (bitmap.GetPixel(j, i).R > 0) ? 0 : 1;

            return features;
        }
        public static double[][][] LoadAllText(string textPath)
        {
            double[][][] dio = new double[2][][];
            string input = File.ReadAllText(textPath);
            StringReader reader = new StringReader(input);
            char[] buffer = new char[(32 + 1) * 32];
            int count = 0;
            List<double[]> ldi = new List<double[]>();
            List<double[]> ldo = new List<double[]>();
            while (true)
            {
                int read = reader.ReadBlock(buffer, 0, buffer.Length);
                string label = reader.ReadLine();

                if (read < buffer.Length || label == null) break;

                double[] d = LoadText(new String(buffer));
                double[] o = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                o[int.Parse(label)] = 1;

                ldi.Add(d);
                ldo.Add(o);
                count++;
            }

            dio[0] = ldi.ToArray();
            dio[1] = ldo.ToArray();
            return dio;
        }
        public static int double2int(double[] doubleOut)
        {
            int a = 0;
            for (int j = 0; j < 10; j++)
            {
                if (doubleOut[a] < doubleOut[j])
                {
                    a = j;
                }
            }
            return a;
        }
    }
}
