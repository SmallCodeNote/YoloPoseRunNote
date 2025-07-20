using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using System.Drawing;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace YoloPoseRun
{
    public class FrameDataSet
    {
        public List<PoseInfo> PoseInfos;
        public Tensor<float> tensor;
        public Bitmap bitmap;
        public int frameIndex;
        public Mat mat;
        public float[] output;
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
        public List<NamedOnnxValue> inputs;

        public string saveDirectoryPath;


        public FrameDataSet(int frameIndex)
        {
            this.frameIndex = frameIndex;
        }

        public FrameDataSet(Mat mat, int frameIndex, string masterDirectoryPath)
        {
            this.mat = mat;
            this.frameIndex = frameIndex;
            this.saveDirectoryPath = masterDirectoryPath;
        }

        public FrameDataSet(float[] output, Bitmap bitmap, int frameIndex, string masterDirectoryPath)
        {
            this.output = output;
            this.bitmap = bitmap;
            this.frameIndex = frameIndex;
            this.saveDirectoryPath = masterDirectoryPath;
        }

        public FrameDataSet(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results, Bitmap bitmap, int frameIndex, string masterDirectoryPath)
        {
            this.results = results;
            this.bitmap = bitmap;
            this.frameIndex = frameIndex;
            this.saveDirectoryPath = masterDirectoryPath;
        }

        public FrameDataSet(List<PoseInfo> PoseInfos, Bitmap bitmap, int frameIndex, string masterDirectoryPath)
        {
            this.PoseInfos = PoseInfos;
            this.bitmap = bitmap;
            this.frameIndex = frameIndex;
            this.saveDirectoryPath = masterDirectoryPath;
        }
        public FrameDataSet(List<PoseInfo> PoseInfos, int frameIndex, string masterDirectoryPath)
        {
            this.PoseInfos = PoseInfos;
            this.frameIndex = frameIndex;
            this.saveDirectoryPath = masterDirectoryPath;
        }
        public FrameDataSet(Bitmap bitmap, int frameIndex, string masterDirectoryPath)
        {
            this.bitmap = bitmap;
            this.frameIndex = frameIndex;
            this.saveDirectoryPath = masterDirectoryPath;
        }
        public FrameDataSet(Tensor<float> tensor, int frameIndex, string masterDirectoryPath)
        {
            this.tensor = tensor;
            this.frameIndex = frameIndex;
            this.saveDirectoryPath = masterDirectoryPath;
        }
        public FrameDataSet(Tensor<float> tensor, Bitmap bitmap, int frameIndex, string masterDirectoryPath)
        {
            this.tensor = tensor;
            this.bitmap = bitmap;
            this.frameIndex = frameIndex;
            this.saveDirectoryPath = masterDirectoryPath;
        }
        public FrameDataSet(List<NamedOnnxValue> inputs, Bitmap bitmap, int frameIndex, string masterDirectoryPath)
        {
            this.inputs = inputs;
            this.bitmap = bitmap;
            this.frameIndex = frameIndex;
            this.saveDirectoryPath = masterDirectoryPath;
        }

        public override string ToString()
        {
            return frameIndex.ToString();
        }
    }
}
