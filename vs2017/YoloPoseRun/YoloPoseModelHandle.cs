using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.IO;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using System.Diagnostics;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using OpenCvSharp;

using System.Runtime.CompilerServices;


namespace YoloPoseRun
{
    class YoloPoseModelHandle : IDisposable
    {
        public string SessionInputName = "";

        //public Tensor<float> ImageTensor;
        public List<PoseInfo> PoseInfos;
        private InferenceSession session;
        private int modelOutputStride = 8400;

        public PoseInfo_ConfidenceLevel ConfidenceSetting;
        public PoseInfo_OverLapThresholds OverLapSetting;

        public YoloPoseModelHandle(string modelfilePath, int deviceID = -1)
        {
            SetModel(modelfilePath, deviceID);
        }
        public YoloPoseModelHandle(string modelfilePath, PoseInfo_ConfidenceLevel confidenceSetting, PoseInfo_OverLapThresholds overLapSetting, int deviceID = -1)
        {
            SetModel(modelfilePath, deviceID);
            this.ConfidenceSetting = confidenceSetting;
            this.OverLapSetting = overLapSetting;
        }

        public void Dispose()
        {
            if (session != null) session.Dispose();
            GC.SuppressFinalize(this);
        }

        public bool SetModel(string modelfilePath, int deviceID = -1)
        {
            if (File.Exists(modelfilePath))
            {
                if (session != null) session.Dispose();
                SessionOptions sessionOptions = new SessionOptions();

                try
                {
                    if (deviceID >= 0) { sessionOptions.AppendExecutionProvider_DML(deviceID); }
                    else { sessionOptions.AppendExecutionProvider_CPU(); }

                    sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

                }
                catch
                {
                    Console.WriteLine("ExecutionProviderAppendError");
                }

                // Platform target = "x64"//
                session = new InferenceSession(modelfilePath, sessionOptions);
                SessionInputName = session.InputMetadata.Keys.First();
            }
            return false;
        }

        public float[] PredictOutput(Tensor<float> ImageTensor, float confidenceThreshold = -1.0f)
        {
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(SessionInputName, ImageTensor) };
            var results = session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();
            results.Dispose();

            return output;
        }
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> PredicteResults(Tensor<float> ImageTensor)
        {
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(SessionInputName, ImageTensor) };
            return session.Run(inputs);
        }

        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> PredicteResults(List<NamedOnnxValue> inputs)
        {
            return session.Run(inputs);
        }

        public List<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> PredicteResults(List<List<NamedOnnxValue>> inputsList)
        {
            List<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> results = new List<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>>();
            foreach (var inputs in inputsList)
            {
                results.Add(session.Run(inputs));
            }
            return results;
        }

        public List<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>> PredictBatch(List<List<NamedOnnxValue>> batchedInputs)
        {
            var results = new List<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>>();

            foreach (var inputs in batchedInputs)
            {
                using (var result = session.Run(inputs))
                {
                    results.Add(result);
                }
            }

            return results;
        }

        public List<NamedOnnxValue> GetInputs(Tensor<float> ImageTensor)
        {
            return new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(SessionInputName, ImageTensor) };
        }

        public unsafe Tensor<float> ConvertBitmapToTensor(Bitmap bitmap, int width = 640, int height = 640)
        {
            var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });
            int widthMax = Math.Min(bitmap.Width, width);
            int heightMax = Math.Min(bitmap.Height, height);

            BitmapData bitmapData = bitmap.LockBits(
                new Rectangle(0, 0, bitmap.Width, bitmap.Height),
                ImageLockMode.ReadOnly,
                PixelFormat.Format24bppRgb
            );

            try
            {
                byte* ptr = (byte*)bitmapData.Scan0;

                for (int y = 0; y < heightMax; y++)
                {
                    byte* row = ptr + y * bitmapData.Stride;

                    for (int x = 0; x < widthMax; x++)
                    {
                        int pixelIndex = x * 3; // 24bpp
                        tensor[0, 0, y, x] = row[pixelIndex + 2] / 255.0f; // R
                        tensor[0, 1, y, x] = row[pixelIndex + 1] / 255.0f; // G
                        tensor[0, 2, y, x] = row[pixelIndex] / 255.0f;     // B
                    }
                }
            }
            finally
            {
                bitmap.UnlockBits(bitmapData);
            }

            return tensor;
        }

        Tensor<float> ConvertMatToTensor(Mat mat, int width = 640, int height = 640)
        {
            Mat resizedMat = new Mat();
            Cv2.Resize(mat, resizedMat, new OpenCvSharp.Size(width, height));

            var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });

            if (resizedMat.Type() != MatType.CV_8UC3)
            {
                throw new ArgumentException("Input Mat must be of type CV_8UC3 (8-bit, 3 channels).");
            }

            for (int y = 0; y < height; y++)
            {
                var row = resizedMat.Row(y);

                for (int x = 0; x < width; x++)
                {
                    Vec3b pixel = row.At<Vec3b>(0, x);
                    tensor[0, 0, y, x] = pixel.Item2 / 255.0f; // R
                    tensor[0, 1, y, x] = pixel.Item1 / 255.0f; // G
                    tensor[0, 2, y, x] = pixel.Item0 / 255.0f; // B
                }
            }

            return tensor;
        }

        public override string ToString()
        {
            return SessionInputName;
        }

        public List<PoseInfo> PoseInfoRead(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
        {
            float[] outputArray = results.First().AsEnumerable<float>().ToArray();
            var result = PoseInfoRead(outputArray);
            return result;
        }
       
        public List<PoseInfo> PoseInfoRead(float[] outputArray)
        {

            var poseInfosBaseQueue = new ConcurrentQueue<List<PoseInfo>>();
            int modelOutputStrideSplit = 105;
            int paraMax = modelOutputStride / modelOutputStrideSplit;
            Parallel.For(0, paraMax, paraIndex =>
            {
                List<PoseInfo> poseInParallelForList = new List<PoseInfo>();
                int iMax = (paraIndex + 1) * modelOutputStrideSplit;
                for (int i = paraIndex * modelOutputStrideSplit; i < iMax; i++)
                {
                    PoseInfo pi = new PoseInfo(outputArray, i, ConfidenceSetting);
                    if (pi.Bbox.Confidence >= ConfidenceSetting.Bbox) { poseInParallelForList.Add(pi); }
                }
                poseInfosBaseQueue.Enqueue(poseInParallelForList);
            });

            List<PoseInfo> PoseInfosBaseList = new List<PoseInfo>();

            while (poseInfosBaseQueue.TryDequeue(out var poseInfosBase))
            {
                PoseInfosBaseList.AddRange(poseInfosBase);
            }

            List<PoseInfo> PoseInfos = new List<PoseInfo>();
            int PoseInfos_Count = PoseInfos.Count;
            bool storedData = false;

            int PoseInfosBaseList_Count = PoseInfosBaseList.Count;
            for (int a = 0; a < PoseInfosBaseList_Count; a++)
            {
                var poseA = PoseInfosBaseList[a];

                PoseInfos_Count = PoseInfos.Count;
                storedData = false;
                for (int b = 0; b < PoseInfos_Count; b++)
                {
                    var poseB = PoseInfos[b];
                    if (poseA.OverlapBbox(poseB) >= OverLapSetting.OverlapBBoxThreshold && OverLapSetting.OverlapBBoxThreshold >= 0
                        || poseA.OverlapTolso(poseB) >= OverLapSetting.OverlapTolsoThreshold && OverLapSetting.OverlapTolsoThreshold >= 0
                        || poseA.OverlapShoulder(poseB) >= OverLapSetting.OverlapShoulderThreshold && OverLapSetting.OverlapShoulderThreshold >= 0
                        )
                    {
                        poseB.Merge(poseA);
                        storedData = true;
                    }

                }

                if (!storedData) { PoseInfos.Add(poseA); }
            }

            this.PoseInfos = PoseInfos;
            return PoseInfos;
        }

        private void __debug_MessageWriteToConsole__(string message, [CallerLineNumber] int lineNumber = 0, [CallerFilePath] string filePath = null)
        {
            Console.WriteLine($"[-]\t{DateTime.Now:HH:mm:ss.fff}\t{0}\t{Path.GetFileName(filePath)}:{lineNumber}\t" + message);
        }

    }
}
