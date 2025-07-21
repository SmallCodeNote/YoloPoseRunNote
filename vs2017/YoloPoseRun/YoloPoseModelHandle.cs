using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.IO;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using OpenCvSharp;

namespace YoloPoseRun
{
    class YoloPoseModelHandle : IDisposable
    {
        public string SessionInputName = "";

        public Tensor<float> ImageTensor;
        public List<PoseInfo> PoseInfos;
        private InferenceSession session;
        private int modelOutputStride = 8400;

        public float ConfidenceThreshold = 0.16f;
        public float OverlapBBoxThreshold = 0.8f;
        public float OverlapTolsoThreshold = 0.8f;
        public float OverlapShoulderThreshold = 0.8f;

        public string ConfidenceParameterLinesString = "";


        public YoloPoseModelHandle(string modelfilePath, int deviceID = -1, string ConfidenceParameterLinesString = "")
        {
            SetModel(modelfilePath, deviceID);
            this.ConfidenceParameterLinesString = ConfidenceParameterLinesString;
        }

        public void Dispose()
        {
            if (session != null) session.Dispose();
            GC.SuppressFinalize(this);
        }

        public void InitializeParamFromTextLines(string LinesString)
        {
            InitializeParamFromTextLines(LinesString.Replace("\r\n", "\n").Trim('\n').Split('\n'));
        }
        public void InitializeParamFromTextLines(string[] Lines)
        {
            foreach (var line in Lines)
            {
                var parts = line.Split('\t');
                if (parts.Length != 2)
                    continue;

                string key = parts[0];
                if (!float.TryParse(parts[1], out float value))
                    continue;

                switch (key)
                {
                    case nameof(ConfidenceThreshold): ConfidenceThreshold = value; break;
                    case nameof(OverlapBBoxThreshold): OverlapBBoxThreshold = value; break;
                    case nameof(OverlapTolsoThreshold): OverlapTolsoThreshold = value; break;
                    case nameof(OverlapShoulderThreshold): OverlapShoulderThreshold = value; break;
                    default: break;
                }
            }
        }

        public string ParamToTextLinesString()
        {
            return string.Join("\r\n", ParamToTextLines());
        }
        public string[] ParamToTextLines()
        {
            return new string[]
            {
                $"{nameof(ConfidenceThreshold)}\t{ConfidenceThreshold}",
                $"{nameof(OverlapBBoxThreshold)}\t{OverlapBBoxThreshold}",
                $"{nameof(OverlapTolsoThreshold)}\t{OverlapTolsoThreshold}",
                $"{nameof(OverlapShoulderThreshold)}\t{OverlapShoulderThreshold}"
            };
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

        public List<PoseInfo> Predict(Bitmap bitmap, float confidenceThreshold = -1.0f)
        {
            if (confidenceThreshold < 0) { confidenceThreshold = ConfidenceThreshold; }
            ImageTensor = ConvertBitmapToTensor(bitmap);

            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(SessionInputName, ImageTensor) };
            var results = session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();
            results.Dispose();

            return PoseInfoRead(output, confidenceThreshold);
        }

        public List<PoseInfo> Predict(Tensor<float> ImageTensor, float confidenceThreshold = -1.0f)
        {
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(SessionInputName, ImageTensor) };
            var results = session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();
            results.Dispose();

            if (confidenceThreshold < 0) { confidenceThreshold = ConfidenceThreshold; }
            return PoseInfoRead(output, confidenceThreshold);
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

        public List<PoseInfo> PoseInfoRead(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results, float confidenceThreshold = -1.0f)
        {
            float[] outputArray = results.First().AsEnumerable<float>().ToArray();
            return PoseInfoRead(outputArray, confidenceThreshold);
        }

        public void SetOverlapThreshold(float OverlapBBoxThreshold, float OverlapTolsoThreshold, float OverlapShoulderThreshold)
        {
            this.OverlapBBoxThreshold = OverlapBBoxThreshold;
            this.OverlapTolsoThreshold = OverlapTolsoThreshold;
            this.OverlapShoulderThreshold = OverlapShoulderThreshold;
        }

        public List<PoseInfo> PoseInfoRead(float[] outputArray, float confidenceThreshold = -1.0f)
        {
            if (confidenceThreshold < 0) { confidenceThreshold = ConfidenceThreshold; }
            List<PoseInfo> PoseInfos = new List<PoseInfo>();
            for (int i = 0; i < modelOutputStride; i++)
            {
                PoseInfo pi = new PoseInfo(outputArray, i, ConfidenceParameterLinesString);
                if (pi.Bbox.Confidence >= confidenceThreshold)
                {
                    if (PoseInfos.Count > 0)
                    {
                        bool update = false;

                        for (int index = 0; index < PoseInfos.Count; index++)
                        {
                            var item = PoseInfos[index];

                            bool flag_OverlapBBox = item.OverlapBbox(pi) >= OverlapBBoxThreshold && OverlapBBoxThreshold >= 0;
                            bool flag_OverlapTolso = item.OverlapTolso(pi) >= OverlapTolsoThreshold && OverlapTolsoThreshold >= 0;
                            bool flag_OverlapShoulder = item.OverlapShoulder(pi) >= OverlapShoulderThreshold && OverlapShoulderThreshold >= 0;

                            if (flag_OverlapBBox || flag_OverlapTolso || flag_OverlapShoulder)
                            {
                                item.Merge(pi);
                                update = true;
                            }
                        }

                        if (!update) PoseInfos.Add(pi);
                    }
                    else
                    {
                        PoseInfos.Add(pi);
                    }
                }
            }

            this.PoseInfos = PoseInfos;
            return PoseInfos;
        }
    }
}
