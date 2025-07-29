using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Imaging;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace YoloPoseRun
{
    public class YoloPoseRunClass : INotifyPropertyChanged, IDisposable
    {
        public int PredictTaskBatchSize = 256;
        public int PredictTaskBuffSize = 3;

        BlockingCollection<List<FrameDataSet>> frameBitmapQueue;
        BlockingCollection<List<FrameDataSet>> frameTensorQueue;
        BlockingCollection<List<FrameDataSet>> framePoseInfoQueue;

        BlockingCollection<List<FrameDataSet>> frameReportQueue;
        BlockingCollection<List<FrameDataSet>> frameVideoMatQueue;

        VideoCapture videoSource;
        YoloPoseModelHandle yoloPoseModelHandle;

        public PoseInfo_ConfidenceLevel ConfidenceLevelSetting;
        public PoseInfo_OverLapThresholds OverLapThresholdsSetting;
        public int DeviceID = -2;

        BlockingCollection<string> videoSourceFilePathQueue;

        public YoloPoseRunClass(BlockingCollection<string> videoSourceFilePathQueue, string modelFilePath, int deviceID, string ConfidenceParameterLinesString)
        {
            this.videoSourceFilePathQueue = videoSourceFilePathQueue;
            this.DeviceID = deviceID;

            if (File.Exists(modelFilePath))
            {
                ConfidenceLevelSetting = new PoseInfo_ConfidenceLevel(ConfidenceParameterLinesString);
                OverLapThresholdsSetting = new PoseInfo_OverLapThresholds(ConfidenceParameterLinesString);
                yoloPoseModelHandle = new YoloPoseModelHandle(modelFilePath, ConfidenceLevelSetting, OverLapThresholdsSetting, deviceID);
            }
        }

        public void Dispose()
        {
            if (yoloPoseModelHandle != null) yoloPoseModelHandle.Dispose();
            GC.SuppressFinalize(this);
        }

        private int _processRunCount;
        public int ProcessRunCount
        {
            get => _processRunCount;
            private set
            {
                if (_processRunCount != value)
                {
                    _processRunCount = value;
                    OnPropertyChanged(nameof(ProcessRunCount));
                }
            }
        }

        private string _videoSourceFilePath = "...";
        public string VideoSourceFilePath
        {
            get => _videoSourceFilePath;
            private set
            {
                if (_videoSourceFilePath != value)
                {
                    _videoSourceFilePath = value;
                    OnPropertyChanged(nameof(VideoSourceFilePath));
                }
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected void OnPropertyChanged(string name) =>
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name)
                );

        public Task Run(CancellationToken cancellationToken)
        {
            taskStartTime = DateTime.Now;

            if (timer1 != null) timer1.Dispose();
            timer1 = new System.Timers.Timer();
            timer1.Interval = 1000;
            timer1.Elapsed += timer1_Tick;
            timer1.Start();

            return Task.Run(() =>
            {
                try
                {
                    frameBitmapQueue = new BlockingCollection<List<FrameDataSet>>(PredictTaskBuffSize);
                    frameTensorQueue = new BlockingCollection<List<FrameDataSet>>(PredictTaskBuffSize);
                    framePoseInfoQueue = new BlockingCollection<List<FrameDataSet>>(PredictTaskBuffSize);
                    frameReportQueue = new BlockingCollection<List<FrameDataSet>>(PredictTaskBuffSize);
                    frameVideoMatQueue = new BlockingCollection<List<FrameDataSet>>(PredictTaskBuffSize);

                    Task task_frameVideoReader = Task.Run(() => dequeue_frameVideoReader(cancellationToken));
                    Task task_frameBitmap = Task.Run(() => dequeue_frameBitmap());
                    Task task_frameTensor = Task.Run(() => dequeue_frameTensor());
                    Task task_framePoseInfo = Task.Run(() => dequeue_framePoseInfo());
                    Task task_frameReport = Task.Run(() => dequeue_frameReport());
                    Task task_frameVideoMat = Task.Run(() => dequeue_frameVideoMat());

                    task_frameVideoReader.Wait();
                    task_frameBitmap.Wait();
                    task_frameTensor.Wait();
                    task_framePoseInfo.Wait();
                    task_frameReport.Wait();
                    task_frameVideoMat.Wait();

                    frameBitmapQueue.Dispose();
                    frameTensorQueue.Dispose();
                    framePoseInfoQueue.Dispose();
                    frameReportQueue.Dispose();
                    frameVideoMatQueue.Dispose();

                    timer1.Stop();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"ERROR:{System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");
                }

                Console.WriteLine($"///  FIN  ///");

            }, cancellationToken);

        }

        private void dequeue_frameVideoReader(CancellationToken cancellationToken)
        {
            try
            {
                ProcessRunCount = 0;

                Stopwatch sw = new Stopwatch();

                string videoSourceFilePath = "";
                int videoSource_FrameCount = 0;

                while (!videoSourceFilePathQueue.IsCompleted)
                {
                    __debug_CodeInfoWriteToConsole__(System.Reflection.MethodBase.GetCurrentMethod().Name + $" : {DeviceID} TryTake");

                    if (videoSourceFilePathQueue.TryTake(out videoSourceFilePath, 100))
                    {
                        VideoSourceFilePath = videoSourceFilePath;
                        ProcessRunCount++;

                        if (!File.Exists(videoSourceFilePath)) continue;

                        string saveDirectoryPath = Path.Combine(Path.GetDirectoryName(videoSourceFilePath), Path.GetFileNameWithoutExtension(videoSourceFilePath));
                        Console.WriteLine($"/// masterDirectoryPath: {saveDirectoryPath}");

                        if (Directory.Exists(saveDirectoryPath))
                        {
                            if (File.Exists(Path.Combine(saveDirectoryPath, "Pose.csv"))) continue;
                        }
                        else { Directory.CreateDirectory(saveDirectoryPath); }

                        Console.WriteLine($"/// capturePath: {videoSourceFilePath}");
                        string ext = Path.GetExtension(videoSourceFilePath);

                        if (ext != ".mp4") { continue; }

                        if (videoSource != null) { videoSource.Dispose(); }
                        videoSource = new VideoCapture(videoSourceFilePath);
                        videoSource_FrameCount = videoSource.FrameCount;
                        if (videoSource == null) continue;

                        int maxIndex = int.MinValue;
                        int frameIndex = 0;
                        videoSource.PosFrames = 0;

                        sw.Restart();
                        List<FrameDataSet> frameList = new List<FrameDataSet>(PredictTaskBatchSize);

                        using (Mat frame = new Mat())
                        {
                            while (videoSource.Read(frame) && !frame.Empty())
                            {
                                maxIndex = Math.Max(maxIndex, frameIndex);
                                frameList.Add(new FrameDataSet(BitmapConverter.ToBitmap(frame), frameIndex, saveDirectoryPath));

                                int frameList_Count = frameList.Count;
                                if (frameList_Count >= PredictTaskBatchSize && frameList_Count > 0)
                                {
                                    __debug_MessageWriteToConsole__($"TaskB\t{frameList[0].frameIndex}\t{frameList.Count}\t-task\t{sw.ElapsedMilliseconds}"); sw.Restart();
                                    frameBitmapQueue.Add(frameList); qB += frameList.Count;
                                    __debug_MessageWriteToConsole__($"Add_B\t{frameList[0].frameIndex}\t{frameList.Count}\t-wait\t{sw.ElapsedMilliseconds}"); sw.Restart();
                                    frameList = new List<FrameDataSet>(PredictTaskBatchSize);
                                }

                                if (cancellationToken.IsCancellationRequested) { break; }

                                frameIndex = videoSource.PosFrames;
                            }

                            if (frameList.Count > 0)
                            {
                                __debug_MessageWriteToConsole__($"TaskB\t{frameList[0].frameIndex}\t{frameList.Count}\t-task\t{sw.ElapsedMilliseconds}"); sw.Restart();
                                frameBitmapQueue.Add(frameList); qB += frameList.Count;
                                __debug_MessageWriteToConsole__($"Add_B\t{frameList[0].frameIndex}\t{frameList.Count}\t-wait\t{sw.ElapsedMilliseconds}"); sw.Restart();
                            }
                        }

                        Console.WriteLine($"Complete: {maxIndex} { System.Reflection.MethodBase.GetCurrentMethod().Name}");
                    }
                }

                frameBitmapQueue.CompleteAdding();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR:{System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");
            }
        }

        private void dequeue_frameBitmap()
        {
            try
            {
                Stopwatch sw = new Stopwatch();
                sw.Restart();

                while (!frameBitmapQueue.IsCompleted)
                {
                    if (frameBitmapQueue.TryTake(out List<FrameDataSet> frameList, 10))
                    {
                        int frameList_Count = frameList.Count;
                        qB -= frameList_Count;

                        __debug_MessageWriteToConsole__($"TakeB\t{frameList[0].frameIndex}\t{ frameList_Count}\t-wait\t{sw.ElapsedMilliseconds}"); sw.Restart();

                        Parallel.For(0, frameList_Count, index =>
                        {
                            var frameInfo = frameList[index];
                            Tensor<float> tensor = ConvertBitmapToTensor(frameInfo.bitmap);
                            frameInfo.inputs = yoloPoseModelHandle.GetInputs(tensor);
                        }
                        );

                        __debug_MessageWriteToConsole__($"TaskT\t{frameList[0].frameIndex}\t{ frameList_Count}\t-task\t{sw.ElapsedMilliseconds}"); sw.Restart();
                        frameTensorQueue.Add(frameList); qT += frameList_Count;
                        __debug_MessageWriteToConsole__($"Add_T\t{frameList[0].frameIndex}\t{ frameList_Count}\t-wait\t{sw.ElapsedMilliseconds}"); sw.Restart();
                    }
                }

                frameTensorQueue.CompleteAdding();
                Console.WriteLine($"Complete: {System.Reflection.MethodBase.GetCurrentMethod().Name}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR:{System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");
            }
        }

        public unsafe Tensor<float> ConvertBitmapToTensor(Bitmap bitmap, int width = 640, int height = 640)
        {
            var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });
            float[] tensorArray = tensor.Buffer.ToArray();
            int widthMax = Math.Min(bitmap.Width, width);
            int heightMax = Math.Min(bitmap.Height, height);
            const float scale = 1.0f / 255.0f;

            BitmapData bitmapData = bitmap.LockBits(
                new Rectangle(0, 0, bitmap.Width, bitmap.Height),
                ImageLockMode.ReadOnly,
                PixelFormat.Format24bppRgb
            );

            byte* ptr = (byte*)bitmapData.Scan0;
            int stride = bitmapData.Stride;

            Parallel.For(0, heightMax, y =>
            {
                byte* row = ptr + y * stride;
                int indexBase = y * width;
                int indexR = indexBase;
                int indexG = indexBase + height * width;
                int indexB = indexBase + height * 2 * width;

                for (int x = 0; x < widthMax; x++)
                {
                    int pixelIndex = x * 3;

                    tensorArray[indexR++] = row[pixelIndex + 2] * scale; // R
                    tensorArray[indexG++] = row[pixelIndex + 1] * scale; // G
                    tensorArray[indexB++] = row[pixelIndex] * scale;     // B
                }
            });

            bitmap.UnlockBits(bitmapData);

            var resultTensor = new DenseTensor<float>(tensorArray, new[] { 1, 3, height, width });
            return resultTensor;
        }

        private void dequeue_frameTensor()
        {
            try
            {
                Stopwatch sw = new Stopwatch();
                sw.Restart();

                int maxIndex = int.MinValue;

                while (!frameTensorQueue.IsCompleted)
                {
                    if (frameTensorQueue.TryTake(out List<FrameDataSet> frameList, 10))
                    {
                        int frameList_Count = frameList.Count;
                        qT -= frameList_Count;
                        __debug_MessageWriteToConsole__($"TakeT\t{frameList[0].frameIndex}\t{frameList_Count}\t-wait\t{sw.ElapsedMilliseconds}"); sw.Restart();

                        if (frameList_Count < 1) continue;


                        for (int i = 0; i < frameList_Count; i++)
                        {
                            frameList[i].results = yoloPoseModelHandle.PredicteResults(frameList[i].inputs);
                        }

                        __debug_MessageWriteToConsole__($"TaskP\t{frameList[0].frameIndex}\t{frameList_Count}\t-task\t{sw.ElapsedMilliseconds}"); sw.Restart();

                        framePoseInfoQueue.Add(frameList); qP += frameList_Count;

                        __debug_MessageWriteToConsole__($"Add_P\t{frameList[0].frameIndex}\t{frameList_Count}\t-wait\t{sw.ElapsedMilliseconds}"); sw.Restart();

                    }
                }

                framePoseInfoQueue.CompleteAdding();

                __debug_MessageWriteToConsole__($"Complete: {maxIndex} {System.Reflection.MethodBase.GetCurrentMethod().Name}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR:{System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");
            }
        }

        private void dequeue_framePoseInfo()
        {
            try
            {
                Stopwatch sw = new Stopwatch();
                sw.Restart();

                while (!framePoseInfoQueue.IsCompleted)
                {
                    if (framePoseInfoQueue.TryTake(out List<FrameDataSet> frameList, 10))
                    {
                        qP -= frameList.Count;
                        __debug_MessageWriteToConsole__($"TakeP\t{frameList[0].frameIndex}\t{frameList.Count}\t-wait\t{sw.ElapsedMilliseconds}"); sw.Restart();
                        drawPoseAndAddQueues(frameList);
                        __debug_MessageWriteToConsole__($"Add_R/V\t{frameList[0].frameIndex}\t{frameList.Count}\t-task\t{sw.ElapsedMilliseconds}"); sw.Restart();

                    }
                }

                frameReportQueue.CompleteAdding();
                frameVideoMatQueue.CompleteAdding();

                Console.WriteLine($"Complete: { System.Reflection.MethodBase.GetCurrentMethod().Name}");

            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: {System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");
            }
        }

        private void drawPoseAndAddQueues(List<FrameDataSet> frameList)
        {
            try
            {
                if (frameList.Count < 1) return;

                var reportArray = new FrameDataSet[frameList.Count];
                var videoArray = new FrameDataSet[frameList.Count];

                int frameList_Count = frameList.Count;
                Parallel.For(0, frameList_Count, i =>
                {
                    try
                    {
                        var frameInfo = frameList[i];
                        var poseInfos = yoloPoseModelHandle.PoseInfoRead(frameInfo.results);
                        frameInfo.results.Dispose();
                        frameInfo.PoseInfos = poseInfos;

                        reportArray[i] = new FrameDataSet(poseInfos, frameInfo.frameIndex, frameInfo.saveDirectoryPath);
                        if (frameInfo.bitmap != null)
                        {
                            drawPose(frameInfo.bitmap, poseInfos);
                            videoArray[i] = frameInfo;
                        }
                        else
                        {
                            Console.WriteLine($"ERROR: {System.Reflection.MethodBase.GetCurrentMethod().Name}  Null Bitmap");
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"ERROR:{System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");

                    }

                });

                List<FrameDataSet> reportList = new List<FrameDataSet>(reportArray);
                List<FrameDataSet> videoList = new List<FrameDataSet>(videoArray);

                frameReportQueue.Add(reportList); qR += reportList.Count;
                frameVideoMatQueue.Add(videoList); qV += videoList.Count;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR:{System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");

            }
        }


        private void dequeue_frameReport()
        {
            try
            {
                Stopwatch sw = new Stopwatch();
                sw.Restart();

                List<string> PoseValue = new List<string>();

                string saveDirectoryPath = "";
                string pathPose = Path.Combine(saveDirectoryPath, "Pose.csv");
                string linePose = "";
                string targetFilename = "";

                while (!frameReportQueue.IsCompleted)
                {
                    if (frameReportQueue != null && frameReportQueue.TryTake(out List<FrameDataSet> frameList, 10))
                    {
                        qR -= frameList.Count;

                        __debug_MessageWriteToConsole__($"TakeR\t{frameList[0].frameIndex}\t{frameList.Count}\t-wait\t{sw.ElapsedMilliseconds}"); sw.Restart();

                        foreach (var frameInfo in frameList)
                        {
                            if (saveDirectoryPath != frameInfo.saveDirectoryPath)
                            {

                                if (PoseValue.Count > 0) File.AppendAllLines(pathPose, PoseValue);
                                PoseValue.Clear();

                                saveDirectoryPath = frameInfo.saveDirectoryPath;
                                pathPose = Path.Combine(saveDirectoryPath, "Pose.csv");
                                targetFilename = Path.GetFileName(saveDirectoryPath);

                                if (!File.Exists(pathPose))
                                {
                                    PoseValue.Add("filename,frame," + PoseInfo.ToLineStringHeader() + ",Label");
                                }
                            }

                            if (frameInfo.frameIndex >= 0)
                            {
                                string posFrame = frameInfo.frameIndex.ToString();

                                foreach (var pose in frameInfo.PoseInfos)
                                {
                                    string poseString = pose.ToLineString();
                                    linePose = targetFilename + "," + posFrame + "," + poseString + ",-1";
                                    PoseValue.Add(linePose);
                                }
                            }
                            else
                            {
                                break;
                            }
                        }
                        __debug_MessageWriteToConsole__($"TaskR\t{frameList[0].frameIndex}\t{frameList.Count}\t-task\t{sw.ElapsedMilliseconds}"); sw.Restart();
                        frameList.Clear();
                    }
                    else
                    {
                        Thread.Sleep(1);
                    }
                }

                if (PoseValue.Count > 0)
                {
                    File.AppendAllLines(pathPose, PoseValue);
                    PoseValue.Clear();
                }

                Console.WriteLine("Complete:" + System.Reflection.MethodBase.GetCurrentMethod().Name);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR:{System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");

            }
        }

        enum AVHWDeviceType
        {
            AV_HWDEVICE_TYPE_NONE,
            AV_HWDEVICE_TYPE_VDPAU,
            AV_HWDEVICE_TYPE_CUDA,
            AV_HWDEVICE_TYPE_VAAPI,
            AV_HWDEVICE_TYPE_DXVA2,
            AV_HWDEVICE_TYPE_QSV,
            AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
            AV_HWDEVICE_TYPE_D3D11VA,
            AV_HWDEVICE_TYPE_DRM,
            AV_HWDEVICE_TYPE_OPENCL,
            AV_HWDEVICE_TYPE_MEDIACODEC,
            AV_HWDEVICE_TYPE_VULKAN,
            AV_HWDEVICE_TYPE_D3D12VA,
            AV_HWDEVICE_TYPE_AMF,
        };

        private void dequeue_frameVideoMat()
        {
            int frameIndex = 0;
            try
            {
                Stopwatch sw = new Stopwatch();
                sw.Restart();

                string saveVideoPath = "";
                string saveDirectoryPath = "";
                VideoWriter videoWriter = null;

                while (!frameVideoMatQueue.IsCompleted)
                {
                    if (frameVideoMatQueue.TryTake(out List<FrameDataSet> frameList, 10))
                    {
                        qV -= frameList.Count;

                        __debug_MessageWriteToConsole__($"TakeV\t{frameList[0].frameIndex}\t{frameList.Count}\t-wait\t{sw.ElapsedMilliseconds}"); sw.Restart();

                        foreach (var frameInfo in frameList)
                        {
                            if (videoWriter != null && saveDirectoryPath != frameInfo.saveDirectoryPath)
                            {
                                videoWriter.Release();
                                videoWriter.Dispose();
                                videoWriter = null;
                            }

                            if (videoWriter == null)
                            {
                                saveDirectoryPath = frameInfo.saveDirectoryPath;
                                saveVideoPath = Path.Combine(saveDirectoryPath, Path.GetFileNameWithoutExtension(saveDirectoryPath) + "_pose.mp4");

                                videoWriter = new VideoWriter(saveVideoPath, FourCC.FromString("mp4v"), 30, new OpenCvSharp.Size(640, 360));
                                videoWriter.Set(VideoWriterProperties.HwAcceleration, (double)AVHWDeviceType.AV_HWDEVICE_TYPE_QSV);
                            }

                            if (!videoWriter.IsOpened())
                            {
                                Console.WriteLine("video not opened");
                            }

                            frameIndex = frameInfo.frameIndex;
                            using (Mat mat = BitmapConverter.ToMat(frameInfo.bitmap))
                            {
                                Task.Run(() =>
                                {
                                    try
                                    {
                                        frameInfo.bitmap.Dispose();
                                    }
                                    catch (Exception ex)
                                    {
                                        Console.WriteLine($"ERROR: frameIndex={frameIndex} : {System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");
                                    }
                                });

                                Mat mat3C = mat.CvtColor(ColorConversionCodes.BGRA2BGR);
                                videoWriter.Write(mat3C);
                                mat3C.Dispose();
                            }
                        }

                        __debug_MessageWriteToConsole__($"TaskV\t{frameList[0].frameIndex}\t{frameList.Count}\t-task\t{sw.ElapsedMilliseconds}"); sw.Restart();
                        frameList.Clear();

                    }
                }

                if (videoWriter != null) videoWriter.Release();

                Console.WriteLine("Complete:" + System.Reflection.MethodBase.GetCurrentMethod().Name);

            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: frameIndex={frameIndex} : {System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");
            }
        }

        private void drawPose(Bitmap bitmap, List<PoseInfo> PoseInfos)
        {
            using (Graphics g = Graphics.FromImage(bitmap))
            {
                drawBBoxs(g, PoseInfos);
                drawBones(g, PoseInfos);
            }
        }

        public void drawBBoxs(Graphics g, List<PoseInfo> PoseInfos)
        {
            if (g != null)
            {
                foreach (var info in PoseInfos)
                {
                    g.DrawRectangle(Pens.Blue, info.Bbox.Rectangle);
                }
            }
        }

        public void drawBones(Graphics g, List<PoseInfo> PoseInfos)
        {
            if (g != null)
            {
                foreach (var info in PoseInfos)
                {
                    info.KeyPoints.drawBone(g);
                }
            }
        }

        DateTime taskStartTime;
        System.Timers.Timer timer1;
        int qB = 0, qT = 0, qP = 0, qR = 0, qV = 0;
        private void timer1_Tick(object sender, EventArgs e)
        {
            Console.WriteLine($"[{DeviceID}]\t{(DateTime.Now - taskStartTime).TotalMilliseconds:0}\tB{qB}\tT{qT}\tP{qP}\tR{qR}\tV{qV}");
        }

        public static void __debug_CodeInfoWriteToConsole__(string methodName, [CallerLineNumber] int lineNumber = 0, [CallerFilePath] string filePath = null)
        {
            Console.WriteLine($"[{DateTime.Now:HH:mm:dd.sss}] {Path.GetFileName(filePath)}:{lineNumber} - {methodName}");
        }

        private void __debug_MessageWriteToConsole__(string message, [CallerLineNumber] int lineNumber = 0, [CallerFilePath] string filePath = null)
        {
            Console.WriteLine($"[{DeviceID}]\t{DateTime.Now:HH:mm:ss.fff}\t{(DateTime.Now - taskStartTime).TotalMilliseconds:0}\t{Path.GetFileName(filePath)}:{lineNumber}\t" + message);
        }

    }
}
