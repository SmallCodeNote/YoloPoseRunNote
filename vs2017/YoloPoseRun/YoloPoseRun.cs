using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Imaging;
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

        BlockingCollection<FrameDataSet> frameBitmapQueue;
        BlockingCollection<FrameDataSet> frameTensorQueue;
        BlockingCollection<FrameDataSet> framePoseInfoQueue;

        BlockingCollection<FrameDataSet> frameReportQueue;
        BlockingCollection<FrameDataSet> frameVideoMatQueue;

        VideoCapture videoSource;
        YoloPoseModelHandle yoloPoseModelHandle;

        public string ConfidenceParameterLinesString;
        public int DeviceID = -2;

        BlockingCollection<string> videoSourceFilePathQueue;
        string saveDirectoryPath = "";

        public YoloPoseRunClass(BlockingCollection<string> videoSourceFilePathQueue, string modelFilePath, int deviceID, string ConfidenceParameterLinesString)
        {
            this.videoSourceFilePathQueue = videoSourceFilePathQueue;
            this.DeviceID = deviceID;

            if (File.Exists(modelFilePath))
            {
                yoloPoseModelHandle = new YoloPoseModelHandle(modelFilePath, deviceID, ConfidenceParameterLinesString);
            }
            this.ConfidenceParameterLinesString = ConfidenceParameterLinesString;
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


        /*
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
                    string ext = "";
                    ProcessRunCount = 0;
                    string videoSourceFilePath = "";
                    while (!videoSourceFilePathQueue.IsCompleted)
                    {
                        getDebugInfo(System.Reflection.MethodBase.GetCurrentMethod().Name + $" : {DeviceID} TryTake");
                        if (videoSourceFilePathQueue.TryTake(out videoSourceFilePath, 100))
                        {

                            VideoSourceFilePath = videoSourceFilePath;
                            ProcessRunCount++;

                            if (!File.Exists(videoSourceFilePath)) continue;

                            Console.WriteLine($"/// capturePath: {videoSourceFilePath}");
                            ext = Path.GetExtension(videoSourceFilePath);

                            if (ext == ".mp4")
                            {
                                if (videoSource != null) { videoSource.Dispose(); targetFilename = ""; }
                                videoSource = new VideoCapture(videoSourceFilePath);
                                targetFilename = Path.GetFileNameWithoutExtension(videoSourceFilePath);
                            }
                            else
                            {
                                continue;
                            }

                            if (videoSource == null) return;

                            saveDirectoryPath = Path.Combine(Path.GetDirectoryName(videoSourceFilePath), Path.GetFileNameWithoutExtension(videoSourceFilePath));
                            Console.WriteLine($"/// masterDirectoryPath: {saveDirectoryPath}");

                            if (!Directory.Exists(saveDirectoryPath)) { Directory.CreateDirectory(saveDirectoryPath); } else { continue; }

                            frameBitmapQueue = new BlockingCollection<FrameDataSet>(PredictTaskBatchSize);
                            frameTensorQueue = new BlockingCollection<FrameDataSet>(PredictTaskBatchSize);
                            framePoseInfoQueue = new BlockingCollection<FrameDataSet>(PredictTaskBatchSize * 2);
                            frameReportQueue = new BlockingCollection<FrameDataSet>(PredictTaskBatchSize * 2);
                            frameVideoMatQueue = new BlockingCollection<FrameDataSet>(PredictTaskBatchSize * 2);

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

                        }
                        getDebugInfo(System.Reflection.MethodBase.GetCurrentMethod().Name + $" : {DeviceID} LoopEnd");
                    }





                    getDebugInfo(System.Reflection.MethodBase.GetCurrentMethod().Name + $" : {DeviceID} Completed");

                    if (videoSource != null) { videoSource.Dispose(); targetFilename = ""; }

                    timer1.Stop();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"ERROR:{System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");
                }

                Console.WriteLine($"///  FIN  ///");

            }, cancellationToken);

        }*/


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
                    frameBitmapQueue = new BlockingCollection<FrameDataSet>(PredictTaskBatchSize);
                    frameTensorQueue = new BlockingCollection<FrameDataSet>(PredictTaskBatchSize);
                    framePoseInfoQueue = new BlockingCollection<FrameDataSet>(PredictTaskBatchSize * 2);
                    frameReportQueue = new BlockingCollection<FrameDataSet>(PredictTaskBatchSize * 2);
                    frameVideoMatQueue = new BlockingCollection<FrameDataSet>(PredictTaskBatchSize * 2);

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

        DateTime taskStartTime;
        System.Timers.Timer timer1;
        private void timer1_Tick(object sender, EventArgs e)
        {
            if (frameBitmapQueue != null && frameTensorQueue != null && framePoseInfoQueue != null && frameReportQueue != null)
            {
                Console.WriteLine($"[{DeviceID}]\t{(DateTime.Now - taskStartTime).TotalSeconds:0}\tB{frameBitmapQueue.Count}\tT{frameTensorQueue.Count}\tP{framePoseInfoQueue.Count}\tR{frameReportQueue.Count}");
            }
        }


        string ProgressReport = "";

        private void dequeue_frameVideoReader(CancellationToken cancellationToken)
        {
            try
            {
                string ext = "";
                ProcessRunCount = 0;
                string videoSourceFilePath = "";
                while (!videoSourceFilePathQueue.IsCompleted)
                {
                    getDebugInfo(System.Reflection.MethodBase.GetCurrentMethod().Name + $" : {DeviceID} TryTake");
                    if (videoSourceFilePathQueue.TryTake(out videoSourceFilePath, 100))
                    {

                        VideoSourceFilePath = videoSourceFilePath;
                        ProcessRunCount++;

                        if (!File.Exists(videoSourceFilePath)) continue;

                        saveDirectoryPath = Path.Combine(Path.GetDirectoryName(videoSourceFilePath), Path.GetFileNameWithoutExtension(videoSourceFilePath));
                        Console.WriteLine($"/// masterDirectoryPath: {saveDirectoryPath}");

                        if (Directory.Exists(saveDirectoryPath))
                        {
                            if (File.Exists(Path.Combine(saveDirectoryPath, "Pose.csv"))) continue;
                        }
                        else { Directory.CreateDirectory(saveDirectoryPath); }

                        Console.WriteLine($"/// capturePath: {videoSourceFilePath}");
                        ext = Path.GetExtension(videoSourceFilePath);

                        if (ext != ".mp4") { continue; }
                        else
                        {
                            if (videoSource != null) { videoSource.Dispose(); targetFilename = ""; }
                            videoSource = new VideoCapture(videoSourceFilePath);
                            targetFilename = Path.GetFileNameWithoutExtension(videoSourceFilePath);
                        }

                        if (videoSource == null) continue;


                        int maxIndex = int.MinValue;
                        List<FrameDataSet> frameList = new List<FrameDataSet>(PredictTaskBatchSize);

                        int frameIndex = 0;
                        videoSource.PosFrames = 0;

                        using (Mat frame = new Mat())
                        {
                            while (videoSource.Read(frame) && !frame.Empty())
                            {
                                maxIndex = Math.Max(maxIndex, frameIndex);

                                frameList.Add(new FrameDataSet(BitmapConverter.ToBitmap(frame), frameIndex, saveDirectoryPath));

                                if (frameList.Count >= PredictTaskBatchSize && frameList.Count > 0)
                                {
                                    ProgressReport = $"{frameIndex} / {videoSource.FrameCount}";
                                    Console.Write($"  [{DeviceID}][{DateTime.Now:HH:mm:ss}] Add+B start {frameList[0].frameIndex} + {frameList.Count}");
                                    foreach (var item in frameList) { frameBitmapQueue.Add(item); }
                                    Console.WriteLine($"  Add+B comp {frameList[0].frameIndex} - {frameList[frameList.Count - 1].frameIndex}");
                                    frameList.Clear();

                                    Task.WaitAll(Task.Delay(3));
                                }

                                if (cancellationToken.IsCancellationRequested) { break; }

                                frameIndex = videoSource.PosFrames;
                            }

                            if (frameList.Count > 0)
                            {
                                Console.Write($"  [{DeviceID}][{DateTime.Now:HH:mm:ss}] Add+B start {frameList[0].frameIndex} + {frameList.Count}");
                                foreach (var item in frameList) { frameBitmapQueue.Add(item); }
                                Console.WriteLine($"  Add+B comp {frameList[0].frameIndex} - {frameList[frameList.Count - 1].frameIndex}");
                                frameList.Clear();
                            }

                            //frameBitmapQueue.CompleteAdding();
                        }

                        ProgressReport = $"{frameIndex} / {videoSource.FrameCount}";
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
                int workersCount = Environment.ProcessorCount - 1;
                workersCount = workersCount < 1 ? 1 : workersCount;
                int frameListMasterSize = PredictTaskBatchSize / workersCount;
                if (frameListMasterSize < 1) frameListMasterSize = 1;

                var frameBitmapQueueTemp = new BlockingCollection<List<FrameDataSet>>(workersCount);
                var frameNextIndexQueue = new ConcurrentQueue<int>();

                Task[] workers = new Task[workersCount];

                for (int i = 0; i < workers.Length; i++)
                {
                    workers[i] = Task.Run(() =>
                    {
                        try
                        {
                            List<FrameDataSet> frameList = new List<FrameDataSet>(frameListMasterSize);

                            foreach (var frameInfos in frameBitmapQueueTemp.GetConsumingEnumerable())
                            {
                                foreach (var frameInfo in frameInfos)
                                {
                                    Tensor<float> tensor = ConvertBitmapToTensor(frameInfo.bitmap);
                                    List<NamedOnnxValue> inputs = yoloPoseModelHandle.GetInputs(tensor);
                                    frameList.Add(new FrameDataSet(inputs, frameInfo.bitmap, frameInfo.frameIndex, frameInfo.saveDirectoryPath));
                                }

                                if (frameList.Count > 0)
                                {
                                    while (!frameNextIndexQueue.TryPeek(out int nextIndex) || frameList[0].frameIndex != nextIndex) { Task.WaitAll(Task.Delay(10)); }

                                    Console.Write($"   [{DeviceID}][{DateTime.Now:HH:mm:ss}] Add+T start {frameList[0].frameIndex} + {frameList.Count}");
                                    foreach (var item in frameList) { frameTensorQueue.Add(item); }
                                    Console.WriteLine($"  Add+T comp {frameList[0].frameIndex} - {frameList[frameList.Count - 1].frameIndex }");
                                    frameList.Clear();
                                    frameNextIndexQueue.TryDequeue(out int result);
                                }

                                frameInfos.Clear();
                            }
                        }
                        catch
                        {
                            Console.WriteLine($"ERROR:{System.Reflection.MethodBase.GetCurrentMethod().Name}");
                        }
                    });
                }

                List<FrameDataSet> frameListMaster = new List<FrameDataSet>(frameListMasterSize);

                while (!frameBitmapQueue.IsCompleted)
                {
                    if (frameBitmapQueue.TryTake(out FrameDataSet frameInfo, 10))
                    {
                        frameListMaster.Add(frameInfo);

                        if (frameListMaster.Count >= frameListMasterSize && frameListMaster.Count > 0)
                        {
                            frameNextIndexQueue.Enqueue(frameListMaster[0].frameIndex);
                            frameBitmapQueueTemp.Add(frameListMaster);
                            frameListMaster = new List<FrameDataSet>(frameListMasterSize);
                        }
                    }
                }

                if (frameListMaster.Count > 0)
                {
                    frameNextIndexQueue.Enqueue(frameListMaster[0].frameIndex);
                    frameBitmapQueueTemp.Add(frameListMaster);
                }

                frameBitmapQueueTemp.CompleteAdding();

                Task.WaitAll(workers);
                frameBitmapQueueTemp.Dispose();
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
                int maxIndex = int.MinValue;
                List<FrameDataSet> frameList = new List<FrameDataSet>(PredictTaskBatchSize);

                while (!frameTensorQueue.IsCompleted)
                {
                    if (frameTensorQueue.TryTake(out FrameDataSet frameInfo, 10))
                    {
                        frameList.Add(frameInfo);

                        if (frameList.Count >= PredictTaskBatchSize)
                        {
                            Console.WriteLine($"  StartPredictTask {frameList[0].frameIndex} - {frameList[frameList.Count - 1].frameIndex}");
                            PredictBatch(frameList);
                            frameList.Clear();
                        }
                    }
                }

                if (frameList.Count > 0)
                {
                    Console.WriteLine($"  LastPredictTask {frameList[0].frameIndex} - {frameList[frameList.Count - 1].frameIndex}");
                    PredictBatch(frameList);
                    frameList.Clear();
                }

                framePoseInfoQueue.CompleteAdding();
                Console.WriteLine($"Complete: {maxIndex} {System.Reflection.MethodBase.GetCurrentMethod().Name}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR:{System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");
            }
        }

        public void PredictBatch(IReadOnlyList<FrameDataSet> frameList)
        {
            try
            {
                if (frameList.Count < 1) return;

                Console.WriteLine($".....Start: {frameList[0].frameIndex} + {frameList.Count} " + System.Reflection.MethodBase.GetCurrentMethod().Name);

                int arrayMax = frameList.Count;
                for (int i = 0; i < arrayMax; i++)
                {
                    frameList[i].results = yoloPoseModelHandle.PredicteResults(frameList[i].inputs);
                }

                Console.WriteLine($".....Complete: {frameList[0].frameIndex} - {frameList[frameList.Count - 1].frameIndex} {System.Reflection.MethodBase.GetCurrentMethod().Name}");

                framePoseInfoQueueEnqueue(frameList);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR:{System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");
            }
        }

        public void framePoseInfoQueueEnqueue(IReadOnlyList<FrameDataSet> frameList)
        {
            try
            {
                if (frameList.Count < 1) return;
                Console.Write($"    [{DeviceID}][{DateTime.Now:HH:mm:ss}] Add+P start {frameList[0].frameIndex} + {frameList.Count}");
                foreach (var frameInfo in frameList) { framePoseInfoQueue.Add(frameInfo); }
                Console.WriteLine($"  Add+P comp {frameList[0].frameIndex} - {frameList[frameList.Count - 1].frameIndex}");
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
                List<FrameDataSet> frameList = new List<FrameDataSet>(PredictTaskBatchSize);

                while (!framePoseInfoQueue.IsCompleted)
                {
                    if (framePoseInfoQueue.TryTake(out FrameDataSet frameInfo, 10))
                    {
                        frameList.Add(frameInfo);

                        if (frameList.Count >= PredictTaskBatchSize)
                        {
                            dequeue_framePoseInfo_addQueue(frameList);
                            frameList.Clear();
                        }
                    }
                }

                if (frameList.Count > 0)
                {
                    dequeue_framePoseInfo_addQueue(frameList);
                    frameList.Clear();
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

        private void dequeue_framePoseInfo_addQueue(IReadOnlyList<FrameDataSet> frameList)
        {
            if (frameList.Count < 1) return;

            Console.Write($"      [{DeviceID}][{DateTime.Now:HH:mm:ss}] Add+R start {frameList[0].frameIndex} + {frameList.Count}");

            var reportDict = new ConcurrentDictionary<int, FrameDataSet>();
            var videoDict = new ConcurrentDictionary<int, FrameDataSet>();

            Parallel.For(0, frameList.Count, index =>
            {
                var frameInfo = frameList[index];
                var poseInfos = yoloPoseModelHandle.PoseInfoRead(frameInfo.results);
                frameInfo.results.Dispose();

                reportDict[index] = new FrameDataSet(poseInfos, frameInfo.frameIndex, frameInfo.saveDirectoryPath);

                if (frameInfo.bitmap != null)
                {
                    drawPose(frameInfo.bitmap, poseInfos);
                    using (Mat mat = BitmapConverter.ToMat(frameInfo.bitmap))
                    {
                        Mat mat3C = mat.CvtColor(ColorConversionCodes.BGRA2BGR);
                        videoDict[index] = new FrameDataSet(mat3C, frameInfo.frameIndex, frameInfo.saveDirectoryPath);
                    }
                }
                else
                {
                    Console.WriteLine($"ERROR: {System.Reflection.MethodBase.GetCurrentMethod().Name}  Null Bitmap");
                }
            });


            for (int i = 0; i < frameList.Count; i++)
            {
                if (reportDict.TryGetValue(i, out var reportItem))
                {
                    frameReportQueue.Add(reportItem);
                }

                if (videoDict.TryGetValue(i, out var videoItem))
                {
                    frameVideoMatQueue.Add(videoItem);
                }
            }

            Console.WriteLine($"  Add+R comp {frameList[0].frameIndex} - {frameList[frameList.Count - 1].frameIndex}");
        }



        string targetFilename = "";

        private void dequeue_frameReport()
        {
            try
            {
                List<string> HeadKeyPoint = new List<string>();
                List<string> NoseKeyPoint = new List<string>();
                List<string> EyeKeyPoint = new List<string>();
                List<string> EarKeyPoint = new List<string>();
                List<string> ShoulderKeyPoint = new List<string>();
                List<string> ElbowKeyPoint = new List<string>();
                List<string> WristKeyPoint = new List<string>();
                List<string> HipKeyPoint = new List<string>();
                List<string> KneeKeyPoint = new List<string>();
                List<string> AnkleKeyPoint = new List<string>();

                List<string> PoseValue = new List<string>();

                string saveDirectoryPath = "";
                string pathPose = Path.Combine(saveDirectoryPath, "Pose.csv");
                string linePose = "";

                bool isFirst = true;

                while (!frameReportQueue.IsCompleted)
                {
                    if (frameReportQueue != null && frameReportQueue.TryTake(out FrameDataSet frameInfo, 10))
                    {
                        if (isFirst)
                        {
                            Console.WriteLine("Start:" + System.Reflection.MethodBase.GetCurrentMethod().Name);
                            isFirst = false;
                        }

                        if (saveDirectoryPath != frameInfo.saveDirectoryPath)
                        {
                            if (PoseValue.Count > 0) File.AppendAllLines(pathPose, PoseValue);
                            PoseValue.Clear();

                            saveDirectoryPath = frameInfo.saveDirectoryPath;
                            pathPose = Path.Combine(saveDirectoryPath, "Pose.csv");

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
                string saveVideoPath = "";
                string saveDirectoryPath = "";
                VideoWriter videoWriter = null;

                while (!frameVideoMatQueue.IsCompleted)
                {
                    if (frameVideoMatQueue.TryTake(out FrameDataSet frameInfo, 10))
                    {
                        if(videoWriter != null && saveDirectoryPath != frameInfo.saveDirectoryPath)
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
                        videoWriter.Write(frameInfo.mat);
                        frameInfo.mat.Dispose();
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

        public static void getDebugInfo(string methodName, [CallerLineNumber] int lineNumber = 0, [CallerFilePath] string filePath = null)
        {
            Console.WriteLine($"[{DateTime.Now:HH:mm:dd.sss}] {Path.GetFileName(filePath)}:{lineNumber} - {methodName}");
        }

    }
}
