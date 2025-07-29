using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Win32;

using OpenCvSharp;
using OpenCvSharp.Extensions;

using ControlValuesToStringClass;

namespace YoloPoseRun
{
    public partial class MainWindow : System.Windows.Window
    {
        YoloPoseRunManager progressManager;

        Task task_GPU0_0 = null;
        Task task_GPU1_0 = null;
        Task task_CPU = null;

        YoloPoseRunClass yoloPose_GPU0_0;
        YoloPoseRunClass yoloPose_GPU1_0;
        YoloPoseRunClass yoloPose_CPU;

        public BlockingCollection<string> srcFileQueue = null;
        public ConcurrentQueue<string> srcFileList = new ConcurrentQueue<string>();

        private List<Task> collectFilePathTaskList = new List<Task>();
        private CancellationTokenSource tokenSource = new CancellationTokenSource();

        public MainWindow()
        {
            InitializeComponent();

            progressManager = new YoloPoseRunManager(srcFileList);
            label_progress.DataContext = progressManager;

        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            string inifilepath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "_Param.ini");
            if (File.Exists(inifilepath)) ControlValuesToString.PutValue(this, File.ReadAllText(inifilepath));
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            string inifilepath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "_Param.ini");
            File.WriteAllText(inifilepath, ControlValuesToString.GetString(this));
        }

        private void button_modelFilePathOpen_Click(object sender, RoutedEventArgs e)
        {
            string path = textBox_modelFilePath.Text;

            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "ONNX|*.onnx";
            ofd.FilterIndex = 0;
            ofd.FileName = path;
            if (ofd.ShowDialog() != true) return;

            textBox_modelFilePath.Text = ofd.FileName;

        }

        private void Button_modelFilePathClear_Click(object sender, RoutedEventArgs e)
        {
            textBox_modelFilePath.Text = "";
        }

        private void Button_directoryListAdd_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new Ookii.Dialogs.Wpf.VistaFolderBrowserDialog();
            dialog.Description = "Select Directory";
            dialog.Multiselect = true;
            bool result = dialog.ShowDialog() ?? false;

            List<string> directoryList = new List<string>(textBox_directoryList.Text.Replace("\r\n", "\n").Split('\n'));

            if (result)
            {
                directoryList.AddRange(dialog.SelectedPaths);
                textBox_directoryList.Text = string.Join("\r\n", directoryList);
            }
        }

        private void Button_directoryListClear_Click(object sender, RoutedEventArgs e)
        {
            textBox_directoryList.Text = "";
        }

        private void button_RunModel_Click(object sender, RoutedEventArgs e)
        {
            if ((string)button_RunModel.Content == "▶")
            {
                if (srcFileQueue != null) srcFileQueue.Dispose();
                srcFileQueue = new BlockingCollection<string>();

                progressManager.Clear();

                initializeTask(textBox_batchSize_GPU0, out yoloPose_GPU0_0, out task_GPU0_0, 0);
                initializeTask(textBox_batchSize_GPU1, out yoloPose_GPU1_0, out task_GPU1_0, 1);
                initializeTask(textBox_batchSize_CPU, out yoloPose_CPU, out task_CPU, -1);

                if (yoloPose_GPU0_0 != null) progressManager.Add(yoloPose_GPU0_0, "GPU0");
                if (yoloPose_GPU1_0 != null) progressManager.Add(yoloPose_GPU1_0, "GPU1");
                if (yoloPose_CPU != null) progressManager.Add(yoloPose_CPU, "CPU");

                string[] directoryPathList = textBox_directoryList.Text.Replace("\r\n", "\n").Split('\n');

                foreach (var directoryPath in directoryPathList)
                {
                    if (directoryPath.Length <= 2 || !Directory.Exists(directoryPath)) continue;
                    collectFilePathTaskList.Add(collectFilePathAsync(directoryPath, tokenSource.Token, "*.mp4"));
                }

                Task.Run(() => taskCreatePathListComplete());
                button_RunModel.Content = "⏸";
            }
            else if ((string)button_RunModel.Content == "⏸")
            {
                tokenSource.Cancel();
                Task.Run(() =>
                {
                    if (task_GPU0_0 != null) task_GPU0_0.Wait();
                    if (task_GPU1_0 != null) task_GPU1_0.Wait();
                    if (task_CPU != null) task_CPU.Wait();

                    taskComplete();

                });
            }
        }

        private string getDirectoryNameAndFilename(string path)
        {
            return System.IO.Path.Combine(System.IO.Path.GetFileName(System.IO.Path.GetDirectoryName(path)), System.IO.Path.GetFileName(path));
        }

        private void initializeTask(TextBox textBox, out YoloPoseRunClass yoloPose, out Task task, int deviceID)
        {
            if (int.TryParse(textBox.Text, out int batchSize) && batchSize > 0)
            {
                yoloPose = new YoloPoseRunClass(srcFileQueue, textBox_modelFilePath.Text, deviceID, textBox_initializeLinesString.Text);
                yoloPose.PredictTaskBatchSize = batchSize;
                task = yoloPose.Run(tokenSource.Token);
            }
            else
            {
                yoloPose = null;
                task = null;
            }
        }

        private void taskCreatePathListComplete()
        {
            Task.WaitAll(collectFilePathTaskList.ToArray());
            srcFileQueue.CompleteAdding();

            Task.Run(() =>
            {
                if (task_GPU0_0 != null) task_GPU0_0.Wait();
                if (task_GPU1_0 != null) task_GPU1_0.Wait();
                if (task_CPU != null) task_CPU.Wait();

                taskComplete();
            });
        }

        private void taskComplete()
        {
            Dispatcher.Invoke((Action)(() =>
            {
                button_RunModel.Content = "▶";
                progressManager.IsComplete = true;
            }));
        }

        public async Task collectFilePathAsync(string targetDirectoryPath, CancellationToken cancellationToken, string searchPattern = "*.*")
        {
            await Task.Run(() =>
            {
                try
                {
                    IEnumerable<string> allFilePaths = Directory.EnumerateFiles(targetDirectoryPath, searchPattern, SearchOption.TopDirectoryOnly);
                    List<string> filePathsBuff = new List<string>();

                    foreach (string filePath in allFilePaths)
                    {
                        filePathsBuff.Add(filePath);
                        if (cancellationToken.IsCancellationRequested) break;

                        if (filePathsBuff.Count > 64)
                        {
                            foreach (var item in filePathsBuff)
                            {
                                srcFileQueue.Add(item);
                                srcFileList.Enqueue(item);
                            }

                            filePathsBuff.Clear();
                        }
                    }

                    foreach (var item in filePathsBuff)
                    {
                        srcFileQueue.Add(item);
                        srcFileList.Enqueue(item);
                    }

                    filePathsBuff.Clear();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"ERROR:{System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");

                }

            }, cancellationToken);
        }

        private void Button_getDefaultConfidence_Click(object sender, RoutedEventArgs e)
        {
            YoloPoseModelHandle y = new YoloPoseModelHandle("");
            PoseKeyPoints k = new PoseKeyPoints(null, -1, "");
            PoseInfo_ConfidenceLevel c = new PoseInfo_ConfidenceLevel();
            PoseInfo_OverLapThresholds o = new PoseInfo_OverLapThresholds();

            textBox_initializeLinesString.Text = o.ParamToTextLinesString() + "\r\n" + c.ParamToTextLinesString();
            y.Dispose();
        }

        private void Button_saveConfigAs_Click(object sender, RoutedEventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.Filter = "TEXT|*.txt";
            sfd.FileName = $"Param_{DateTime.Now:yyyyMMdd_HHmm}.ini";

            if (sfd.ShowDialog() != true) return;

            string inifilepath = sfd.FileName;
            File.WriteAllText(inifilepath, ControlValuesToString.GetString(this));
        }

        private void Button_defaultUpdate_Click(object sender, RoutedEventArgs e)
        {
            string inifilepath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "_Param.ini");
            File.WriteAllText(inifilepath, ControlValuesToString.GetString(this));
        }


        VideoCapture capture;
        string targetFilename = "";
        int frameIndex = 0;

        private void OpenMovieFile(string filePath)
        {
            if (!Application.Current.Dispatcher.CheckAccess())
            {
                Application.Current.Dispatcher.Invoke(() => OpenMovieFile(filePath));
                return;
            }

            if (capture != null)
            {
                capture.Dispose();
                targetFilename = "";
            }

            string ext = System.IO.Path.GetExtension(filePath);
            if (string.Equals(ext, ".mp4", StringComparison.OrdinalIgnoreCase))
            {
                targetFilename = System.IO.Path.GetFileNameWithoutExtension(filePath);
                capture = new VideoCapture(filePath);

                slider_frameIndex.Maximum = capture.FrameCount;
                slider_frameIndex.Value = 0;
                ShowFrame(0);
            }
        }

        private void slider_frameIndex_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            frameIndex = (int)e.NewValue;
            ShowFrame(frameIndex);
        }

        private void ShowFrame(int frameIndex)
        {
            if (!Application.Current.Dispatcher.CheckAccess())
            {
                Application.Current.Dispatcher.Invoke(() => ShowFrame(frameIndex));
                return;
            }

            if (capture == null || !capture.IsOpened()) return;

            capture.Set(VideoCaptureProperties.PosFrames, frameIndex);

            using (var frame = new Mat())
            {

                if (capture.Read(frame) && !frame.Empty())
                {
                    string ConfidenceParameterLinesString = textBox_initializeLinesString.Text;
                    string modelFilePath = textBox_modelFilePath.Text;
                    int deviceID = 0;

                    var ConfidenceLevelSetting = new PoseInfo_ConfidenceLevel(ConfidenceParameterLinesString);
                    var OverLapThresholdsSetting = new PoseInfo_OverLapThresholds(ConfidenceParameterLinesString);
                    var yoloPoseModelHandle = new YoloPoseModelHandle(modelFilePath, ConfidenceLevelSetting, OverLapThresholdsSetting, deviceID);

                    var bitmap = BitmapConverter.ToBitmap(frame);
                    Tensor<float> tensor = YoloPoseRunClass.ConvertBitmapToTensor(bitmap);
                    var inputs = yoloPoseModelHandle.GetInputs(tensor);
                    var results = yoloPoseModelHandle.PredicteResults(inputs);
                    var poseInfos = yoloPoseModelHandle.PoseInfoRead(results);
                    YoloPoseRunClass.drawPose(bitmap, poseInfos);

                    var hBitmap = bitmap.GetHbitmap();
                    System.Windows.Media.Imaging.BitmapSource bitmapSrc = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
                                            hBitmap,
                                            IntPtr.Zero,
                                            Int32Rect.Empty,
                                            System.Windows.Media.Imaging.BitmapSizeOptions.FromEmptyOptions());

                    bitmapSrc.Freeze();
                    image_DisplayedImage.Source = bitmapSrc;

                    results.Dispose();
                    bitmap.Dispose();
                    yoloPoseModelHandle.Dispose();

                }

            }

            if (slider_frameIndex.Value != frameIndex) { slider_frameIndex.Value = frameIndex; }

            this.frameIndex = frameIndex;

            label_framePosition.Content = $"{frameIndex} / {slider_frameIndex.Maximum}";

        }

        private void button_LoadFile_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "MP4|*.mp4";
            ofd.FilterIndex = 0;
            if (ofd.ShowDialog() != true) return;

            OpenMovieFile(ofd.FileName);
        }

        private void frameIndexShift(int shiftValue)
        {
            if (capture == null) return;

            int targetIndex = frameIndex + shiftValue;
            if (targetIndex < 0) targetIndex = 0;
            if (targetIndex >= (int)slider_frameIndex.Maximum) targetIndex = (int)slider_frameIndex.Maximum - 1;

            slider_frameIndex.Value = targetIndex;

        }

        private void Window_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (tabItem_Config.IsSelected)
            {
                if (e.Key == Key.F5)
                {
                    ShowFrame((int)slider_frameIndex.Value);
                    e.Handled = true;
                }

                if (e.Key == Key.Left)
                {
                    if (Keyboard.Modifiers == ModifierKeys.Control)
                    {
                        frameIndexShift(-5); e.Handled = true;
                    }
                    else if (Keyboard.Modifiers == ModifierKeys.Alt)
                    {
                        frameIndexShift(-1); e.Handled = true;
                    }

                }
                if (e.Key == Key.Right)
                {
                    if (Keyboard.Modifiers == ModifierKeys.Control)
                    {
                        frameIndexShift(5); e.Handled = true;
                    }
                    else if (Keyboard.Modifiers == ModifierKeys.Alt)
                    {
                        frameIndexShift(1); e.Handled = true;
                    }

                }

                if ((Keyboard.Modifiers & ModifierKeys.Alt) == ModifierKeys.Alt && e.SystemKey == Key.Right)
                {
                    frameIndexShift(1);
                    e.Handled = true;
                }

                if ((Keyboard.Modifiers & ModifierKeys.Alt) == ModifierKeys.Alt && e.SystemKey == Key.Left)
                {
                    frameIndexShift(-1);
                    e.Handled = true;
                }

            }
        }

        private void button_SkipBackFewFrames_Click(object sender, RoutedEventArgs e)
        {
            frameIndexShift(-5);
        }

        private void button_PreviousFrame_Click(object sender, RoutedEventArgs e)
        {
            frameIndexShift(-1);
        }

        private void button_NextFrame_Click(object sender, RoutedEventArgs e)
        {
            frameIndexShift(1);
        }

        private void button_SkipForwardFewFrames_Click(object sender, RoutedEventArgs e)
        {
            frameIndexShift(5);
        }

        private void button_ReLoadFrame_Click(object sender, RoutedEventArgs e)
        {
            ShowFrame((int)slider_frameIndex.Value);
        }


    }

}


