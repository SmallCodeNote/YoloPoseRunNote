using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

using Microsoft.Win32;

using ControlValuesToStringClass;

namespace YoloPoseRun
{
    public partial class MainWindow : Window
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
                yoloPose = new YoloPoseRunClass(srcFileQueue, textBox_modelFilePath.Text, deviceID,textBox_initializeLinesString.Text);
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
            PoseKeyPoints k = new PoseKeyPoints(null,-1,"");
            PoseInfo_ConfidenceLevel c = new PoseInfo_ConfidenceLevel();
            PoseInfo_OverLapThresholds o = new PoseInfo_OverLapThresholds();

            textBox_initializeLinesString.Text = o.ParamToTextLinesString()+"\r\n"+ c.ParamToTextLinesString();
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
    }

    

}
