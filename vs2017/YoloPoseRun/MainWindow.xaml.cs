using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Threading;

using System.ComponentModel;
using System.Collections.ObjectModel;


using System.IO;

using Microsoft.Win32;

using ControlValuesToStringClass;

namespace YoloPoseRun
{
    public partial class MainWindow : Window
    {
        Task task_GPU0 = null;
        Task task_GPU1 = null;
        Task task_CPU = null;

        YoloPoseRunManager progressManager;
        YoloPoseRunClass yoloPose_GPU0;
        YoloPoseRunClass yoloPose_GPU1;
        YoloPoseRunClass yoloPose_CPU;

        public BlockingCollection<string> srcFileQueue =null;
        public List<string> srcFileList = new List<string>();
        List<Task> collectFilePathTaskList = new List<Task>();

        private CancellationTokenSource tokenSource = new CancellationTokenSource();

        public MainWindow()
        {
            InitializeComponent();

            progressManager = new YoloPoseRunManager();
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

                initiarizeTask(textBox_batchSize_CPU, out yoloPose_CPU, out task_CPU, -1);
                initiarizeTask(textBox_batchSize_GPU0, out yoloPose_GPU0, out task_GPU0, 0);
                initiarizeTask(textBox_batchSize_GPU1, out yoloPose_GPU1, out task_GPU1, 1);

                if (yoloPose_CPU != null) progressManager.Add(yoloPose_CPU);
                if (yoloPose_GPU0 != null) progressManager.Add(yoloPose_GPU0);
                if (yoloPose_GPU1 != null) progressManager.Add(yoloPose_GPU1);

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
                    if (task_GPU0 != null) task_GPU0.Wait();
                    if (task_GPU1 != null) task_GPU1.Wait();
                    if (task_CPU != null) task_CPU.Wait();

                    taskComplete();

                });
            }
        }
        
        private string getDirectoryNameAndFilename(string path)
        {
            return System.IO.Path.Combine(System.IO.Path.GetFileName(System.IO.Path.GetDirectoryName(path)), System.IO.Path.GetFileName(path));
        }

        private void initiarizeTask(TextBox textBox,out YoloPoseRunClass yoloPose, out Task task, int deviceID)
        {
            if (int.TryParse(textBox.Text, out int batchSize) && batchSize > 0)
            {
                yoloPose = new YoloPoseRunClass(srcFileQueue, textBox_modelFilePath.Text, deviceID);
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
                if (task_GPU0 != null) task_GPU0.Wait();
                if (task_GPU1 != null) task_GPU1.Wait();
                if (task_CPU != null) task_CPU.Wait();

                taskComplete();

                

            });
        }

        private void taskComplete()
        {
            Dispatcher.Invoke((Action)(() =>
            {
                button_RunModel.Content = "▶";
            }));
        }

        public async Task collectFilePathAsync(string targetDirectoryPath, CancellationToken cancellationToken, string searchPattern = "*.*")
        {
            await Task.Run(() =>
            {
                try
                {
                    IEnumerable<string> allFilePaths = Directory.EnumerateFiles(targetDirectoryPath, searchPattern, SearchOption.TopDirectoryOnly);

                    foreach (string filePath in allFilePaths)
                    {
                        if (cancellationToken.IsCancellationRequested) break;
                        srcFileQueue.Add(filePath);
                        srcFileList.Add(filePath);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"ERROR:{System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");

                }

            }, cancellationToken);
        }
    }

    public class YoloPoseRunManager : INotifyPropertyChanged
    {
        public ObservableCollection<YoloPoseRunClass> ProcessRuns { get; } = new ObservableCollection<YoloPoseRunClass>();
        private string _aggregatedCountText = "... no data ...";
        public string AggregatedCountText
        {
            get => _aggregatedCountText;
            private set
            {
                if (_aggregatedCountText != value)
                {
                    _aggregatedCountText = value;
                    OnPropertyChanged(nameof(AggregatedCountText));
                }
            }
        }

        public void Add(YoloPoseRunClass run)
        {
            if (run == null) return;

            run.PropertyChanged += (_, e) =>
            {
                if (e.PropertyName == nameof(YoloPoseRunClass.ProcessRunCount))
                    UpdateAggregatedText();
            };

            ProcessRuns.Add(run);
        }

        private void UpdateAggregatedText()
        {
            try
            {
                AggregatedCountText = string.Join(",", ProcessRuns.Select(p => p.ProcessRunCount));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR:{System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected void OnPropertyChanged(string propertyName) =>
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));

    }


}
