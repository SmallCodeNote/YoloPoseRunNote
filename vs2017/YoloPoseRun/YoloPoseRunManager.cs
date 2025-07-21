using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

using System.IO;

using System.ComponentModel;
using System.Collections.ObjectModel;

namespace YoloPoseRun
{
    public class YoloPoseRunManager : INotifyPropertyChanged
    {
        public ConcurrentQueue<string> srcFileList;
        public ObservableCollection<YoloPoseRunClass> ProcessRuns;
        public List<string> ProcessNames;
        private string _aggregatedCountText = "... no progress data ...";

        public YoloPoseRunManager(ConcurrentQueue<string> srcFileList)
        {
            this.srcFileList = srcFileList;
            ProcessNames = new List<string>();
            ProcessRuns = new ObservableCollection<YoloPoseRunClass>();
        }

        public void Clear()
        {
            if (srcFileList != null) while (srcFileList.TryDequeue(out _)) { };
            if (ProcessRuns != null) ProcessRuns.Clear();
            if (ProcessNames != null) ProcessNames.Clear();
            IsComplete = false;
        }

        private int _progressCount = 0;
        public int ProgressCount
        {
            get { return _progressCount; }
            private set
            {
                getDebugInfo(System.Reflection.MethodBase.GetCurrentMethod().Name + $" : {value}");
                _progressCount = value;
                Update_aggregatedText();
            }
        }

        private bool _isComplete = false;
        public bool IsComplete
        {
            get { return _isComplete; }
            set
            {
                getDebugInfo(System.Reflection.MethodBase.GetCurrentMethod().Name + $" : {value}");

                if (_isComplete != value)
                {
                    _isComplete = value;
                    Update_aggregatedText();
                }
            }
        }

        public string AggregatedCountText
        {
            get => _aggregatedCountText;
        }

        public void Add(YoloPoseRunClass run, string name)
        {
            getDebugInfo(System.Reflection.MethodBase.GetCurrentMethod().Name + $" :{name}");

            if (run == null) return;

            run.PropertyChanged += (_, e) =>
            {
                getDebugInfo(System.Reflection.MethodBase.GetCurrentMethod().Name);
                Update_aggregatedText();
            };

            ProcessRuns.Add(run);
            ProcessNames.Add(name);
        }

        private void Update_aggregatedText()
        {
            getDebugInfo(System.Reflection.MethodBase.GetCurrentMethod().Name);

            int totalCount = 0;
            string aggregatedCountText = "";
            int progressCount = 0;

            try
            {
                List<string> report = new List<string>();
                if (srcFileList != null) totalCount = srcFileList.Count;
                int iMax = ProcessRuns.Count;
                Console.WriteLine($"-CALL:{System.Reflection.MethodBase.GetCurrentMethod().Name} {ProcessRuns.Count} {ProcessNames.Count}");
                for (int i = 0; i < iMax; i++)
                {
                    progressCount += ProcessRuns[i].ProcessRunCount;
                    report.Add($"{ProcessNames[i]} : {ProcessRuns[i].ProcessRunCount}");
                }

                aggregatedCountText = $"[{progressCount} / {totalCount}] " + string.Join(", ", report);

            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR:{System.Reflection.MethodBase.GetCurrentMethod().Name} {ex.Message} {ex.StackTrace}");
            }

            if (IsComplete) aggregatedCountText += " ... Task Run Complete";

            _aggregatedCountText = aggregatedCountText;
            _progressCount = progressCount;
            OnPropertyChanged(nameof(AggregatedCountText));
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected void OnPropertyChanged(string propertyName)
        {
            getDebugInfo(System.Reflection.MethodBase.GetCurrentMethod().Name + $" :{propertyName} [{_aggregatedCountText}]");
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public static void getDebugInfo(string methodName, [CallerLineNumber] int lineNumber = 0, [CallerFilePath] string filePath = null)
        {
            Console.WriteLine($"-CALL: {methodName} - {Path.GetFileName(filePath)}:{lineNumber}");
        }
    }
}
