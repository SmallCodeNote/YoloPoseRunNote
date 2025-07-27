using System;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.IO;

namespace YoloPoseRun
{
    public class PoseInfo_ConfidenceLevel
    {
        public float Bbox = 0.16f;

        public float Nose = 0.6f;
        public float Head = 0.6f;
        public float Eye = 0.6f;
        public float Ear = 0.6f;
        public float Shoulder = 0.6f;
        public float Elbow = 0.6f;
        public float Wrist = 0.6f;
        public float Hip = 0.6f;
        public float Knee = 0.6f;
        public float Ankle = 0.6f;

        public PoseInfo_ConfidenceLevel()
        {
        }

        public PoseInfo_ConfidenceLevel(string ConfidenceParameterLinesString)
        {
            InitializeParamFromTextLines(ConfidenceParameterLinesString);
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
                    case nameof(Bbox): Bbox = value; break;
                    case nameof(Nose): Nose = value; break;
                    case nameof(Head): Head = value; break;
                    case nameof(Eye): Eye = value; break;
                    case nameof(Ear): Ear = value; break;
                    case nameof(Shoulder): Shoulder = value; break;
                    case nameof(Elbow): Elbow = value; break;
                    case nameof(Wrist): Wrist = value; break;
                    case nameof(Hip): Hip = value; break;
                    case nameof(Knee): Knee = value; break;
                    case nameof(Ankle): Ankle = value; break;
                    default: break;
                }
            }
        }

        public void setKeypointsConfidenceLevel(float commonConfidenceLevel)
        {
            Nose = commonConfidenceLevel;
            Head = commonConfidenceLevel;
            Eye = commonConfidenceLevel;
            Ear = commonConfidenceLevel;
            Shoulder = commonConfidenceLevel;
            Elbow = commonConfidenceLevel;
            Wrist = commonConfidenceLevel;
            Hip = commonConfidenceLevel;
            Knee = commonConfidenceLevel;
            Ankle = commonConfidenceLevel;
        }


        public string ParamToTextLinesString()
        {
            return string.Join("\r\n", ParamToTextLines());
        }

        public string[] ParamToTextLines()
        {
            return new string[]
            {
                $"{nameof(Bbox)}\t{Bbox}",
                $"{nameof(Nose)}\t{Nose}",
                $"{nameof(Head)}\t{Head}",
                $"{nameof(Eye)}\t{Eye}",
                $"{nameof(Ear)}\t{Ear}",
                $"{nameof(Shoulder)}\t{Shoulder}",
                $"{nameof(Elbow)}\t{Elbow}",
                $"{nameof(Wrist)}\t{Wrist}",
                $"{nameof(Hip)}\t{Hip}",
                $"{nameof(Knee)}\t{Knee}",
                $"{nameof(Ankle)}\t{Ankle}"
            };
        }
    }

    public class PoseInfo_OverLapThresholds
    {
        public float OverlapBBoxThreshold = 0.8f;
        public float OverlapTolsoThreshold = 0.8f;
        public float OverlapShoulderThreshold = 0.8f;

        public PoseInfo_OverLapThresholds()
        {
        }

        public PoseInfo_OverLapThresholds(string LinesString)
        {
            InitializeParamFromTextLines(LinesString);
        }

        public void InitializeParamFromTextLines()
        {

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
                $"{nameof(OverlapBBoxThreshold)}\t{OverlapBBoxThreshold}",
                $"{nameof(OverlapTolsoThreshold)}\t{OverlapTolsoThreshold}",
                $"{nameof(OverlapShoulderThreshold)}\t{OverlapShoulderThreshold}"
            };
        }

    }


    public class PoseInfo
    {
        public Bbox Bbox;
        public PoseKeyPoints KeyPoints;

        public PoseInfo(float[] outputArray, int startIndex, PoseInfo_ConfidenceLevel confidenceLevel)
        {
            Bbox = new Bbox(outputArray, startIndex);
            KeyPoints = new PoseKeyPoints(outputArray, startIndex, confidenceLevel);
        }

        static public string ToLineStringHeader()
        {
            string linePoseHeader = "";

            linePoseHeader += "Bbox.X,Bbox.Y,Bbox.W,Bbox.H";

            linePoseHeader += ",Head.X,Head.Y";
            linePoseHeader += ",WristLeft.X,WristLeft.Y";
            linePoseHeader += ",WristRight.X,WristRight.Y";
            linePoseHeader += ",ElbowLeftAngle,ElbowLeftLength,WristLeftLength";
            linePoseHeader += ",ElbowRightAngle,ElbowRightLength,WristRightLength";
            linePoseHeader += ",KneeLeftAngle,KneeLeftLength,AnkleLeftLength";
            linePoseHeader += ",KneeRightAngle,KneeRightLength,AnkleRightLength";
            linePoseHeader += ",EyeWidth,EarWidth,ShoulderWidth,HipWidth";
            linePoseHeader += ",TorsoLength";
            linePoseHeader += ",HeadYawAngle";
            linePoseHeader += ",TorsoSlope";
            linePoseHeader += ",ShoulderSlope";
            linePoseHeader += ",ThighLeftTorsoAngle";
            linePoseHeader += ",ThighRightTorsoAngle";
            linePoseHeader += ",ArmLeftTorsoAngle";
            linePoseHeader += ",ArmRightTorsoAngle";

            return linePoseHeader;
        }

        public string ToLineString()
        {
            string linePose = "";
            linePose += $"{Bbox.Center_x:0},{Bbox.Center_y:0},{Bbox.Width:0},{Bbox.Height:0}";

            linePose += $",{KeyPoints.Head().X:0},{KeyPoints.Head().Y:0}";
            linePose += $",{KeyPoints.WristLeft.X:0},{KeyPoints.WristLeft.Y:0}";
            linePose += $",{KeyPoints.WristRight.X:0},{KeyPoints.WristRight.Y:0}";
            linePose += $",{KeyPoints.ElbowLeftAngle:0},{KeyPoints.ElbowLeftLength:0},{KeyPoints.WristLeftLength:0}";
            linePose += $",{KeyPoints.ElbowRightAngle:0},{KeyPoints.ElbowRightLength:0},{KeyPoints.WristRightLength:0}";
            linePose += $",{KeyPoints.KneeLeftAngle:0},{KeyPoints.KneeLeftLength:0},{KeyPoints.AnkleLeftLength:0}";
            linePose += $",{KeyPoints.KneeRightAngle:0},{KeyPoints.KneeRightLength:0},{KeyPoints.AnkleRightLength:0}";
            linePose += $",{KeyPoints.EyeWidth:0},{KeyPoints.EarWidth:0},{KeyPoints.ShoulderWidth:0},{KeyPoints.HipWidth:0}";
            linePose += $",{KeyPoints.TorsoLength:0}";
            linePose += $",{KeyPoints.HeadYawAngle:0}";
            linePose += $",{KeyPoints.TorsoSlope:0}";
            linePose += $",{KeyPoints.ShoulderSlope:0}";
            linePose += $",{KeyPoints.ThighLeftTorsoAngle:0}";
            linePose += $",{KeyPoints.ThighRightTorsoAngle:0}";
            linePose += $",{KeyPoints.ArmLeftTorsoAngle:0}";
            linePose += $",{KeyPoints.ArmRightTorsoAngle:0}";

            return linePose;
        }

        public string ToColString()
        {
            var headers = ToLineStringHeader().Split(',');
            var values = ToLineString().Split(',');

            if (headers.Length != values.Length)
            {
                return $"value.Length unmatch error \r\n{values.Length}/{headers.Length}";
            }

            var result = new StringBuilder();
            for (int i = 0; i < headers.Length; i++)
            {
                result.AppendLine($"{headers[i]}: {values[i]}");
            }

            return result.ToString();
        }

        public float Overlap(PoseInfo poseInfo, float threshold = 0.8f)
        {
            float o1 = this.Bbox.Overlap(poseInfo.Bbox);
            float ko1 = this.KeyPoints.OverlapTolso(poseInfo.KeyPoints);
            float ko2 = this.KeyPoints.OverlapUpperBody(poseInfo.KeyPoints);

            float o2 = (new float[] { ko1, ko2 }).Max();

            float result = o2 > threshold ? o2 : o1;
            //float result = Math.Max(o1, o2); ;

            return result;
        }

        public float OverlapBbox(PoseInfo poseInfo)
        {
            return this.Bbox.Overlap(poseInfo.Bbox);
        }

        public float OverlapTolso(PoseInfo poseInfo)
        {
            return this.KeyPoints.OverlapTolso(poseInfo.KeyPoints);
        }

        public float OverlapShoulder(PoseInfo poseInfo)
        {
            return this.KeyPoints.OverlapUpperBody(poseInfo.KeyPoints);
        }

        public void Merge(PoseInfo poseInfo)
        {
            this.KeyPoints.Merge(poseInfo.KeyPoints);
            this.Bbox.Merge(poseInfo.Bbox);
        }


        public static void __debug_CodeInfoWriteToConsole__(string methodName, [CallerLineNumber] int lineNumber = 0, [CallerFilePath] string filePath = null)
        {
            Console.WriteLine($"[{DateTime.Now:HH:mm:dd.sss}] {Path.GetFileName(filePath)}:{lineNumber} - {methodName}");
        }

        private void __debug_MessageWriteToConsole__(string message, [CallerLineNumber] int lineNumber = 0, [CallerFilePath] string filePath = null)
        {
            Console.WriteLine($"[-]\t{DateTime.Now:HH:mm:ss.fff}\t-\t{Path.GetFileName(filePath)}:{lineNumber}\t" + message);
        }

    }
}
