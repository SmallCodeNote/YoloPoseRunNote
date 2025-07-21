using System;
using System.Linq;
using System.Text;

namespace YoloPoseRun
{
    public class PoseInfo
    {
        public Bbox Bbox;
        public PoseKeyPoints KeyPoints;


        public PoseInfo(float[] outputArray, int startIndex, string ConfidenceParameterLinesString)
        {
            Bbox = new Bbox(outputArray, startIndex);
            KeyPoints = new PoseKeyPoints(outputArray, startIndex, ConfidenceParameterLinesString);
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

    }
}
