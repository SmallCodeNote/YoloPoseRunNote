using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;

namespace YoloPoseRun
{
    public class PoseKeyPoints
    {
        public KeyPoint Nose;
        public KeyPoint EyeLeft;
        public KeyPoint EyeRight;
        public KeyPoint EarLeft;
        public KeyPoint EarRight;
        public KeyPoint ShoulderLeft;
        public KeyPoint ShoulderRight;
        public KeyPoint ElbowLeft;
        public KeyPoint ElbowRight;
        public KeyPoint WristLeft;
        public KeyPoint WristRight;
        public KeyPoint HipLeft;
        public KeyPoint HipRight;
        public KeyPoint KneeLeft;
        public KeyPoint KneeRight;
        public KeyPoint AnkleLeft;
        public KeyPoint AnkleRight;

        private float KeyPointAngle(KeyPoint pA0, float ConfidenceA0, KeyPoint pA1, float ConfidenceA1,
                                    KeyPoint pB0, float ConfidenceB0, KeyPoint pB1, float ConfidenceB1)
        {
            if (pA0.Confidence >= ConfidenceA0 && pA1.Confidence >= ConfidenceA1 &&
                pB0.Confidence >= ConfidenceB0 && pB1.Confidence >= ConfidenceB1)
            {
                double vA_X = pA1.X - pA0.X;
                double vA_Y = pA1.Y - pA0.Y;
                double vB_X = pB1.X - pB0.X;
                double vB_Y = pB1.Y - pB0.Y;

                double dotProduct = vA_X * vB_X + vA_Y * vB_Y;
                double magnitudeA = Math.Sqrt(vA_X * vA_X + vA_Y * vA_Y);
                double magnitudeB = Math.Sqrt(vB_X * vB_X + vB_Y * vB_Y);

                double angleRad = Math.Acos(dotProduct / (magnitudeA * magnitudeB));

                return (float)(angleRad * (180.0 / Math.PI));
            }
            else
            {
                return -1f;
            }
        }

        private float KeyPointAngle(KeyPoint p0, float Confidence0, KeyPoint p1, float Confidence1, KeyPoint p2, float Confidence2)
        {
            if (p0.Confidence >= Confidence0 && p1.Confidence >= Confidence1 && p2.Confidence >= Confidence2)
            {
                double v1X = p1.X - p0.X;
                double v1Y = p1.Y - p0.Y;
                double v2X = p2.X - p0.X;
                double v2Y = p2.Y - p0.Y;

                double dotProduct = v1X * v2X + v1Y * v2Y;

                double magnitudeV1 = Math.Sqrt(v1X * v1X + v1Y * v1Y);
                double magnitudeV2 = Math.Sqrt(v2X * v2X + v2Y * v2Y);

                double angleRad = Math.Acos(dotProduct / (magnitudeV1 * magnitudeV2));

                return (float)(angleRad * (180.0 / Math.PI));
            }
            else
            {
                return -1f;
            }
        }

        private float KeyPointAngleXLength(KeyPoint p0, float Confidence0, KeyPoint p1, float Confidence1, KeyPoint p2, float Confidence2)
        {
            if (p0.Confidence >= Confidence0 && p1.Confidence >= Confidence1 && p2.Confidence >= Confidence2)
            {
                double Length = KeyPointLength(p1, Confidence1, p2, Confidence2);

                double v1X = p1.X - p0.X;
                double v1Y = p1.Y - p0.Y;
                double v2X = p2.X - p0.X;
                double v2Y = p2.Y - p0.Y;

                double crossProduct = v1X * v2Y - v1Y * v2X;

                double magnitudeV1 = Math.Sqrt(v1X * v1X + v1Y * v1Y);
                double magnitudeV2 = Math.Sqrt(v2X * v2X + v2Y * v2Y);

                double angleRad = Math.Sin(crossProduct / (magnitudeV1 * magnitudeV2));

                if (angleRad >= 0) { return (float)Length; }
                else { return (float)(-1.0 * Length); }

            }
            else
            {
                return 0;
            }

        }

        private float KeyPointLength(KeyPoint p0, float Confidence0, KeyPoint p1, float Confidence1)
        {
            if (p0.Confidence >= Confidence0 && p1.Confidence >= Confidence1)
            {
                double v1X = p1.X - p0.X;
                double v1Y = p1.Y - p0.Y;

                return (float)(Math.Sqrt(v1X * v1X + v1Y * v1Y));
            }
            else
            {
                return -1f;
            }
        }

        private KeyPoint KeyPointSum(KeyPoint p1, KeyPoint p2, float Confidence)
        {
            if (p1.Confidence >= Confidence && p2.Confidence >= Confidence)
            {
                return new KeyPoint((p1.X + p2.X) * 0.5f, (p1.Y + p2.Y) * 0.5f, p1.Confidence < p2.Confidence ? p1.Confidence : p2.Confidence);
            }
            else if (p1.Confidence >= Confidence)
            {
                return new KeyPoint(p1.X, p1.Y, p1.Confidence);
            }
            else if (p2.Confidence >= Confidence)
            {
                return new KeyPoint(p2.X, p2.Y, p2.Confidence);
            }


            return new KeyPoint();
        }

        private KeyPoint KeyPointAve(float Confidence, params KeyPoint[] ps)
        {
            float pCount = 0;
            float Xsum = 0;
            float Ysum = 0;
            float ConfidenceMin = 1f;

            foreach (var p in ps)
            {
                if (p.Confidence >= Confidence)
                {
                    ConfidenceMin = p.Confidence < ConfidenceMin ? p.Confidence : ConfidenceMin;
                    Xsum += p.X;
                    Ysum += p.Y;
                    pCount++;
                }
            }

            if (pCount <= 0) { return new KeyPoint(); }
            return new KeyPoint(Xsum / pCount, Ysum / pCount, ConfidenceMin);
        }

        public KeyPoint Head(float confidenceLevel) { return KeyPointAve(confidenceLevel, Nose, EyeLeft, EyeRight, EarLeft, EarRight); }

        public KeyPoint Eye(float confidenceLevel) { return KeyPointSum(EyeLeft, EyeRight, confidenceLevel); }
        public KeyPoint Ear(float confidenceLevel) { return KeyPointSum(EarLeft, EarRight, confidenceLevel); }
        public KeyPoint Shoulder(float confidenceLevel) { return KeyPointSum(ShoulderLeft, ShoulderRight, confidenceLevel); }
        public KeyPoint Elbow(float confidenceLevel) { return KeyPointSum(ElbowLeft, ElbowRight, confidenceLevel); }
        public KeyPoint Wrist(float confidenceLevel) { return KeyPointSum(WristLeft, WristRight, confidenceLevel); }
        public KeyPoint Hip(float confidenceLevel) { return KeyPointSum(HipLeft, HipRight, confidenceLevel); }
        public KeyPoint Knee(float confidenceLevel) { return KeyPointSum(KneeLeft, KneeRight, confidenceLevel); }
        public KeyPoint Ankle(float confidenceLevel) { return KeyPointSum(AnkleLeft, AnkleRight, confidenceLevel); }

        public KeyPoint Head() { return KeyPointAve(confidenceLevel_Head, Nose, EyeLeft, EyeRight, EarLeft, EarRight); }
        public KeyPoint Eye() { return KeyPointSum(EyeLeft, EyeRight, confidenceLevel_Eye); }
        public KeyPoint Ear() { return KeyPointSum(EarLeft, EarRight, confidenceLevel_Ear); }
        public KeyPoint Shoulder() { return KeyPointSum(ShoulderLeft, ShoulderRight, confidenceLevel_Shoulder); }
        public KeyPoint Elbow() { return KeyPointSum(ElbowLeft, ElbowRight, confidenceLevel_Elbow); }
        public KeyPoint Wrist() { return KeyPointSum(WristLeft, WristRight, confidenceLevel_Wrist); }
        public KeyPoint Hip() { return KeyPointSum(HipLeft, HipRight, confidenceLevel_Hip); }
        public KeyPoint Knee() { return KeyPointSum(KneeLeft, KneeRight, confidenceLevel_Knee); }
        public KeyPoint Ankle() { return KeyPointSum(AnkleLeft, AnkleRight, confidenceLevel_Ankle); }

        public float ElbowLeftAngle { get { return KeyPointAngle(ElbowLeft, confidenceLevel_Elbow, ShoulderLeft, confidenceLevel_Shoulder, WristLeft, confidenceLevel_Wrist); } }
        public float ElbowRightAngle { get { return KeyPointAngle(ElbowRight, confidenceLevel_Elbow, ShoulderRight, confidenceLevel_Shoulder, WristRight, confidenceLevel_Wrist); } }
        public float KneeLeftAngle { get { return KeyPointAngle(KneeLeft, confidenceLevel_Knee, HipLeft, confidenceLevel_Hip, AnkleLeft, confidenceLevel_Ankle); } }
        public float KneeRightAngle { get { return KeyPointAngle(KneeRight, confidenceLevel_Knee, HipRight, confidenceLevel_Hip, AnkleRight, confidenceLevel_Ankle); } }

        public float ThighLeftTorsoAngle { get { return KeyPointAngle(HipLeft, confidenceLevel_Hip, KneeLeft, confidenceLevel_Knee, Hip(), confidenceLevel_Hip, Shoulder(), confidenceLevel_Shoulder); } }
        public float ThighRightTorsoAngle { get { return KeyPointAngle(HipRight, confidenceLevel_Hip, KneeRight, confidenceLevel_Knee, Hip(), confidenceLevel_Hip, Shoulder(), confidenceLevel_Shoulder); } }
        public float ArmLeftTorsoAngle { get { return KeyPointAngle(ShoulderLeft, confidenceLevel_Shoulder, ElbowLeft, confidenceLevel_Elbow, Shoulder(), confidenceLevel_Shoulder, Hip(), confidenceLevel_Hip); } }
        public float ArmRightTorsoAngle { get { return KeyPointAngle(ShoulderRight, confidenceLevel_Shoulder, ElbowRight, confidenceLevel_Elbow, Shoulder(), confidenceLevel_Shoulder, Hip(), confidenceLevel_Hip); } }

        public float WristLeftLength { get { return KeyPointLength(ElbowLeft, confidenceLevel_Elbow, WristLeft, confidenceLevel_Wrist); } }
        public float WristRightLength { get { return KeyPointLength(ElbowRight, confidenceLevel_Elbow, WristRight, confidenceLevel_Wrist); } }
        public float ElbowLeftLength { get { return KeyPointLength(ShoulderLeft, confidenceLevel_Shoulder, ElbowLeft, confidenceLevel_Elbow); } }
        public float ElbowRightLength { get { return KeyPointLength(ShoulderRight, confidenceLevel_Shoulder, ElbowRight, confidenceLevel_Elbow); } }
        public float KneeLeftLength { get { return KeyPointLength(HipLeft, confidenceLevel_Hip, KneeLeft, confidenceLevel_Knee); } }
        public float KneeRightLength { get { return KeyPointLength(HipRight, confidenceLevel_Hip, KneeRight, confidenceLevel_Knee); } }
        public float AnkleLeftLength { get { return KeyPointLength(KneeLeft, confidenceLevel_Knee, AnkleLeft, confidenceLevel_Ankle); } }
        public float AnkleRightLength { get { return KeyPointLength(KneeRight, confidenceLevel_Knee, AnkleRight, confidenceLevel_Ankle); } }

        public float TorsoLength { get { return KeyPointLength(Shoulder(), confidenceLevel_Shoulder, Hip(), confidenceLevel_Hip); } }
        public float ShoulderWidth { get { return KeyPointAngleXLength(Head(), confidenceLevel_Head, ShoulderLeft, confidenceLevel_Shoulder, ShoulderRight, confidenceLevel_Shoulder); } }

        public float HipWidth { get { return KeyPointAngleXLength(Shoulder(), confidenceLevel_Shoulder, HipLeft, confidenceLevel_Hip, HipRight, confidenceLevel_Hip); } }
        public float EyeWidth { get { return KeyPointAngleXLength(Shoulder(), confidenceLevel_Shoulder, EyeRight, confidenceLevel_Eye, EyeLeft, confidenceLevel_Eye); } }
        public float EarWidth { get { return KeyPointAngleXLength(Shoulder(), confidenceLevel_Shoulder, EarRight, confidenceLevel_Ear, EarLeft, confidenceLevel_Ear); } }

        public float HeadYawAngle
        {
            get
            {
                double dLeft = KeyPointLength(Nose, confidenceLevel_Nose, EarLeft, confidenceLevel_Ear);
                double dRight = KeyPointLength(Nose, confidenceLevel_Nose, EarRight, confidenceLevel_Ear);
                double sum = dLeft + dRight;
                if (sum == 0)
                    return -1;
                double ratio = (dRight - dLeft) / sum;
                double angleRad = (float)Math.Asin(ratio);
                if (double.IsNaN(angleRad)) return -1;

                return (float)(angleRad * (180.0 / Math.PI));
            }
        }

        public float TorsoSlope
        {
            get
            {
                KeyPoint shoulderMidpoint = Shoulder();
                KeyPoint hipMidpoint = Hip();
                return (float)(Math.Atan2(shoulderMidpoint.X - hipMidpoint.X, shoulderMidpoint.Y - hipMidpoint.Y) * (180.0 / Math.PI));
            }
        }

        public float ShoulderSlope
        {
            get
            {
                return (float)(Math.Atan2(ShoulderRight.Y - ShoulderLeft.Y, ShoulderRight.X - ShoulderLeft.X) * (180.0 / Math.PI));
            }
        }

        public float confidenceLevel_Nose = 0.6f;
        public float confidenceLevel_Head = 0.6f;
        public float confidenceLevel_Eye = 0.6f;
        public float confidenceLevel_Ear = 0.6f;
        public float confidenceLevel_Shoulder = 0.6f;
        public float confidenceLevel_Elbow = 0.6f;
        public float confidenceLevel_Wrist = 0.6f;
        public float confidenceLevel_Hip = 0.6f;
        public float confidenceLevel_Knee = 0.6f;
        public float confidenceLevel_Ankle = 0.6f;

        public void setConfidenceLevel(float confidenceLevel)
        {
            confidenceLevel_Nose = confidenceLevel;
            confidenceLevel_Head = confidenceLevel;
            confidenceLevel_Eye = confidenceLevel;
            confidenceLevel_Ear = confidenceLevel;
            confidenceLevel_Shoulder = confidenceLevel;
            confidenceLevel_Elbow = confidenceLevel;
            confidenceLevel_Wrist = confidenceLevel;
            confidenceLevel_Hip = confidenceLevel;
            confidenceLevel_Knee = confidenceLevel;
            confidenceLevel_Ankle = confidenceLevel;

        }

        public PoseKeyPoints(float[] output, int startIndex)
        {
            Nose = new KeyPoint(output, startIndex, 0);
            EyeLeft = new KeyPoint(output, startIndex, 1);
            EyeRight = new KeyPoint(output, startIndex, 2);
            EarLeft = new KeyPoint(output, startIndex, 3);
            EarRight = new KeyPoint(output, startIndex, 4);
            ShoulderLeft = new KeyPoint(output, startIndex, 5);
            ShoulderRight = new KeyPoint(output, startIndex, 6);
            ElbowLeft = new KeyPoint(output, startIndex, 7);
            ElbowRight = new KeyPoint(output, startIndex, 8);
            WristLeft = new KeyPoint(output, startIndex, 9);
            WristRight = new KeyPoint(output, startIndex, 10);
            HipLeft = new KeyPoint(output, startIndex, 11);
            HipRight = new KeyPoint(output, startIndex, 12);
            KneeLeft = new KeyPoint(output, startIndex, 13);
            KneeRight = new KeyPoint(output, startIndex, 14);
            AnkleLeft = new KeyPoint(output, startIndex, 15);
            AnkleRight = new KeyPoint(output, startIndex, 16);
        }

        public void drawBone(Graphics g, float confidenceLevel = 0.6f, float diameter = 8)
        {
            Pen p = new Pen(Color.Blue, 2);

            if (Nose.Confidence >= confidenceLevel && EyeLeft.Confidence >= confidenceLevel)
                g.DrawLine(p, Nose.Position, EyeLeft.Position);
            if (EyeLeft.Confidence >= confidenceLevel && EarLeft.Confidence >= confidenceLevel)
                g.DrawLine(p, EyeLeft.Position, EarLeft.Position);
            if (Nose.Confidence >= confidenceLevel && EyeRight.Confidence >= confidenceLevel)
                g.DrawLine(p, Nose.Position, EyeRight.Position);
            if (EyeRight.Confidence >= confidenceLevel && EarRight.Confidence >= confidenceLevel)
                g.DrawLine(p, EyeRight.Position, EarRight.Position);
            if (ShoulderLeft.Confidence >= confidenceLevel && ShoulderRight.Confidence >= confidenceLevel)
                g.DrawLine(p, ShoulderLeft.Position, ShoulderRight.Position);
            if (ShoulderLeft.Confidence >= confidenceLevel && ElbowLeft.Confidence >= confidenceLevel)
                g.DrawLine(p, ShoulderLeft.Position, ElbowLeft.Position);
            if (ElbowLeft.Confidence >= confidenceLevel && WristLeft.Confidence >= confidenceLevel)
                g.DrawLine(p, ElbowLeft.Position, WristLeft.Position);
            if (ShoulderRight.Confidence >= confidenceLevel && ElbowRight.Confidence >= confidenceLevel)
                g.DrawLine(p, ShoulderRight.Position, ElbowRight.Position);
            if (ElbowRight.Confidence >= confidenceLevel && WristRight.Confidence >= confidenceLevel)
                g.DrawLine(p, ElbowRight.Position, WristRight.Position);
            if (HipLeft.Confidence >= confidenceLevel && HipRight.Confidence >= confidenceLevel)
                g.DrawLine(p, HipLeft.Position, HipRight.Position);
            if (HipLeft.Confidence >= confidenceLevel && KneeLeft.Confidence >= confidenceLevel)
                g.DrawLine(p, HipLeft.Position, KneeLeft.Position);
            if (KneeLeft.Confidence >= confidenceLevel && AnkleLeft.Confidence >= confidenceLevel)
                g.DrawLine(p, KneeLeft.Position, AnkleLeft.Position);
            if (HipRight.Confidence >= confidenceLevel && KneeRight.Confidence >= confidenceLevel)
                g.DrawLine(p, HipRight.Position, KneeRight.Position);
            if (KneeRight.Confidence >= confidenceLevel && AnkleRight.Confidence >= confidenceLevel)
                g.DrawLine(p, KneeRight.Position, AnkleRight.Position);


            if (AnkleRight.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.LightPink, AnkleRight.GetRectangle(diameter));
            if (AnkleLeft.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.LightBlue, AnkleLeft.GetRectangle(diameter));
            if (KneeRight.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.LightPink, KneeRight.GetRectangle(diameter));
            if (KneeLeft.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.LightBlue, KneeLeft.GetRectangle(diameter));
            if (HipRight.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.LightPink, HipRight.GetRectangle(diameter));
            if (HipLeft.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.LightBlue, HipLeft.GetRectangle(diameter));
            if (WristRight.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.LightPink, WristRight.GetRectangle(diameter));
            if (WristLeft.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.LightBlue, WristLeft.GetRectangle(diameter));
            if (ElbowRight.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.LightPink, ElbowRight.GetRectangle(diameter));
            if (ElbowLeft.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.LightBlue, ElbowLeft.GetRectangle(diameter));
            if (ShoulderRight.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.LightPink, ShoulderRight.GetRectangle(diameter));
            if (ShoulderLeft.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.LightBlue, ShoulderLeft.GetRectangle(diameter));
            if (EarRight.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.LightPink, EarRight.GetRectangle(diameter));
            if (EarLeft.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.LightBlue, EarLeft.GetRectangle(diameter));
            if (EyeRight.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.LightPink, EyeRight.GetRectangle(diameter));
            if (EyeLeft.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.LightBlue, EyeLeft.GetRectangle(diameter));
            if (Nose.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.GreenYellow, Nose.GetRectangle(diameter));

            KeyPoint hip = Hip();
            if (hip.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.Violet, hip.GetRectangle(diameter - 2));

            KeyPoint shoulder = Shoulder();
            if (shoulder.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.Violet, shoulder.GetRectangle(diameter - 2));

            KeyPoint ear = Ear();
            if (ear.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.Violet, ear.GetRectangle(diameter - 2));

            KeyPoint eye = Eye();
            if (eye.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.Violet, eye.GetRectangle(diameter - 2));

            KeyPoint head = Head();
            if (head.Confidence >= confidenceLevel)
                g.FillEllipse(Brushes.Violet, head.GetRectangle(diameter - 2));

        }

        public override string ToString()
        {
            return $"{Nose.Confidence:0.00}, {EyeLeft.Confidence:0.00}, {EyeRight.Confidence:0.00}, " +
                   $"{EarLeft.Confidence:0.00}, {EarRight.Confidence:0.00}, {ShoulderLeft.Confidence:0.00}, " +
                   $"{ShoulderRight.Confidence:0.00}, {ElbowLeft.Confidence:0.00}, {ElbowRight.Confidence:0.00}, " +
                   $"{WristLeft.Confidence:0.00}, {WristRight.Confidence:0.00}, {HipLeft.Confidence:0.00}, " +
                   $"{HipRight.Confidence:0.00}, {KneeLeft.Confidence:0.00}, {KneeRight.Confidence:0.00}, " +
                   $"{AnkleLeft.Confidence:0.00}, {AnkleRight.Confidence:0.00}";
        }


        public static List<KeyPoint> SortClockwise(List<KeyPoint> points)
        {

            float centerX = 0, centerY = 0;
            foreach (var pt in points)
            {
                centerX += pt.X;
                centerY += pt.Y;
            }
            centerX /= points.Count;
            centerY /= points.Count;


            points.Sort((a, b) =>
            {
                double angleA = Math.Atan2(a.Y - centerY, a.X - centerX);
                double angleB = Math.Atan2(b.Y - centerY, b.X - centerX);
                return angleB.CompareTo(angleA);
            });

            return points;
        }

        public static List<KeyPoint> SortCounterClockwise(List<KeyPoint> points)
        {

            float centerX = 0, centerY = 0;
            foreach (var pt in points)
            {
                centerX += pt.X;
                centerY += pt.Y;
            }
            centerX /= points.Count;
            centerY /= points.Count;


            points.Sort((a, b) =>
            {
                double angleA = Math.Atan2(a.Y - centerY, a.X - centerX);
                double angleB = Math.Atan2(b.Y - centerY, b.X - centerX);
                return angleA.CompareTo(angleB);
            });

            return points;
        }

        public static List<KeyPoint> ClipPolygon(List<KeyPoint> subjectSrc, List<KeyPoint> clipSrc)
        {
            List<KeyPoint> subject = SortCounterClockwise(subjectSrc);
            List<KeyPoint> clip = SortCounterClockwise(clipSrc);


            List<KeyPoint> output = new List<KeyPoint>(subject);

            for (int i = 0; i < clip.Count; i++)
            {
                List<KeyPoint> input = new List<KeyPoint>(output);
                output.Clear();

                KeyPoint A = clip[i];
                int K = (i + 1) % clip.Count;
                KeyPoint B = clip[K];

                for (int j = 0; j < input.Count; j++)
                {
                    KeyPoint P = input[j];
                    int L = (j + 1) % input.Count;
                    KeyPoint Q = input[L];

                    bool insideP = IsInside(A, B, P);
                    bool insideQ = IsInside(A, B, Q);

                    if (insideP && insideQ)
                    {
                        output.Add(Q);
                    }
                    else if (insideP && !insideQ)
                    {
                        output.Add(Intersect(A, B, P, Q));
                    }
                    else if (!insideP && insideQ)
                    {
                        output.Add(Intersect(A, B, P, Q));
                        output.Add(Q);
                    }
                }
            }

            return output;
        }


        private static bool IsInside(KeyPoint A, KeyPoint B, KeyPoint P)
        {
            return (B.X - A.X) * (P.Y - A.Y) > (B.Y - A.Y) * (P.X - A.X);
        }

        private static KeyPoint Intersect(KeyPoint A, KeyPoint B, KeyPoint P, KeyPoint Q)
        {
            float confidence = new float[] { A.Confidence, B.Confidence, P.Confidence, Q.Confidence }.Max();

            double A1 = B.Y - A.Y;
            double B1 = A.X - B.X;
            double C1 = A1 * A.X + B1 * A.Y;

            double A2 = Q.Y - P.Y;
            double B2 = P.X - Q.X;
            double C2 = A2 * P.X + B2 * P.Y;

            double det = A1 * B2 - A2 * B1;
            if (Math.Abs(det) < 1e-10)
                return new KeyPoint(0, 0, confidence); // Degenerate case

            double x = (B2 * C1 - B1 * C2) / det;
            double y = (A1 * C2 - A2 * C1) / det;
            return new KeyPoint((float)x, (float)y, confidence);
        }

        private static double GetPolygonArea(List<KeyPoint> corners)
        {
            if (corners.Count < 3) return 0;

            double area = 0;
            for (int i = 0; i < corners.Count; i++)
            {
                KeyPoint p1 = corners[i];
                KeyPoint p2 = corners[(i + 1) % corners.Count];
                area += (p1.X * p2.Y) - (p2.X * p1.Y);
            }
            return Math.Abs(area) / 2.0;
        }

        public static float CalculateIntersectionOverUnion(List<KeyPoint> poly1, List<KeyPoint> poly2)
        {

            if (poly1.Count < 3 || poly2.Count < 3) return 0f;


            float area1 = (float)GetPolygonArea(poly1);
            float area2 = (float)GetPolygonArea(poly2);

            List<KeyPoint> intersectionPolygon = ClipPolygon(poly1, poly2);
            float intersectionArea = (float)GetPolygonArea(intersectionPolygon);

            float unionArea = area1 + area2 - intersectionArea;
            float result = unionArea > 0f ? intersectionArea / unionArea : 0;

            string Lines = "";

            foreach (var item in poly1)
            {
                Lines += item.ToString().Replace(",", "\t") + "\r\n";
            }

            foreach (var item in poly2)
            {
                Lines += item.ToString().Replace(",", "\t") + "\r\n";
            }


            return result;
        }


        public float OverlapUpperBody(PoseKeyPoints kps1, PoseKeyPoints kps2)
        {
            List<KeyPoint> poly1 = new List<KeyPoint> { kps1.ShoulderLeft, kps1.ShoulderRight, kps1.Head() };
            List<KeyPoint> poly2 = new List<KeyPoint> { kps2.ShoulderLeft, kps2.ShoulderRight, kps2.Head() };

            float result = CalculateIntersectionOverUnion(poly1, poly2);

            return result;
        }

        public float OverlapUpperBodyo(PoseInfo poseInfo)
        {
            return OverlapUpperBody(this, poseInfo.KeyPoints);
        }

        public float OverlapUpperBody(PoseKeyPoints kps)
        {
            float result = OverlapUpperBody(this, kps);
            return result;
        }

        public float OverlapTolso(PoseKeyPoints kps1, PoseKeyPoints kps2)
        {
            List<KeyPoint> poly1 = new List<KeyPoint> { kps1.ShoulderLeft, kps1.ShoulderRight, kps1.HipRight, kps1.HipLeft };
            List<KeyPoint> poly2 = new List<KeyPoint> { kps2.ShoulderLeft, kps2.ShoulderRight, kps2.HipRight, kps2.HipLeft };

            float result = CalculateIntersectionOverUnion(poly1, poly2);

            return result;
        }

        public float OverlapTolso(PoseInfo poseInfo)
        {
            return OverlapTolso(this, poseInfo.KeyPoints);
        }

        public float OverlapTolso(PoseKeyPoints kps)
        {
            float result = OverlapTolso(this, kps);
            return result;
        }

        public void Merge(PoseKeyPoints poseKeyPoints)
        {
            if (poseKeyPoints == null) return;

            this.Nose?.Merge(poseKeyPoints.Nose);
            this.EyeLeft?.Merge(poseKeyPoints.EyeLeft);
            this.EyeRight?.Merge(poseKeyPoints.EyeRight);
            this.EarLeft?.Merge(poseKeyPoints.EarLeft);
            this.EarRight?.Merge(poseKeyPoints.EarRight);
            this.ShoulderLeft?.Merge(poseKeyPoints.ShoulderLeft);
            this.ShoulderRight?.Merge(poseKeyPoints.ShoulderRight);
            this.ElbowLeft?.Merge(poseKeyPoints.ElbowLeft);
            this.ElbowRight?.Merge(poseKeyPoints.ElbowRight);
            this.WristLeft?.Merge(poseKeyPoints.WristLeft);
            this.WristRight?.Merge(poseKeyPoints.WristRight);
            this.HipLeft?.Merge(poseKeyPoints.HipLeft);
            this.HipRight?.Merge(poseKeyPoints.HipRight);
            this.KneeLeft?.Merge(poseKeyPoints.KneeLeft);
            this.KneeRight?.Merge(poseKeyPoints.KneeRight);
            this.AnkleLeft?.Merge(poseKeyPoints.AnkleLeft);
            this.AnkleRight?.Merge(poseKeyPoints.AnkleRight);

        }
    }

    public class KeyPoint
    {
        public float X;
        public float Y;
        public float Confidence;
        private int stride = 8400;

        public System.Drawing.Point Position { get { return new System.Drawing.Point((int)X, (int)Y); } }

        public Rectangle GetRectangle(float diameter = 12)
        {
            float radius = (diameter / 2.0f);
            return new Rectangle((int)(X - radius), (int)(Y - radius), (int)diameter, (int)diameter);
        }

        public KeyPoint(float[] output, int startIndex, int keyIndex)
        {
            this.X = output[startIndex + stride * (keyIndex * 3 + 5)];
            this.Y = output[startIndex + stride * (keyIndex * 3 + 6)];
            this.Confidence = output[startIndex + stride * (keyIndex * 3 + 7)];
        }

        public KeyPoint(float X, float Y, float Confidence)
        {
            this.X = X;
            this.Y = Y;
            this.Confidence = Confidence;
        }

        public KeyPoint()
        {
            this.X = 0;
            this.Y = 0;
            this.Confidence = 0;
        }

        public override string ToString()
        {
            return $"{X:0},{Y:0},{Confidence:0.00}";
        }

        public void Merge(KeyPoint keyPoint)
        {
            if (this.X == 0) { this.X = keyPoint.X; }
            if (this.Y == 0) { this.Y = keyPoint.Y; }
            if (this.X != 0 && keyPoint.X != 0) { this.X = (this.X + keyPoint.X) * 0.5f; }
            if (this.Y != 0 && keyPoint.Y != 0) { this.Y = (this.Y + keyPoint.Y) * 0.5f; }
        }
    }
}
