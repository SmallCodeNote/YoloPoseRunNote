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
    public class Bbox
    {
        public float Center_x;
        public float Center_y;
        public float Width;
        public float Height;
        public float Confidence;
        private int stride = 8400;

        public Bbox(float[] outputArray, int startIndex)
        {
            this.Center_x = outputArray[startIndex + stride * 0];
            this.Center_y = outputArray[startIndex + stride * 1];
            this.Width = outputArray[startIndex + stride * 2];
            this.Height = outputArray[startIndex + stride * 3];
            this.Confidence = outputArray[startIndex + stride * 4];
        }

        public float Left { get { return Center_x - Width / 2.0f; } }
        public float Right { get { return Center_x + Width / 2.0f; } }
        public float Top { get { return Center_y - Height / 2.0f; } }
        public float Bottom { get { return Center_y + Height / 2.0f; } }
        public float Area { get { return Width * Height; } }

        public Rectangle Rectangle { get { return new Rectangle((int)Left, (int)Top, (int)Width, (int)Height); } }

        public float Overlap(Bbox bbox)
        {
            float intersectionLeft = Math.Max(this.Left, bbox.Left);
            float intersectionTop = Math.Max(this.Top, bbox.Top);
            float intersectionRight = Math.Min(this.Right, bbox.Right);
            float intersectionBottom = Math.Min(this.Bottom, bbox.Bottom);

            float intersectionWidth = Math.Max(0, intersectionRight - intersectionLeft);
            float intersectionHeight = Math.Max(0, intersectionBottom - intersectionTop);

            float intersectionArea = intersectionWidth * intersectionHeight;
            float thisArea = this.Area;
            float otherArea = bbox.Area;

            float unionArea = thisArea + otherArea - intersectionArea;

            return intersectionArea / unionArea;
        }

        public float Merge(Bbox bbox)
        {
            float intersectionLeft = Math.Max(this.Left, bbox.Left);
            float intersectionTop = Math.Max(this.Top, bbox.Top);
            float intersectionRight = Math.Min(this.Right, bbox.Right);
            float intersectionBottom = Math.Min(this.Bottom, bbox.Bottom);

            float intersectionWidth = Math.Max(0, intersectionRight - intersectionLeft);
            float intersectionHeight = Math.Max(0, intersectionBottom - intersectionTop);

            float intersectionArea = intersectionWidth * intersectionHeight;
            float thisArea = this.Area;
            float otherArea = bbox.Area;

            float unionArea = thisArea + otherArea - intersectionArea;

            if (unionArea > 0)
            {
                this.Center_x = (intersectionLeft + intersectionRight) * 0.5f;
                this.Center_y = (intersectionTop + intersectionBottom) * 0.5f;
                this.Width = intersectionWidth;
                this.Height = intersectionHeight;
            }

            return intersectionArea / unionArea;
        }

        public override string ToString()
        {
            return $"{Confidence:0.00},{Center_x:0},{Center_y:0},{Width:0},{Height:0}";
        }
    }
}
