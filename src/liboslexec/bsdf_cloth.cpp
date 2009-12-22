/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
fmodification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO evenT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, even IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cmath>
#include <OpenEXR/ImathMatrix.h>
#include <OpenEXR/ImathFun.h>

#include "oslops.h"
#include "oslexec_pvt.h"

#include "bsdf_cloth_fncs.h"

inline bool odd(int x)  { return (x & 1) == 1; }
inline bool even(int x) { return (x & 1) == 0; }

inline int whichtile(float x, float freq) { return (int)floor(x * freq); }

inline void set_ellipse_axes(Point2 &semimajor, Point2 &semiminor, float alpha, float eccentricity, float angle)
{
    semimajor.x = alpha;
    semimajor.y = 0.f;
    semiminor.x = 0.f;
    semiminor.y = semimajor.x * sqrtf(1.f-eccentricity*eccentricity);

    // rotate ellipse
    rotate_2D(semimajor, angle);
    rotate_2D(semiminor, angle);
}

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {


enum threadType
{
    LONGWARP,
    SHORTWARP,
    LONGWEFT,
    SHORTWEFT
};

typedef struct threadSegment
{
    float angle;
    float eccentricity;
    float Kx;
    float Ky;
    float alpha;
    float Sx;
    float Sy;
    Color3   diffCol;
    Color3   specCol;
    float R0;
    float patternWeight;    // determined by number of thread widths of this segment type used per thread pattern tile
    float beamCoverage;     // determined by coverage in uv texture space by beam footprint
    float AR;               // area of rectangular window
    Point2 semimajor;
    Point2 semiminor;
};


class ClothClosure : public BSDFClosure {
    Vec3 m_N;

    float m_s;
    float m_t;

    float m_dsdx;
    float m_dtdx;
    float m_dsdy;
    float m_dtdy;
    float m_area_scaled;
    Vec3 m_dPdu;

    Color3 m_diff_warp_col;
    Color3 m_diff_weft_col;
    Color3 m_spec_warp_col;
    Color3 m_spec_weft_col;

    float m_fresnel_warp;
    float m_fresnel_weft;
    float m_spread_x_mult;
    float m_spread_y_mult;
    int   m_pattern;
    float m_pattern_angle;
    float m_warp_width_scale;
    float m_weft_width_scale;
    float m_thread_count_mult_u;
    float m_thread_count_mult_v;

private:
    int threadTypeCount;
    threadSegment t_seg[4];
    int currentThreadSegment; // current BTF & unfiltered BRDF thread segment
    int threadPattern[4];     // there are four types of thread segments (for now)

    float uu, vv;
    float offset, uux, vvx;

    float threadCoverageEstimate;
    float maxThreadWidthBTF;
    float BRDF_interp;
    float BTF_interp;

public:
    CLOSURE_CTOR (ClothClosure) : BSDFClosure(side, Labels::DIFFUSE)
    {
        CLOSURE_FETCH_ARG (m_N, 1);

        CLOSURE_FETCH_ARG (m_s, 2);
        CLOSURE_FETCH_ARG (m_t, 3);

        CLOSURE_FETCH_ARG (m_dsdx, 4);
        CLOSURE_FETCH_ARG (m_dtdx, 5);
        CLOSURE_FETCH_ARG (m_dsdy, 6);
        CLOSURE_FETCH_ARG (m_dtdy, 7);
        CLOSURE_FETCH_ARG (m_area_scaled, 8);
        CLOSURE_FETCH_ARG (m_dPdu, 9);


        CLOSURE_FETCH_ARG (m_diff_warp_col, 10);
        CLOSURE_FETCH_ARG (m_diff_weft_col, 11);
        CLOSURE_FETCH_ARG (m_spec_warp_col, 12);
        CLOSURE_FETCH_ARG (m_spec_weft_col, 13);

        CLOSURE_FETCH_ARG (m_fresnel_warp, 14);  // warp is y direction
        CLOSURE_FETCH_ARG (m_fresnel_weft, 15);  // weft is x direction
        CLOSURE_FETCH_ARG (m_spread_x_mult, 16);
        CLOSURE_FETCH_ARG (m_spread_y_mult, 17);
        CLOSURE_FETCH_ARG (m_pattern, 18);      // the weave pattern
        CLOSURE_FETCH_ARG (m_pattern_angle, 19);
        CLOSURE_FETCH_ARG (m_warp_width_scale, 20);
        CLOSURE_FETCH_ARG (m_weft_width_scale, 21);
        CLOSURE_FETCH_ARG (m_thread_count_mult_u, 22);
        CLOSURE_FETCH_ARG (m_thread_count_mult_v, 23);

        //

        m_spread_x_mult = std::min(1.f/m_spread_x_mult, 10.f);
        m_spread_y_mult = std::min(1.f/m_spread_y_mult, 10.f);

        // long warp thread
        t_seg[LONGWARP].angle                   = 95.f;  // convert to radians
        t_seg[LONGWARP].eccentricity            = Imath::lerp(0.94625f, 0.995f, 1.0f-m_warp_width_scale);  // 0.995f
        t_seg[LONGWARP].Kx                      = 1.5f*m_spread_x_mult;
        t_seg[LONGWARP].Ky                      = 3.1f*m_spread_y_mult;
        t_seg[LONGWARP].alpha                   = 1.5f;
        t_seg[LONGWARP].Sx                      = 1.f;
        t_seg[LONGWARP].Sy                      = 3.f;
        t_seg[LONGWARP].diffCol                 = m_diff_warp_col;
        t_seg[LONGWARP].specCol                 = m_spec_warp_col;
        t_seg[LONGWARP].R0                      = m_fresnel_warp;
        t_seg[LONGWARP].patternWeight           = 1.f;
        t_seg[LONGWARP].beamCoverage            = 1.f;
        t_seg[LONGWARP].AR                      = t_seg[LONGWARP].Sx * t_seg[LONGWARP].Sy;
        set_ellipse_axes(t_seg[LONGWARP].semimajor, t_seg[LONGWARP].semiminor,
                         t_seg[LONGWARP].alpha, t_seg[LONGWARP].eccentricity, 
                         t_seg[LONGWARP].angle);

        // short warp thread
        t_seg[SHORTWARP].angle                  = 100.f;
        t_seg[SHORTWARP].eccentricity           = Imath::lerp(0.8f, 0.9f, 1.0f-m_warp_width_scale); // 0.9f
        t_seg[SHORTWARP].Kx                     = 1.6f*m_spread_x_mult;
        t_seg[SHORTWARP].Ky                     = 2.5f*m_spread_y_mult;
        t_seg[SHORTWARP].alpha                  = .5f;
        t_seg[SHORTWARP].Sx                     = 1.f;
        t_seg[SHORTWARP].Sy                     = 1.f;
        t_seg[SHORTWARP].diffCol                = m_diff_warp_col;
        t_seg[SHORTWARP].specCol                = m_spec_warp_col;
        t_seg[SHORTWARP].R0                     = m_fresnel_warp;
        t_seg[SHORTWARP].patternWeight          = 1.f;
        t_seg[SHORTWARP].beamCoverage           = 1.f;
        t_seg[SHORTWARP].AR                     = t_seg[SHORTWARP].Sx * t_seg[SHORTWARP].Sy;
        set_ellipse_axes(t_seg[SHORTWARP].semimajor, t_seg[SHORTWARP].semiminor,
                         t_seg[SHORTWARP].alpha, t_seg[SHORTWARP].eccentricity, 
                         t_seg[SHORTWARP].angle);

        // long weft thread
        t_seg[LONGWEFT].angle                  = 5.f;
        t_seg[LONGWEFT].eccentricity           = Imath::lerp(0.94625f, 0.995f, 1.0f-m_weft_width_scale); //0.995f;
        t_seg[LONGWEFT].Kx                     = 3.1f*m_spread_x_mult;
        t_seg[LONGWEFT].Ky                     = 1.5f*m_spread_y_mult;
        t_seg[LONGWEFT].alpha                  = 1.5f;
        t_seg[LONGWEFT].Sx                     = 3.f;
        t_seg[LONGWEFT].Sy                     = 1.f;
        t_seg[LONGWEFT].diffCol                = m_diff_weft_col;
        t_seg[LONGWEFT].specCol                = m_spec_weft_col;
        t_seg[LONGWEFT].R0                     = m_fresnel_weft;
        t_seg[LONGWEFT].patternWeight          = 1.f;
        t_seg[LONGWEFT].beamCoverage           = 1.f;
        t_seg[LONGWEFT].AR                     = t_seg[LONGWEFT].Sx * t_seg[LONGWEFT].Sy;
        set_ellipse_axes(t_seg[LONGWEFT].semimajor, t_seg[LONGWEFT].semiminor,
                         t_seg[LONGWEFT].alpha, t_seg[LONGWEFT].eccentricity, 
                         t_seg[LONGWEFT].angle);

        // short weft thread
        t_seg[SHORTWEFT].angle                  = 80.f;
        t_seg[SHORTWEFT].beamCoverage           = 1.f;
        t_seg[SHORTWEFT].eccentricity           = Imath::lerp(0.8f, 0.9f, 1.0f-m_weft_width_scale); //0.9f;
        t_seg[SHORTWEFT].Kx                     = 2.5f*m_spread_x_mult;
        t_seg[SHORTWEFT].Ky                     = 1.6f*m_spread_y_mult;
        t_seg[SHORTWEFT].alpha                  = 0.5f;
        t_seg[SHORTWEFT].Sx                     = 1.f;
        t_seg[SHORTWEFT].Sy                     = 1.f;
        t_seg[SHORTWEFT].diffCol                = m_diff_weft_col;
        t_seg[SHORTWEFT].specCol                = m_spec_weft_col;
        t_seg[SHORTWEFT].R0                     = m_fresnel_weft;
        t_seg[SHORTWEFT].patternWeight          = 1.f;
        t_seg[SHORTWEFT].beamCoverage           = 1.f;
        t_seg[SHORTWEFT].AR                     = t_seg[SHORTWEFT].Sx * t_seg[SHORTWEFT].Sy;
        set_ellipse_axes(t_seg[SHORTWEFT].semimajor, t_seg[SHORTWEFT].semiminor,
                         t_seg[SHORTWEFT].alpha, t_seg[SHORTWEFT].eccentricity, 
                         t_seg[SHORTWEFT].angle);

        Point2 uv;
        uv.x = m_s; uv.y = m_t;

        rotate_2D(uv, m_pattern_angle);

        uu = fmod((uv.x*m_thread_count_mult_u), 1);
        vv = fmod((uv.y*m_thread_count_mult_v), 1);

        int row, col;
        threadTypeCount = 1;

        // TWILL PATTERN
        //
        if(m_pattern == 1)
        {
            col = whichtile(uu, 12);

            uux = fmod((uu*12.f), 1);    // reference column uv coords
            vvx = fmod((vv*3.f), 1);     // warp uv coords

            offset = (col) * 0.25f;
            vvx = fmod(vvx+offset,1);
            vvx /= 0.75f;

            if(vvx > 1.0)
            {
                currentThreadSegment = SHORTWEFT;
                vvx = (vvx-1.f)*3.f;
            }
            else
                currentThreadSegment = LONGWARP;


            threadPattern[0] = LONGWARP;
            threadPattern[1] = SHORTWEFT;
            threadTypeCount = 2;
            t_seg[LONGWARP].patternWeight  = 0.75f;  // 108 thread widths out of 144
            t_seg[SHORTWEFT].patternWeight = 0.25f;  // 36 thread widths remaining
        }

        // PLAIN WEAVE PATTERN
        //
        if(m_pattern == 2)
        {
            col = whichtile(uu, 12);
            row = whichtile(vv, 12);

            uux = fmod((uu*12.f), 1.f);
            vvx = fmod((vv*12.f), 1.f);


            if((even(row) == true && even(col) == true) || (odd(row) == true && odd(col) == true) || row == col)
                currentThreadSegment = SHORTWEFT;

            else
                currentThreadSegment = SHORTWARP;


            threadPattern[0] = SHORTWARP;
            threadPattern[1] = SHORTWEFT;
            threadTypeCount = 2;
            t_seg[SHORTWARP].patternWeight = 0.5f;  // 72 thread widths out of 144
            t_seg[SHORTWEFT].patternWeight = 0.5f;  // ditto the above
        }

        // SATIN PATTERN
        //
        if(m_pattern == 3)
        {
            uu = fmod((uv.x*m_thread_count_mult_u*1.5), 1.f);

            col = whichtile(uu, 8);
            row = whichtile(vv, 12);

            uux = fmod((uu*8), 1.f);
            vvx = fmod((vv*3), 1.f);

            // TODO: this is screwy... starting this pattern in the wrong place, damnit.
            //
            if(col == 1)
                offset = 0.25f;
            else if(col == 3)
                offset = 0.0f;
            else if(col == 5)
                offset = 0.75f;
            else if(col == 7)
                offset = 0.5f;

            vvx = fmod(vvx+offset,1);
            vvx /= 0.75f;

            if(odd(col))
            {
                if(vvx > 1.f)
                {
                    vvx = (vvx-1.f)*3.f;
                    uux = uux/3.f + 0.33333f;

                    currentThreadSegment = LONGWEFT;
                }
                else
                    currentThreadSegment = LONGWARP;
            }
            else
            {
                vvx = fmod((vv*12.f), 1.f);

                if(even(col))     // er... of course this is an even row.  why bother checking?
                {
                    currentThreadSegment = LONGWEFT;

                    // need to handle this one negative case...
                    int L = (row-1)%4;
                        if(L < 0)
                            L=4+L;

                    if(L == (col/2)%4)
                        uux = uux/3.f + 0.66666f;  // right side of horizontal ellipse

                    else if((row+2)%4 == (col/2)%4)
                        uux = uux/3.f;             // left side of horizontal ellipse

                    else
                        currentThreadSegment = SHORTWEFT;

                }
            }

            threadPattern[0] = LONGWARP;
            threadPattern[1] = LONGWEFT;
            threadPattern[2] = SHORTWEFT;
            threadTypeCount = 3;
            t_seg[LONGWARP].patternWeight  = 0.375f;      // 54 thread widths out of an available 144
            t_seg[LONGWEFT].patternWeight  = 0.375f;      // 54 thread widths out of an available 144
            t_seg[SHORTWEFT].patternWeight = 0.25f;       // 36 thread widths out of an available 144
        }

        // more BRDF filtering setup stuff
        //
        float uLength = sqrtf(m_dsdx*m_dsdx + m_dsdy*m_dsdy);
        float vLength = sqrtf(m_dtdx*m_dtdx + m_dtdy*m_dtdy);

        // get scaling factor for thread count multiplier
        float uTileLength = 1.0/m_thread_count_mult_u;
        float vTileLength = 1.0/m_thread_count_mult_v;

        // estimate of thread counts in u and v directions
        // for thread pattern tile
        float weftTileEstimate = uLength/uTileLength;
        float warpTileEstimate = vLength/vTileLength;

        // there are approx 12 threads across in a pattern,
        // a thread is 1/12th the width of the pattern
        float uThreadWidth = uTileLength/12.f;
        float vThreadWidth = vTileLength/12.f;

        // we need to handle the worst case, minimum thread width
        float threadWidth = std::min(uThreadWidth, vThreadWidth);

        // start LERPing to filtered multiple thread segment
        // BRDF at one half thread width, end at two thread
        // widths per sample width
        float minThreadWidthBRDF = threadWidth * 0.5f;
        float maxThreadWidthBRDF = threadWidth * 2.f;

        // start LERPing to BRDF at one quarter thread width,
        // end at one half thread width per sample width
        float minThreadWidthBTF  = threadWidth * 0.25f;
        maxThreadWidthBTF        = threadWidth * 0.5f;

        threadCoverageEstimate = std::max(uLength, vLength);

        // thread width is just what it sounds like: the width of a thread, not the length.
        // smoothstep to get 0->1 interpolant to LERP between single thread BRDF and filtered multi-thread BRDF
        //
        BRDF_interp = smoothstep(minThreadWidthBRDF, maxThreadWidthBRDF, threadCoverageEstimate);
        // this interpolant is used to blend between the BTF and BRDF
        BTF_interp  = smoothstep(minThreadWidthBTF, maxThreadWidthBTF, threadCoverageEstimate);

        Vec3 percentCoverage;
        Vec3 tmp;

        tmp.x = weftTileEstimate;
        tmp.y = warpTileEstimate;
        tmp.z = 0.f;

        percentCoverage = tmp.normalize();

        if(threadCoverageEstimate > minThreadWidthBRDF)
        {
            t_seg[LONGWARP].beamCoverage  = percentCoverage.y * M_SQRT2;
            t_seg[SHORTWARP].beamCoverage = percentCoverage.y * M_SQRT2;

            t_seg[LONGWEFT].beamCoverage  = percentCoverage.x * M_SQRT2;
            t_seg[SHORTWEFT].beamCoverage = percentCoverage.x * M_SQRT2;
        }

    }

    void print_on (std::ostream &out) const
    {
        out << "cloth (";
        out << "(" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "), ";

        out << m_s << ",";
        out << m_t << ",";

        out << m_dsdx << ",";
        out << m_dtdx << ",";

        out << m_dsdy << ",";
        out << m_dtdy << ",";
        out << m_area_scaled << ",";
        out << "(" << m_dPdu[0] << ", " << m_dPdu[1] << ", " << m_dPdu[2] << "), ";

        out << m_diff_warp_col  << ", ";
        out << m_diff_weft_col  << ", ";

        out << m_spec_warp_col  << ", ";
        out << m_spec_weft_col  << ", ";

        out << m_fresnel_warp << ", ";
        out << m_fresnel_weft << ", ";

        out << m_spread_x_mult  << ", ";
        out << m_spread_y_mult  << ", ";

        out << m_pattern        << ", ";
        out << m_pattern_angle  << ", ";

        out << m_warp_width_scale     << ", ";
        out << m_weft_width_scale     << ", ";

        out << m_thread_count_mult_u     << ", ";
        out << m_thread_count_mult_v     << ", ";

        out << ")";
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float &pdf) const
    {
        float cosNI = m_N.dot(omega_in);
        float cosNO = m_N.dot(omega_out);

        if (!(cosNI > 0 && cosNO > 0))
            return Color3 (0, 0, 0);


        // whittle this down to a 3x3
        Matrix44 M;
        Matrix44 M_inv;

        M.makeIdentity();

        Vec3 rect[4];


        Vec3 H = omega_in + omega_out;
        H.normalize();
        Point2 H2 = H_projected(H, m_N, m_dPdu);


        Color3 diff[4];
        // per thread segment diffuse
        float cos_pi = cosNI * (float) M_1_PI;        
        diff[LONGWARP]  = cos_pi * t_seg[LONGWARP].diffCol;
        diff[SHORTWARP] = diff[LONGWARP];
        diff[LONGWEFT]  = cos_pi * t_seg[LONGWEFT].diffCol;
        diff[SHORTWEFT] = diff[LONGWEFT];
        
        // per thread segment fresnel 
        float F[4];
        F[LONGWARP] = schlick_fresnel(cosNO, t_seg[LONGWARP].R0);
        F[SHORTWARP] = F[LONGWARP];

        if(m_fresnel_warp == m_fresnel_weft)
        {
            F[LONGWEFT]  = F[LONGWARP];
            F[SHORTWEFT] = F[LONGWARP];
        }
        else
        {
            F[LONGWEFT]  = schlick_fresnel(cosNO, t_seg[LONGWEFT].R0);
            F[SHORTWEFT] = F[LONGWEFT];
        }

        float I, G;
        float AE    = 0.f;  
        float AE_st = 0.f;  // specular area, current single thread segment
        float AE_f  = 0.f;  // specular area, filtered aggregate

        Color3 DE_st (0, 0, 0);  // diffuse result * col, current single thread segment
        Color3 DE_f  (0, 0, 0);  // diffuse result * col, filtered aggregate 

        float btf, brdf_spec;
        Point2 BTF_center, BTF_semimajor;

        // stuff for brdf_spec derived from btf "thread segment" highlight ellipse area here...
        //
        for(int i=0; i<threadTypeCount; i++)
        {

            rect[0].x=0;                             rect[0].y=0;                             rect[0].z = 0;
            rect[1].x=0;                             rect[1].y=1*t_seg[threadPattern[i]].Sy;  rect[1].z = 0;
            rect[2].x=1*t_seg[threadPattern[i]].Sx;  rect[2].y=1*t_seg[threadPattern[i]].Sy;  rect[2].z = 0;
            rect[3].x=1*t_seg[threadPattern[i]].Sx;  rect[3].y=0;                             rect[3].z = 0;

            M.x[0][0] = t_seg[threadPattern[i]].semimajor.x;
            M.x[0][1] = t_seg[threadPattern[i]].semimajor.y;
            M.x[1][0] = t_seg[threadPattern[i]].semiminor.x;
            M.x[1][1] = t_seg[threadPattern[i]].semiminor.y;

            // get ellipse center for 2D half vector
            Point2 center = ellipse_center(t_seg[threadPattern[i]].Sx, t_seg[threadPattern[i]].Sy,
                                           t_seg[threadPattern[i]].Kx, t_seg[threadPattern[i]].Ky, H2);

            M.x[0][3] = center.x;
            M.x[1][3] = center.y;

            M_inv = M.inverse();

            Vec3 rect_xfmd[4];

            for (int j = 0; j < 4; j++)
            {
               rect_xfmd[j].x = M_inv.x[0][0]*rect[j].x+
                                M_inv.x[1][0]*rect[j].y+
                                M_inv.x[2][0]*rect[j].z+
                                M_inv.x[3][0];
               rect_xfmd[j].y = M_inv.x[0][1]*rect[j].x+
                                M_inv.x[1][1]*rect[j].y+
                                M_inv.x[2][1]*rect[j].z+
                                M_inv.x[3][1];
               rect_xfmd[j].z = M_inv.x[0][2]*rect[j].x+
                                M_inv.x[1][2]*rect[j].y+
                                M_inv.x[2][2]*rect[j].z+
                                M_inv.x[3][2];
            }

            Vec3 highlight_xfmd;
            Vec3 CENTER;

            CENTER.x = center.x; CENTER.y = center.y; CENTER.z = 0.f;

            highlight_xfmd.x = M_inv.x[0][0]*CENTER.x+
                               M_inv.x[1][0]*CENTER.y+
                               M_inv.x[2][0]*CENTER.z+
                               M_inv.x[3][0];
            highlight_xfmd.y = M_inv.x[0][1]*CENTER.x+
                               M_inv.x[1][1]*CENTER.y+
                               M_inv.x[2][1]*CENTER.z+
                               M_inv.x[3][1];
            highlight_xfmd.z = M_inv.x[0][2]*CENTER.x+
                               M_inv.x[1][2]*CENTER.y+
                               M_inv.x[2][2]*CENTER.z+
                               M_inv.x[3][2];

            // It's far easier to tell if we're inside or outside of the untransformed rectangle
            // instead of testing to see if the circle center is inside the transformed rectangle, which
            // is generally speaking, a parallelogram.
            // SO, pass boolean "isOutside" as hint to computeAC()
            bool isOutside = (center.x < rect[0].x || center.y < rect[0].y || center.x > rect[2].x || center.y > rect[2].y);

            float AC = compute_AC(highlight_xfmd, rect_xfmd, isOutside);

            // this can be made a 3x3 instead.
            float det = fabs(det4x4(M));

            // ellipse segment area
            // apply per-thread type fresnel here
            AE = (det*AC)/t_seg[threadPattern[i]].AR  * F[threadPattern[i]];

            //  grab the current single thread ellipse segment area in AE_st
            //  - to be used in LERP with AE_f (for when feature scale is under Nyquist limit (hopefully))
            //
            if(threadPattern[i]== currentThreadSegment)
            {
                AE_st = AE;
                DE_st = diff[threadPattern[i]];
                BTF_center = center;
                BTF_semimajor = t_seg[threadPattern[i]].semimajor;
            }

            // filtered AE
            //
            float weight = t_seg[threadPattern[i]].patternWeight * t_seg[threadPattern[i]].beamCoverage;
            AE_f += AE * weight;
            DE_f += diff[threadPattern[i]] * weight;

        }  // end for threadTypeCount
        // blend between single thread and antialiased, multiple thread segment BRDF
        //
        AE_f = Imath::lerp(AE_st, AE_f, BRDF_interp); 
        DE_f = Imath::lerp(DE_st, DE_f, BRDF_interp);

        // microfacet geometric term
        G = computeG_Smith(m_N, H, omega_in, omega_out);

        // this bit happens when we're close enough to resolve the BTF
        if(threadCoverageEstimate < maxThreadWidthBTF)
        {
            //get foci of ellipse
            Point2 F1, F2;
            ellipse_foci(BTF_semimajor, t_seg[currentThreadSegment].eccentricity, BTF_center, &F1, &F2);

            // check to see if current shaded point (uu,vv) is inside the ellipse
            I = inside_ellipse(F1, F2, uux*t_seg[currentThreadSegment].Sx, vvx*t_seg[currentThreadSegment].Sy,
                               t_seg[currentThreadSegment].alpha, m_area_scaled);

            // 'I' replaces slope distribution term 'D' found in generalized cookTorrance microfacet model
            // Unlike 'D' in the canonical microfacet model which is 1.0/steradians, our term is 0->1 by design.
            btf = I*G*F[currentThreadSegment];
            btf /= cosNO;
        }

        brdf_spec = AE_f*G; // F is already figured in AE_f, above...
        brdf_spec /= cosNO;

        Color3 out;
        float filtered;
        filtered = Imath::lerp(btf, brdf_spec, BTF_interp);
        

        //printf("BTF_interp: %f\n", BTF_interp);

        out = filtered * t_seg[currentThreadSegment].specCol;
        out += DE_f;

        pdf = 0.f;

        return Color3 (out[0], out[1], out[2]);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float normal_size, float &pdf) const
    {
        return Color3 (0.f, 0.f, 0.f);
    }

    ustring sample (const Vec3 &Ng,
                    const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                    float randu, float randv,
                    Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                    float &pdf, Color3 &eval) const
    {
        Vec3 Ngf, Nf;
        if (faceforward (omega_out, Ng, m_N, Ngf, Nf)) {

                pdf = 0;
        }
        return Labels::REFLECT;
    }

};

DECLOP (OP_cloth)
{
    closure_op_guts<ClothClosure, 24> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}

}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif

