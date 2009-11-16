/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
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
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cmath>

#include "oslops.h"
#include "oslexec_pvt.h"

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {

class RefractionClosure : public BSDFClosure {
    Vec3  m_N;     // shading normal
    float m_eta;   // ratio of indices of refraction (inside / outside)
public:
    CLOSURE_CTOR (RefractionClosure)
    {
        CLOSURE_FETCH_ARG (m_N  , 1);
        CLOSURE_FETCH_ARG (m_eta, 2);
    }

    void print_on (std::ostream &out) const {
        out << "refraction (";
        out << "(" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "), ";
        out << m_eta;
        out << ")";
    }

    bool get_cone(const Vec3 &omega_out, Vec3 &axis, float &angle) const
    {
        // does not need to be integrated directly
        return false;
    }

    Color3 eval (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &omega_in, Labels &labels) const
    {
        // should never be called - because get_cone is empty
        return Color3 (0.0f, 0.0f, 0.0f);
    }

    void sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval, Labels &labels) const
    {
        labels.set (Labels::SURFACE, Labels::TRANSMIT, Labels::SINGULAR);
        Vec3 R, dRdx, dRdy;
        Vec3 T, dTdx, dTdy;
        float Ft = 1 - fresnel_dielectric(m_eta, m_N,
                                          omega_out, domega_out_dx, domega_out_dy,
                                          R, dRdx, dRdy,
                                          T, dTdx, dTdy);
        if (Ft > 0) {
            pdf = 1;
            eval.setValue(Ft, Ft, Ft);
            omega_in = T;
            domega_in_dx = dTdx;
            domega_in_dy = dTdy;
        }
    }

    float pdf (const Vec3 &Ng,
               const Vec3 &omega_out, const Vec3 &omega_in) const
    {
        // the pdf for an arbitrary direction is 0 because only a single
        // direction is actually possible
        return 0;
    }
};



class DielectricClosure : public BSDFClosure {
    Vec3  m_N;     // shading normal
    float m_eta;   // ratio of indices of refraction (inside / outside)
public:
    CLOSURE_CTOR (DielectricClosure)
    {
        CLOSURE_FETCH_ARG (m_N  , 1);
        CLOSURE_FETCH_ARG (m_eta, 2);
    }

    void print_on (std::ostream &out) const {
        out << "dielectric (";
        out << "(" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "), ";
        out << m_eta;
        out << ")";
    }

    bool get_cone(const Vec3 &omega_out, Vec3 &axis, float &angle) const
    {
        // does not need to be integrated directly
        return false;
    }

    Color3 eval (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &omega_in, Labels &labels) const
    {
        // should never be called - because get_cone is empty
        return Color3 (0.0f, 0.0f, 0.0f);
    }

    void sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval, Labels &labels) const
    {
        Vec3 R, dRdx, dRdy;
        Vec3 T, dTdx, dTdy;
        // randomly choose between reflection/refraction
        float Fr = fresnel_dielectric(m_eta, m_N,
                                      omega_out, domega_out_dx, domega_out_dy,
                                      R, dRdx, dRdy,
                                      T, dTdx, dTdy);
        if (randu < Fr) {
            eval.setValue(Fr, Fr, Fr);
            pdf = Fr;
            omega_in = R;
            domega_in_dx = dRdx;
            domega_in_dy = dRdy;
            labels.set (Labels::SURFACE, Labels::REFLECT, Labels::SINGULAR);
        } else {
            pdf = 1 - Fr;
            eval.setValue(pdf, pdf, pdf);
            omega_in = T;
            domega_in_dx = dTdx;
            domega_in_dy = dTdy;
            labels.set (Labels::SURFACE, Labels::TRANSMIT, Labels::SINGULAR);
        }
    }

    float pdf (const Vec3 &Ng,
               const Vec3 &omega_out, const Vec3 &omega_in) const
    {
        // the pdf for an arbitrary direction is 0 because only a single
        // direction is actually possible
        return 0;
    }
};



DECLOP (OP_refraction)
{
    closure_op_guts<RefractionClosure> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}


DECLOP (OP_dielectric)
{
    closure_op_guts<DielectricClosure> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}


}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
