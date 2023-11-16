#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include "iso3dfd.h"
#include "dpc_common.hpp"

using namespace amrex;
// amrex
// i is the slowest moving index.
// j is the second slowest moving index.
// k is the fastest moving index.
//  SYCL
// int i = it.get_global_id(0) is the slowest moving index
// int j = it.get_global_id(1) is the second slowest moving index
// int k = it.get_global_id(2) is teh fastest moving index

void Iso3dfdIterationGlobal(int i, int j, int k, Real *next, Real *prev,
                            Real *vel, const Real *coeff, int nx, int nxy,
                            int bx, int by, int z_offset, int full_end_z)
{

    auto begin_z = i * z_offset + kHalfLength;
    auto end_z = begin_z + z_offset;
    if (end_z > full_end_z)
        end_z = full_end_z;
    int gid = (k + bx) + (j + by) + (begin_z * nxy);

    Real front[kHalfLength + 1];
    Real back[kHalfLength];
    Real c[kHalfLength + 1];

    for (auto iter = 0; iter <= kHalfLength; iter++)
    {
        front[iter] = prev[gid + iter * nxy];
    }

    float value = c[0] * front[0];
#pragma unroll(kHalfLength)
    for (auto iter = 1; iter <= kHalfLength; iter++)
    {
        value += c[iter] * (front[iter] + back[iter - 1] + prev[gid + iter] + prev[gid - iter] + prev[gid + iter * nx] + prev[gid - iter * nx]);
    }
    next[gid] = 2.0f * front[0] - next[gid] + value * vel[gid];

    gid += nxy;
    begin_z++;

    while (begin_z < end_z)
    {
        // Input data in front and back are shifted to discard the
        // oldest value and read one new value.
        for (auto iter = kHalfLength - 1; iter > 0; iter--)
        {
            back[iter] = back[iter - 1];
        }
        back[0] = front[0];

        for (auto iter = 0; iter < kHalfLength; iter++)
        {
            front[iter] = front[iter + 1];
        }

        // Only one new data-point read from global memory
        // in z-dimension (depth)
        front[kHalfLength] = prev[gid + kHalfLength * nxy];

        // Stencil code to update grid point at position given by global id (gid)
        float value = c[0] * front[0];
#pragma unroll(kHalfLength)
        for (auto iter = 1; iter <= kHalfLength; iter++)
        {
            value += c[iter] * (front[iter] + back[iter - 1] + prev[gid + iter] +
                                prev[gid - iter] + prev[gid + iter * nx] +
                                prev[gid - iter * nx]);
        }

        next[gid] = 2.0f * front[0] - next[gid] + value * vel[gid];

        gid += nxy;
        begin_z++;
    }
}


void PrintStats(double time, size_t n1, size_t n2, size_t n3,
                unsigned int nIterations)
{
    float throughput_mpoints = 0.0f, mflops = 0.0f, normalized_time = 0.0f;
    double mbytes = 0.0f;

    normalized_time = (double)time / nIterations;
    throughput_mpoints = ((n1 - 2 * kHalfLength) * (n2 - 2 * kHalfLength) *
                          (n3 - 2 * kHalfLength)) /
                         (normalized_time * 1e3f);
    mflops = (7.0f * kHalfLength + 5.0f) * throughput_mpoints;
    mbytes = 12.0f * throughput_mpoints;

    std::cout << "--------------------------------------\n";
    std::cout << "time         : " << time / 1e3f << " secs\n";
    std::cout << "throughput   : " << throughput_mpoints << " Mpts/s\n";
    std::cout << "flops        : " << mflops / 1e3f << " GFlops\n";
    std::cout << "bytes        : " << mbytes / 1e3f << " GBytes/s\n";
    std::cout << "\n--------------------------------------\n";
    std::cout << "\n--------------------------------------\n";
}

void Iso3dfdDevice(const Array4<Real> &prev_arr, const Array4<Real> &next_arr, const Array4<Real> &vel_arr, Real *coeff, Box &domain, const int n1, const int n2, const int n3,
                   int n1_block, int n2_block, int n3_block, int end_z, const unsigned int nreps)
{
    std::cout << "Inside Iso3dfdDevice" << std::endl;
    int nx = n1;
    int nxy = n1 * n2;
    int bx = kHalfLength;
    int by = kHalfLength;
    int local_range = n2_block * n1_block;
    int size1 = (n3 - 2 * kHalfLength) / n3_block;
    int size2 = n2 - 2 * kHalfLength;
    int size3 = n1 - 2 * kHalfLength;
     amrex::IntVect dom_lo(0, 0, 0);
    amrex::IntVect dom_hi(size1,size2, size3);
    //Box domain1 = (IntVect(0,0,0), IntVect(size1,size2,size3));
    Box domain1(dom_lo, dom_hi);
    
    for (auto i = 0; i < nreps; i += 1)
    {
        if (i % 2 == 0)
        {

            amrex::ParallelFor(domain1,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   // int idx = *it;
                      
                                   Iso3dfdIterationGlobal(i, j, k, next_arr.dataPtr(), prev_arr.dataPtr(), vel_arr.dataPtr(),
                                                          coeff, nx, nxy, bx, by,
                                                          n3_block, end_z);
                               });
        }
        else
        {
            amrex::ParallelFor(domain1,
                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                               {
                                   // int idx = *it;
                                   Iso3dfdIterationGlobal(i, j, k, prev_arr.dataPtr(), next_arr.dataPtr(), vel_arr.dataPtr(),
                                                          coeff, nx, nxy, bx, by,
                                                          n3_block, end_z);
                               });
        }
    }
}

void Initialize_dataamrex(MultiFab &prev_base, MultiFab &next_base, MultiFab &vel_base, int n1, int n2, int n3)
{

    prev_base.setVal(0.0);
    next_base.setVal(0.0);
    vel_base.setVal(2250000.0 * dt * dt); // Integration of the v*v and dt*dt
    float val = 1.f;
    for (int s = 5; s >= 0; s--)
    {
        for (int i = n3 / 2 - s; i < n3 / 2 + s; i++)
        {
            for (int j = n2 / 4 - s; j < n2 / 4 + s; j++)
            {
                IntVect offset(i, j, n1 / 4 - s);
                IntVect size(1, 1, 2 * s + 1);
                Box region(offset, offset + size - IntVect::TheUnitVector());
                prev_base.setVal(val, region, 0, 1);
            }
        }
        val *= 10;
    }
}
void main_main(int n1, int n2, int n3, int num_iterations, int n1_block, int n2_block, int n3_block, float *temp)
{

    {
        ParmParse pp;

        // std::cout << n1 << std::endl;
        pp.query("n1", n1);
        // n1 = n1 + (2 * kHalfLength);
        pp.query("n2", n2);
        // n2 = n2 + (2 * kHalfLength);
        pp.query("n3", n3);
        // n3 = n3 + (2 * kHalfLength);
        pp.query("n1_block", n1_block);
        pp.query("n2_block", n2_block);
        pp.query("n3_block", n3_block);

        pp.query("num_iterations", num_iterations);
    }
    size_t nsize = n1 * n2 * n3;
    std::cout << n1 << " " << n2 << " " << n3 << std::endl;
    BoxArray ba;
    Geometry geom;
    amrex::IntVect dom_lo(0, 0, 0);
    amrex::IntVect dom_hi(n1, n2, n3);
    // Make a single box that is the entire domain
    Box domain(dom_lo, dom_hi);
    ba.define(domain);
    // Initialize the boxarray "ba" from the single box "domain"
    // ba.define(domain);
    // Do i care about realbox if I am just constructing the 3d grid?
    RealBox real_box({AMREX_D_DECL(-1.0, -1.0, -1.0)},
                     {AMREX_D_DECL(1.0, 1.0, 1.0)});
    geom.define(domain, &real_box, CoordSys::cartesian);
    Real time = 0.0;
    DistributionMapping dm(ba);
    // Nghost = number of ghost cells for each array
    int Nghost = 0;
    // Ncomp = number of components for each array
    int Ncomp = 1;
    // MultiFabs used to update the wavefield.
    MultiFab prev_base(ba, dm, Ncomp, Nghost);
    MultiFab next_base(ba, dm, Ncomp, Nghost);
    // Array to store wave velocity
    MultiFab vel_base(ba, dm, Ncomp, Nghost);
    // Array to store results for comparison
    // MultiFab temp(ba, dm, Ncomp, Nghost);

    Real coeff[kHalfLength + 1] = {-3.0548446, +1.7777778, -3.1111111e-1,
                                   +7.572087e-2, -1.76767677e-2, +3.480962e-3,
                                   -5.180005e-4, +5.074287e-5, -2.42812e-6};

    // Apply the DX DY and DZ to coefficients
    coeff[0] = (3.0 * coeff[0]) / (dxyz * dxyz);
    for (int i = 1; i <= kHalfLength; i++)
    {
        coeff[i] = coeff[i] / (dxyz * dxyz);
    }

    std::cout << "Grid Sizes: " << n1 - 2 * kHalfLength << " "
              << n2 - 2 * kHalfLength << " " << n3 - 2 * kHalfLength << "\n";
    std::cout << "Memory Usage: " << ((3 * nsize * sizeof(float)) / (1024 * 1024))
              << " MB\n";

    Initialize_dataamrex(prev_base, next_base, vel_base, n1, n2, n3);

    dpc_common::TimeInterval t_amr;
    for (MFIter mfi(next_base); mfi.isValid(); ++mfi)
    {
        // const Box& bx = mfi.tilebox();
        const Array4<Real> &next_arr = next_base.array(mfi);
        const Array4<Real> &prev_arr = prev_base.array(mfi); // Use const_array for prev since it's read-only
        const Array4<Real> &vel_arr = vel_base.array(mfi);   // Use const_array for vel since it's read-only
       
        Iso3dfdDevice(next_arr, prev_arr, vel_arr, coeff, domain, n1, n2, n3, n1_block, n2_block, n3_block, n3 - kHalfLength, num_iterations);
       
        // t = amrex::second()-t0;
    }
     PrintStats(t_amr.Elapsed() * 1e3, n1, n2, n3, num_iterations);
    /*
    //Real* ptr = mf.dataPtr();
    temp = new float[nsize];
    if (num_iterations % 2)
    {
        // Copy data from next_base to temp
        next_base.copyTo(temp, 0);
    }
    else
    {
        // Copy data from prev_base to temp
        prev_base.copyTo(temp, 0);
    }*/
}



int main(int argc, char *argv[])

{

    int n1 = 256;
    int n2 = 256;
    int n3 = 256;
    int num_iterations = 10;
    size_t nsize;
    int n1_block = 32;
    int n2_block = 8;
    int n3_block = 64;
    n1 = n1 + (2 * kHalfLength);
    n2 = n2 + (2 * kHalfLength);
    n3 = n3 + (2 * kHalfLength);
    float *temp;
    float *temp1;

    {
        amrex::Initialize(argc, argv);
        main_main(n1, n2, n3, num_iterations, n1_block, n2_block, n3_block, temp);
        amrex::Finalize();
    }

    return 0;
}
