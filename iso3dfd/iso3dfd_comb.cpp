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

/*void Iso3dfdIteration(float* ptr_next_base, float* ptr_prev_base,
                      float* ptr_vel_base, float* coeff, const size_t n1,
                      const size_t n2, const size_t n3, const size_t n1_block,
                      const size_t n2_block, const size_t n3_block) {
  size_t dimn1n2 = n1 * n2;
  size_t n3End = n3 - kHalfLength;
  size_t n2End = n2 - kHalfLength;
  size_t n1End = n1 - kHalfLength;

#pragma omp parallel default(shared)
#pragma omp for schedule(static) collapse(3)
  for (size_t bz = kHalfLength; bz < n3End;
       bz += n3_block) {  // start of cache blocking
    for (size_t by = kHalfLength; by < n2End; by += n2_block) {
      for (size_t bx = kHalfLength; bx < n1End; bx += n1_block) {
        int izEnd = std::min(bz + n3_block, n3End);
        int iyEnd = std::min(by + n2_block, n2End);
        int ixEnd = std::min(n1_block, n1End - bx);
        for (size_t iz = bz; iz < izEnd; iz++) {  // start of inner iterations
          for (size_t iy = by; iy < iyEnd; iy++) {
            float* ptr_next = ptr_next_base + iz * dimn1n2 + iy * n1 + bx;
            float* ptr_prev = ptr_prev_base + iz * dimn1n2 + iy * n1 + bx;
            float* ptr_vel = ptr_vel_base + iz * dimn1n2 + iy * n1 + bx;
#pragma omp simd
            for (size_t ix = 0; ix < ixEnd; ix++) {
              float value = 0.0;
              value += ptr_prev[ix] * coeff[0];
#pragma unroll(kHalfLength)
              for (unsigned int ir = 1; ir <= kHalfLength; ir++) {
                value += coeff[ir] *
                         ((ptr_prev[ix + ir] + ptr_prev[ix - ir]) +
                          (ptr_prev[ix + ir * n1] + ptr_prev[ix - ir * n1]) +
                          (ptr_prev[ix + ir * dimn1n2] +
                           ptr_prev[ix - ir * dimn1n2]));
              }
              ptr_next[ix] =
                  2.0f * ptr_prev[ix] - ptr_next[ix] + value * ptr_vel[ix];
            }
          }
        }  // end of inner iterations
      }
    }
  }  // end of cache blocking
}*/

/*void Iso3dfd(float* ptr_next, float* ptr_prev, float* ptr_vel, float* coeff,
             const size_t n1, const size_t n2, const size_t n3,
             const unsigned int nreps, const size_t n1_block,
             const size_t n2_block, const size_t n3_block) {
  for (unsigned int it = 0; it < nreps; it += 1) {
    Iso3dfdIteration(ptr_next, ptr_prev, ptr_vel, coeff, n1, n2, n3, n1_block,
                     n2_block, n3_block);

    // here's where boundary conditions and halo exchanges happen
    // Swap previous & next between iterations
    it++;
    if (it < nreps)
      Iso3dfdIteration(ptr_prev, ptr_next, ptr_vel, coeff, n1, n2, n3, n1_block,
                       n2_block, n3_block);
  }  // time loop
}*/

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

void Initialize_amrex(MultiFab &prev_base, MultiFab &next_base, MultiFab &vel_base, int n1, int n2, int n3)
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
    int Ncomp = 0;
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

    Initialize_amrex(prev_base, next_base, vel_base, n1, n2, n3);

    for (MFIter mfi(next_base); mfi.isValid(); ++mfi)
    {
        // const Box& bx = mfi.tilebox();
        const Array4<Real> &next_arr = next_base.array(mfi);
        const Array4<Real> &prev_arr = prev_base.array(mfi); // Use const_array for prev since it's read-only
        const Array4<Real> &vel_arr = vel_base.array(mfi);   // Use const_array for vel since it's read-only
        dpc_common::TimeInterval t_amr;
        Iso3dfdDevice(next_arr, prev_arr, vel_arr, coeff, domain, n1, n2, n3, n1_block, n2_block, n3_block, n3 - kHalfLength, num_iterations);
        PrintStats(t_amr.Elapsed() * 1e3, n1, n2, n3, num_iterations);
        // t = amrex::second()-t0;
    }

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

/*void Initialize(float* ptr_prev, float* ptr_next, float* ptr_vel, size_t n1,
                size_t n2, size_t n3) {
  std::cout << "Initializing ... \n";
  size_t dim2 = n2 * n1;

  for (size_t i = 0; i < n3; i++) {
    for (size_t j = 0; j < n2; j++) {
      size_t offset = i * dim2 + j * n1;
#pragma omp simd
      for (int k = 0; k < n1; k++) {
        ptr_prev[offset + k] = 0.0f;
        ptr_next[offset + k] = 0.0f;
        ptr_vel[offset + k] =
            2250000.0f * dt * dt;  // Integration of the v*v and dt*dt
      }
    }
  }
  // Add a source to initial wavefield as an initial condition
  float val = 1.f;
  for (int s = 5; s >= 0; s--) {
    for (int i = n3 / 2 - s; i < n3 / 2 + s; i++) {
      for (int j = n2 / 4 - s; j < n2 / 4 + s; j++) {
        size_t offset = i * dim2 + j * n1;
        for (int k = n1 / 4 - s; k < n1 / 4 + s; k++) {
          ptr_prev[offset + k] = val;
        }
      }
    }
    val *= 10;
  }
}*/

/*void main_nonamrex(int n1, int n2,int n3, int num_iterations,int n1_block, int n2_block, int n3_block ){

    // Arrays used to update the wavefield
    float* prev_base;
    float* next_base;
    // Array to store wave velocity
    float* vel_base;
    // Array to store results for comparison
    float* temp;

    size_t nsize = n1 * n2 * n3;

    prev_base = new float[nsize];
    next_base = new float[nsize];
    vel_base = new float[nsize];

    float coeff[kHalfLength + 1] = {-3.0548446,   +1.7777778,     -3.1111111e-1,
                                    +7.572087e-2, -1.76767677e-2, +3.480962e-3,
                                    -5.180005e-4, +5.074287e-5,   -2.42812e-6};

    coeff[0] = (3.0f * coeff[0]) / (dxyz * dxyz);
    for (int i = 1; i <= kHalfLength; i++) {
            coeff[i] = coeff[i] / (dxyz * dxyz);
    }

    std::cout << "Grid Sizes: " << n1 - 2 * kHalfLength << " "
            << n2 - 2 * kHalfLength << " " << n3 - 2 * kHalfLength << "\n";
    std::cout << "Memory Usage: " << ((3 * nsize * sizeof(float)) / (1024 * 1024))
            << " MB\n";

    Initialize(prev_base, next_base, vel_base, n1, n2, n3);


    dpc_common::TimeInterval t_ser;
    // Invoke the driver function to perform 3D wave propogation
    // using OpenMP/Serial version
    Iso3dfd(next_base, prev_base, vel_base, coeff, n1, n2, n3, num_iterations,
            n1_block, n2_block, n3_block);

    PrintStats(t_ser.Elapsed() * 1e3, n1, n2, n3, num_iterations);

}*/

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

    /*std::cout << "Using non AMReX version" << "\n";
     main_nonamrex(n1,n2,n3,num_iterations,n1_block, n2_block, n3_block, temp1);*/
    return 0;
}
