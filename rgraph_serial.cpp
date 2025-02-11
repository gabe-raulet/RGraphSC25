#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "json.hpp"
#include <fstream>
using json = nlohmann::json;

#include "fmt/core.h"
#include "points.h"
#include <iostream>
#include <vector>
#include <limits>
#include <random>
#include <algorithm>
#include <unistd.h>
#include <omp.h>

#ifndef DIM_SIZE
#error "DIM_SIZE is undefined"
#endif

using PointTraits = FloatingPointTraits<DIM_SIZE>;
using PointVector = PointTraits::PointVector;

PointVector points;
size_t num_points;
json results;
int nthreads = 1;

int main(int argc, char *argv[])
{
    #pragma omp parallel
    nthreads = omp_get_num_threads();

    if (argc != 3)
    {
        fprintf(stderr, "usage: %s <points> <results>\n", argv[0]);
        return 1;
    }

    const char *points_fname = argv[1];
    const char *results_fname = argv[2];

    double t;

    t = -omp_get_wtime();
    PointTraits::read_fvecs(points, points_fname); num_points = points.size();
    t += omp_get_wtime();

    fmt::print("[time={:.3f}] read {} points from file '{}'\n", t, num_points, points_fname);

    results["dataset"] = points_fname;
    results["num_points"] = num_points;
    results["dim"] = DIM_SIZE;
    results["num_threads"] = nthreads;

    std::ofstream f(results_fname);
    f << std::setw(4) << results << std::endl;
    f.close();

    return 0;
}
