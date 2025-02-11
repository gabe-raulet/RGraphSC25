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

using Index = int64_t;
using Real = float;

using PointTraits = FloatingPointTraits<DIM_SIZE>;
using Distance = L2Distance<PointTraits>;
using Point = typename PointTraits::Point;

using PointVector = std::vector<Point>;
using IndexVector = std::vector<Index>;
using RealVector = std::vector<Real>;

Distance distance;

PointVector points;
Index num_points;
Index num_sites;
Real epsilon;
json results;
int nthreads = 1;

Real covering_factor = 2.0;
Index leaf_size = 10;

int main(int argc, char *argv[])
{
    #pragma omp parallel
    nthreads = omp_get_num_threads();

    if (argc != 5)
    {
        fprintf(stderr, "usage: %s <points> <num_sites> <epsilon> <results>\n", argv[0]);
        return 1;
    }

    const char *points_fname = argv[1];
    const char *results_fname = argv[4];
    num_sites = atoi(argv[2]);
    epsilon = atof(argv[3]);

    double t;

    t = -omp_get_wtime();
    PointTraits::read_fvecs(points, points_fname); num_points = points.size();
    t += omp_get_wtime();

    fmt::print("[time={:.3f}] read {} points from file '{}'\n", t, num_points, points_fname);

    results["dataset"] = points_fname;
    results["num_points"] = num_points;
    results["dim"] = DIM_SIZE;
    results["num_threads"] = nthreads;
    results["num_sites"] = num_sites;
    results["epsilon"] = epsilon;

    std::ofstream f(results_fname);
    f << std::setw(4) << results << std::endl;
    f.close();

    return 0;
}
