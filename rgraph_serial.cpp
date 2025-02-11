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

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "usage: %s <points>\n", argv[0]);
        return 1;
    }

    const char *fname = argv[1];

    double t;

    t = -omp_get_wtime();
    PointTraits::read_fvecs(points, fname); num_points = points.size();
    t += omp_get_wtime();

    fmt::print("[time={:.3f}] read {} points from file '{}'\n", t, num_points, fname);

    return 0;
}
