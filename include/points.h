#ifndef POINTS_H_
#define POINTS_H_

#include <array>
#include <vector>
#include <algorithm>
#include <functional>
#include <fstream>
#include <assert.h>
#include "mpienv.h"

template <int D>
struct FloatingPointTraits
{
    static inline constexpr int dimension = D;

    using Real = float;
    using Index = int64_t;

    using Point = std::array<float, D>;
    using PointRecord = std::array<char, sizeof(int) + sizeof(Point)>;
    using PointVector = std::vector<Point>;
    using IndexVector = std::vector<Index>;
    using Comm = MPIEnv::Comm;

    static void unpack_point(const PointRecord& record, Point& p);
    static void read_fvecs(PointVector& points, const char *fname);
    static void read_fvecs(PointVector& mypoints, Index& myoffset, Index& totsize, const char *fname);
};

template <class PointTraits_>
struct L2Distance;

#include "points.hpp"

#endif
