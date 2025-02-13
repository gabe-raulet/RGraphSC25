#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "json.hpp"
#include <fstream>
using json = nlohmann::json;

#include "fmt/core.h"
#include "points.h"
#include "ctree.h"
#include <unordered_set>
#include <unordered_map>
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

using IndexMap = std::unordered_map<Index, Index>;
using IndexSet = std::unordered_set<Index>;
using IndexVectorMap = std::unordered_map<Index, IndexVector>;
using IndexVectorVector = std::vector<IndexVector>;

using Tree = CoverTree<PointTraits, Distance>;
using TreeMap = std::unordered_map<Index, Tree>;

const char *points_fname;
const char *results_fname;

PointVector points;
Index num_points;
Index num_sites; /* number of Voronoi sites */
Real epsilon; /* graph radius */
json results;
int nthreads = 1;

Real covering_factor = 2.0;
Index leaf_size = 10;

Distance distance;

void build_greedy_net(); /* build Voronoi diagram */
void build_replication_tree(); /* build replication tree on Voronoi sites */
void compute_ghost_points(); /* compute ghost points for each Voronoi cell */
void build_ghost_trees(); /* build cover trees on Voronoi cells plus ghost points */
void build_epsilon_graph(); /* build epsilon graph */

Index range_query(IndexVector& neighbors, Point query, Real radius); /* query replication tree for intersecting Voronoi sites */

IndexVector net; /* indices of Voronoi sites */
IndexVector cells; /* Voronoi cell site for each point */
RealVector dists; /* distance of each point to its Voronoi site */
Real net_sep; /* minimum separation of net sites */

std::vector<IndexVectorMap> reptree; /* replication tree */
Index minlevel, maxlevel, root; /* minimum/maximum level of replication tree, and root point id */

TreeMap trees; /* maps Voronoi sites to their "ghost" trees */
IndexVectorMap treeids, cellids; /* maps net sites to point+ghost_point sets (treeids) and just point sets (cellids) */

IndexVectorVector graph; /* epsilon graph */
double total_time = 0;

int main(int argc, char *argv[])
{
    #pragma omp parallel
    nthreads = omp_get_num_threads();

    auto usage = [&] (int err)
    {
        fprintf(stderr, "Usage: %s [options] <points> <num_sites> <epsilon>\n", argv[0]);
        fprintf(stderr, "Options: -c FLOAT  covering factor [%.2f]\n", covering_factor);
        fprintf(stderr, "         -l INT    leaf size [%lu]\n", (size_t)leaf_size);
        fprintf(stderr, "         -o FILE   output results json\n");
        fprintf(stderr, "         -h        help message\n");
        exit(err);
    };

    int c;
    while ((c = getopt(argc, argv, "c:l:o:h")) >= 0)
    {
        if      (c == 'c') covering_factor = atof(optarg);
        else if (c == 'l') leaf_size = atoi(optarg);
        else if (c == 'o') results_fname = optarg;
        else if (c == 'h') usage(0);
    }

    if (argc - optind < 3)
    {
        fmt::print(stderr, "[err::{}] missing argument(s)\n", __func__);
        usage(1);
    }

    points_fname = argv[optind++];
    num_sites = atoi(argv[optind++]);
    epsilon = atof(argv[optind]);

    double t;

    t = -omp_get_wtime();
    PointTraits::read_fvecs(points, points_fname); num_points = points.size();
    t += omp_get_wtime();
    total_time += t;

    fmt::print("[time={:.3f}] read {} points from file '{}'\n", t, num_points, points_fname);

    results["file_io_time"] = t;
    results["dataset"] = points_fname;
    results["num_points"] = num_points;
    results["dim"] = DIM_SIZE;
    results["num_threads"] = nthreads;
    results["num_sites"] = num_sites;
    results["epsilon"] = epsilon;
    results["covering_factor"] = covering_factor;
    results["leaf_size"] = leaf_size;

    build_greedy_net();
    build_replication_tree();
    compute_ghost_points();
    build_ghost_trees();
    build_epsilon_graph();

#ifdef VERIFY

    t = -omp_get_wtime();

    bool correct = true;
    Index n = graph.size();

    #pragma omp parallel for
    for (Index i = 0; i < n; ++i)
    {
        if (!correct) continue;

        IndexVector neighbors; neighbors.reserve(graph[i].size());

        for (Index j = 0; j < n; ++j)
            if (distance(points[i], points[j]) <= epsilon)
                neighbors.push_back(j);

        if (neighbors.size() != graph[i].size() || !std::is_permutation(neighbors.begin(), neighbors.end(), graph[i].begin()))
        {
            #pragma omp atomic write
            correct = false;
        }
    }


    t += omp_get_wtime();
    fmt::print("[time={:.3f}] epsilon graph {} verification\n", t, correct? "PASSED" : "FAILED");

    results["graph_is_correct"] = correct;

#endif

    results["total_time"] = total_time;

    fmt::print("[total_runtime={:.3f}]\n", total_time);

    if (results_fname)
    {
        std::ofstream f(results_fname);
        f << std::setw(4) << results << std::endl;
        f.close();
    }

    return 0;
}

void build_greedy_net()
{
    json my_results;

    Index n = num_points;
    Index m = num_sites;

    double t;
    Index dist_comps = 0;

    t = -omp_get_wtime();

    net.reserve(m);
    net.push_back(0);
    cells.resize(n);
    dists.resize(n);
    net_sep = -1;

    Index farthest;

    for (Index p = 0; p < n; ++p)
    {
        dists[p] = distance(points[net.back()], points[p]);
        dist_comps++;

        if (dists[p] > net_sep)
        {
            farthest = p;
            net_sep = dists[p];
        }
    }

    for (Index i = 1; i < m; ++i)
    {
        net.push_back(farthest);
        net_sep = -1;

        for (Index p = 0; p < n; ++p)
        {
            Real d = distance(points[net.back()], points[p]);
            dist_comps++;

            if (d < dists[p])
            {
                cells[p] = net.back();
                dists[p] = d;
            }

            if (dists[p] > net_sep)
            {
                farthest = p;
                net_sep = dists[p];
            }
        }
    }

    t += omp_get_wtime();
    total_time += t;

    fmt::print("[time={:.3f}] built r-net Voronoi diagram [sep={:.3f},num_sites={},dist_comps={}]\n", t, net_sep, m, dist_comps);

    my_results["net_sep"] = net_sep;
    my_results["dist_comps"] = dist_comps;
    my_results["time"] = t;

    results["build_greedy_net"] = my_results;
}

void build_replication_tree()
{
    json my_results;

    double t;
    Index verts = 0;
    Index dist_comps = 0;

    t = -omp_get_wtime();

    minlevel = std::ceil(std::log(net_sep)/std::log(covering_factor)) - 1;
    Index i = minlevel;

    IndexSet C;

    for (Index j = 0; j < net.size(); ++j)
        C.insert(j);

    while (C.size() > 1)
    {
        reptree.emplace_back();
        IndexVectorMap& children = reptree.back();

        while (!C.empty())
        {
            auto it = std::min_element(C.begin(), C.end());
            Index p = *it;
            C.erase(it);

            children.insert({p, {p}});
            IndexSet S;

            for (Index q : C)
            {
                if (distance(points[net[p]], points[net[q]]) <= std::pow(covering_factor, i+1))
                {
                    children[p].push_back(q);
                    S.insert(q);
                }
            }

            dist_comps += C.size();

            for (Index q : S)
                C.erase(q);
        }

        C.clear();

        for (const auto& [q, _] : children)
            C.insert(q);

        i++;
    }

    maxlevel = i;

    assert((reptree.back().size() == 1));
    auto it = reptree.back().begin();
    root = it->first;

    t += omp_get_wtime();
    total_time += t;

    for (const auto& level : reptree)
        verts += level.size();

    verts += net.size();

    fmt::print("[time={:.3f}] built replication tree [verts={},dist_comps={}]\n", t, verts, dist_comps);

    my_results["verts"] = verts;
    my_results["dist_comps"] = dist_comps;
    my_results["time"] = t;

    results["build_replication_tree"] = my_results;
}

void compute_ghost_points()
{
    json my_results;

    Index n = num_points;
    Index dist_comps = 0;

    double t;

    t = -omp_get_wtime();

    for (Index p_i : net)
    {
        treeids.insert({p_i, {}});
        cellids.insert({p_i, {}});
    }

    Index num_ghost_points = 0;

    for (Index p = 0; p < n; ++p)
    {
        Index p_i = cells[p];

        cellids[p_i].push_back(p);
        treeids[p_i].push_back(p);

        IndexVector neighbors;
        dist_comps += range_query(neighbors, points[p], dists[p] + 2*epsilon);

        for (Index p_j : neighbors)
        {
            if (p_i != p_j)
            {
                treeids[p_j].push_back(p);
                num_ghost_points++;
            }
        }
    }

    t += omp_get_wtime();
    total_time += t;

    fmt::print("[time={:.3f}] computed ghost points [num_ghost_points={},dist_comps={}]\n", t, num_ghost_points, dist_comps);

    my_results["num_ghost_points"] = num_ghost_points;
    my_results["dist_comps"] = dist_comps;
    my_results["time"] = t;

    results["compute_ghost_points"] = my_results;
}

void build_ghost_trees()
{
    json my_results;

    double t;
    Index dist_comps = 0;

    t = -omp_get_wtime();

    for (Index p_i : net)
    {
        PointVector pts;
        IndexVector ids = treeids.at(p_i);

        pts.reserve(ids.size());

        for (Index i = 0; i < ids.size(); ++i)
            pts.push_back(points[ids[i]]);

        trees.insert({p_i, Tree(pts, ids)});
        dist_comps += trees.at(p_i).build(covering_factor, leaf_size);
    }

    t += omp_get_wtime();
    total_time += t;

    fmt::print("[time={:.3f}] computed ghost trees [dist_comps={}]\n", t, dist_comps);

    my_results["time"] = t;
    my_results["dist_comps"] = dist_comps;
    results["build_ghost_trees"] = my_results;
}

void build_epsilon_graph()
{
    json my_results;

    double t;
    Index dist_comps = 0;

    t = -omp_get_wtime();

    Index n_edges = 0;
    graph.resize(points.size());

    for (Index i = 0; i < num_sites; ++i)
    {
        Index p_i = net[i];

        const auto& V_i = cellids.at(p_i);
        const auto& T_i = trees.at(p_i);

        for (Index u : V_i)
        {
            dist_comps += T_i.range_query(points[u], epsilon, graph[u]);
            IndexSet tmp(graph[u].begin(), graph[u].end());
            graph[u].assign(tmp.begin(), tmp.end());
            n_edges += graph[u].size();
        }
    }

    t += omp_get_wtime();
    total_time += t;

    fmt::print("[time={:.3f}] built epsilon graph [density={:.3f},edges={},dist_comps={}]\n", t, (n_edges+0.0)/points.size(), n_edges, dist_comps);

    my_results["density"] = (n_edges+0.0)/points.size();
    my_results["num_edges"] = n_edges;
    my_results["dist_comps"] = dist_comps;
    my_results["time"] = t;

    results["build_epsilon_graph"] = my_results;
}

Index range_query(IndexVector& neighbors, Point query, Real radius)
{
    using IndexPair = std::pair<Index, Index>;
    using IndexPairVector = std::vector<IndexPair>;

    IndexPairVector stack;
    stack.emplace_back(root, maxlevel);

    Index dist_comps = 0;

    neighbors.clear();

    while (!stack.empty())
    {
        Index u = stack.back().first;
        Index l = stack.back().second;

        stack.pop_back();

        if (l == minlevel)
        {
            if (distance(points[net[u]], query) <= radius)
            {
                neighbors.push_back(net[u]);
            }

            dist_comps++;
        }
        else
        {
            const auto& level = reptree.at(l-minlevel-1);
            const auto& children = level.at(u);

            for (Index v : children)
            {
                if (distance(query, points[net[v]]) <= std::pow(covering_factor, l+1) + radius)
                {
                    stack.emplace_back(v, l-1);
                }
            }

            dist_comps += children.size();
        }
    }

    return dist_comps;
}
