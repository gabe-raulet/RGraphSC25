#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "json.hpp"
#include <fstream>
using json = nlohmann::json;

#include "fmt/core.h"
#include "points.h"
#include "ctree.h"
#include "mpienv.h"
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <unistd.h>
#include <omp.h>

#ifndef DIM_SIZE
#error "DIM_SIZE is undefined"
#endif

using Index = int64_t;
using Real = float;
using Comm = MPIEnv::Comm;

using PointTraits = FloatingPointTraits<DIM_SIZE>;
using Distance = L2Distance<PointTraits>;
using Point = typename PointTraits::Point;

using RealVector = std::vector<Real>;
using IndexVector = std::vector<Index>;
using PointVector = std::vector<Point>;

using IndexSet = std::unordered_set<Index>;
using IndexMap = std::unordered_map<Index, Index>;
using IndexVectorMap = std::unordered_map<Index, IndexVector>;
using PointVectorMap = std::unordered_map<Index, PointVector>;
using IndexVectorVector = std::vector<IndexVector>;

using Tree = CoverTree<PointTraits, Distance>;
using TreeMap = std::unordered_map<Index, Tree>;

const char *points_fname = NULL; /* input points filename */
const char *results_fname = NULL; /* output results json (optional) */

Distance distance;

Index mysize; /* my rank's number of points */
Index totsize; /* total number of points */
Index myoffset; /* my rank's offset */
Index num_sites; /* number of points in net */
PointVector mypoints; /* my rank's points */
Real epsilon; /* graph radius */

Real covering_factor = 2.0; /* covering factor */
Index leaf_size = 10; /* min leaf size */

/* void parse_arguments(int argc, char *argv[], const Comm& comm); */
/* bool check_graph(); */

void build_greedy_net(); /* build Voronoi diagram */
void build_replication_tree(); /* build replication tree on Voronoi sites */
void compute_cell_assignments(); /* compute cell assignments to processors */
void compute_ghost_points(); /* compute ghost points for eac Voronoi cell */
void build_ghost_trees(); /* build cover trees on Voronoi cells plus ghost points */
void build_epsilon_graph(); /* build epsilon graph */

MPI_Op MPI_ARGMAX;
void mpi_argmax(void *_in, void *_out, int *len, MPI_Datatype *dtype);

struct PointBall { Point pt; Index id; Real radius; };

Index range_query(IndexVector& neighbors, Point query, Real radius); /* query replication tree for intersecting Voronoi sites */

IndexVector net; /* global indices of Voronoi sites */
IndexVector mycells; /* Voronoi cell site for each local point */
PointVector netpoints; /* Voronoi cell points (shared globally) */
RealVector mydists; /* distance of each local point to its Voronoi site */
Real net_sep; /* minimum separation of each point to its Voronoi site */
TreeMap mytrees; /* maps locally assigned Voronoi sites to their "ghost trees" */
std::vector<IndexVectorMap> reptree; /* replication tree */
IndexMap repids; /* maps Voronoi sites to their slot */
IndexVectorVector mygraph; /* local epsilon subgraph */

Index farthest, minlevel, maxlevel, root; /* farthest point, minimum/maximum level of replication tree, and root point id */
PointVectorMap mytreepts, mycellpts; /* maps net sites to point info */
IndexVectorMap mytreeids, mycellids; /* maps net sites to poitn index info */
IndexVector cell_assignments; /* assignments of each cell to processor */

json results; /* results json */

double total_time = 0;

int main_mpi(int atgc, char *argv[]);
int main(int argc, char *argv[])
{
    MPIEnv::initialize(&argc, &argv);
    int err = main_mpi(argc, argv);
    MPIEnv::finalize();
    return err;
}

int main_mpi(int argc, char *argv[])
{
    auto comm = Comm::world();

    auto usage = [&] (int err, bool isroot)
    {
        if (isroot)
        {
            fprintf(stderr, "Usage: %s [options] <points> <num_sites> <epsilon>\n", argv[0]);
            fprintf(stderr, "Options: -c FLOAT  covering factor [%.2f]\n", covering_factor);
            fprintf(stderr, "         -l INT    leaf size [%lu]\n", (size_t)leaf_size);
            fprintf(stderr, "         -o FILE   output results json\n");
            fprintf(stderr, "         -h        help message\n");
        }

        MPIEnv::exit(err);
    };

    int c;
    while ((c = getopt(argc, argv, "c:l:o:h")) >= 0)
    {
        if      (c == 'c') covering_factor = atof(optarg);
        else if (c == 'l') leaf_size = atoi(optarg);
        else if (c == 'o') results_fname = optarg;
        else if (c == 'h') usage(0, !comm.rank());
    }

    if (argc - optind < 3)
    {
        if (!comm.rank()) fmt::print(stderr, "[err::{}] missing argument(s)\n", __func__);
        usage(1, !comm.rank());
    }

    points_fname = argv[optind++];
    num_sites = atoi(argv[optind++]);
    epsilon = atof(argv[optind]);

    auto timer = comm.get_timer();
    timer.start_timer();

    {
        /*
         * TODO: Parallelize this to fix memory issues for large point clouds
         */

        PointVector points;
        std::vector<int> sendcounts;

        if (!comm.rank())
        {
            PointTraits::read_fvecs(points, points_fname);
            totsize = points.size();

            sendcounts.resize(comm.size());
            get_balanced_counts(sendcounts, (size_t)totsize);
        }

        comm.bcast(totsize, 0);

        comm.scatterv(points, sendcounts, mypoints, 0);
        mysize = mypoints.size();

        comm.exscan(mysize, myoffset, MPI_SUM, static_cast<Index>(0));
    }

    timer.stop_timer();
    double t = timer.get_max_time();
    total_time += t;

    if (!comm.rank()) fmt::print("[time={:.3f}] read {} points from file '{}'\n", t, totsize, points_fname);

    MPI_Op_create(&mpi_argmax, 1, &MPI_ARGMAX);

    results["file_io_time"] = t;
    results["dataset"] = points_fname;
    results["num_points"] = totsize;
    results["num_sites"] = num_sites;
    results["dim"] = DIM_SIZE;
    results["covering_factor"] = covering_factor;
    results["leaf_size"] = leaf_size;
    results["epsilon"] = epsilon;
    results["num_ranks"] = comm.size();

    build_greedy_net();
    build_replication_tree();
    compute_cell_assignments();
    compute_ghost_points();
    build_ghost_trees();
    build_epsilon_graph();

    MPI_Op_free(&MPI_ARGMAX);

    results["total_time"] = total_time;

    if (!comm.rank()) fmt::print("[total_runtime={:.3f}]\n", total_time);

    if (results_fname && !comm.rank())
    {
        std::ofstream f(results_fname);
        f << std::setw(4) << results << std::endl;
        f.close();
    }

    return 0;
}

void mpi_argmax(void *_in, void *_inout, int *len, MPI_Datatype *dtype)
{
    PointBall *in = (PointBall *)_in;
    PointBall *inout = (PointBall *)_inout;

    for (int i = 0; i < *len; ++i)
        if (in[i].radius > inout[i].radius)
            inout[i] = in[i];
}

void build_greedy_net()
{
    json my_results;

    Index n = mypoints.size();
    Index m = num_sites;

    auto comm = Comm::world();
    auto timer = comm.get_timer();

    timer.start_timer();

    net.reserve(m);
    netpoints.reserve(m);
    net.push_back(0);
    mycells.resize(n, 0);
    mydists.resize(n);
    farthest = 0;

    netpoints.push_back(mypoints.front());
    comm.bcast(netpoints.back(), 0);

    net_sep = 0;

    for (Index p = 0; p < n; ++p)
    {
        mydists[p] = distance(netpoints.back(), mypoints[p]);

        if (mydists[p] > net_sep)
        {
            farthest = myoffset + p;
            net_sep = mydists[p];
        }
    }

    PointBall cand = {mypoints[farthest-myoffset], farthest, net_sep};

    comm.allreduce(cand, MPI_ARGMAX);

    netpoints.push_back(cand.pt);
    farthest = cand.id;
    net_sep = cand.radius;

    for (Index i = 1; i < m; ++i)
    {
        net.push_back(farthest);
        net_sep = 0;

        for (Index p = 0; p < n; ++p)
        {
            Real d = distance(mypoints[p], netpoints.back());

            if (d < mydists[p])
            {
                mycells[p] = net.back();
                mydists[p] = d;
            }

            if (mydists[p] > net_sep)
            {
                farthest = myoffset + p;
                net_sep = mydists[p];
            }
        }

        cand.pt = mypoints[farthest-myoffset];
        cand.id = farthest;
        cand.radius = net_sep;

        comm.allreduce(cand, MPI_ARGMAX);

        netpoints.push_back(cand.pt);
        farthest = cand.id;
        net_sep = cand.radius;
    }

    timer.stop_timer();
    double t = timer.get_max_time();
    total_time += t;

    if (!comm.rank()) fmt::print("[time={:.3f}] built r-net Voronoi diagram [sep={:.3f},num_sites={}]\n", t, net_sep, num_sites);

    my_results["net_sep"] = net_sep;
    my_results["time"] = t;

    results["build_greedy_net"] = my_results;
}

void build_replication_tree()
{
    json my_results;

    auto comm = Comm::world();
    auto timer = comm.get_timer();

    Index verts = 0;

    timer.start_timer();

    minlevel = std::ceil(std::log(net_sep)/std::log(covering_factor)) - 1;
    Index i = minlevel;

    IndexSet C;

    for (Index j = 0; j < net.size(); ++j)
    {
        C.insert(j);
        repids.insert({net[j], j});
    }

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
                if (distance(netpoints[p], netpoints[q]) <= std::pow(covering_factor, i+1))
                {
                    children[p].push_back(q);
                    S.insert(q);
                }
            }

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

    timer.stop_timer();
    double t = timer.get_max_time();
    total_time += t;

    for (const auto& level : reptree)
        verts += level.size();

    verts += net.size();

    if (!comm.rank()) fmt::print("[time={:.3f}] built replication tree [verts={}]\n", t, verts);

    my_results["verts"] = verts;
    my_results["time"] = t;

    results["build_replication_tree"] = my_results;
}

void compute_cell_assignments()
{
    json my_results;

    auto comm = Comm::world();
    auto timer = comm.get_timer();
    timer.start_timer();

    cell_assignments.reserve(net.size());
    Index nprocs = comm.size();

    for (Index i = 0; i < net.size(); ++i)
    {
        cell_assignments.push_back(i % nprocs);
    }

    timer.stop_timer();
    double t = timer.get_max_time();
    total_time += t;

    if (!comm.rank()) fmt::print("[time={:.3f}] computed cell assignments\n", t);

    my_results["time"] = t;

    results["compute_cell_assignments"] = my_results;
}

void compute_ghost_points()
{
    json my_results;

    using PointRecord = std::tuple<Index, Index, bool, Point>;
    using PointRecordVector = std::vector<PointRecord>;

    Index n = mypoints.size();

    auto comm = Comm::world();
    auto timer = comm.get_timer();

    timer.start_timer();

    std::vector<PointRecordVector> sendbufs(comm.size());
    PointRecordVector recvbuf;

    Index my_num_ghost_points = 0;

    for (Index p = 0; p < n; ++p)
    {
        Index p_i = mycells[p];
        Index dest = cell_assignments[repids.at(p_i)];

        sendbufs[dest].emplace_back(myoffset+p, p_i, false, mypoints[p]);

        IndexVector neighbors;
        range_query(neighbors, mypoints[p], mydists[p] + 2*epsilon);

        for (Index p_j : neighbors)
        {
            if (p_i != p_j)
            {
                dest = cell_assignments[repids.at(p_j)];
                sendbufs[dest].emplace_back(myoffset+p, p_j, true, mypoints[p]);
                my_num_ghost_points++;
            }
        }
    }

    comm.alltoallv(sendbufs, recvbuf);

    for (const auto& [p, p_i, is_ghost, pt] : recvbuf)
    {
        if (mycellids.find(p_i) == mycellids.end())
        {
            mytreeids.insert({p_i, {}});
            mycellids.insert({p_i, {}});
            mytreepts.insert({p_i, {}});
            mycellpts.insert({p_i, {}});
        }

        mytreeids[p_i].push_back(p);
        mytreepts[p_i].push_back(pt);

        if (!is_ghost)
        {
            mycellids[p_i].push_back(p);
            mycellpts[p_i].push_back(pt);
        }
    }

    timer.stop_timer();
    double t = timer.get_max_time();
    total_time += t;

    Index num_ghost_points = 0;
    comm.reduce(my_num_ghost_points, num_ghost_points, 0, MPI_SUM);

    if (!comm.rank()) fmt::print("[time={:.3f}] computed ghost points [num_ghost_points={}]\n", t, num_ghost_points);

    my_results["num_ghost_points"] = num_ghost_points;
    my_results["time"] = t;

    results["compute_ghost_points"] = my_results;
}

void build_ghost_trees()
{
    json my_results;

    auto comm = Comm::world();
    auto timer = comm.get_timer();
    timer.start_timer();

    for (const auto& [p_i, V_i] : mycellids)
    {
        const IndexVector& ids = mytreeids.at(p_i);
        const PointVector& pts = mytreepts.at(p_i);

        mytrees.insert({p_i, Tree(pts, ids)});
        mytrees.at(p_i).build(covering_factor, leaf_size);
    }

    timer.stop_timer();
    double t = timer.get_max_time();
    total_time += t;

    if (!comm.rank()) fmt::print("[time={:.3f}] computed ghost trees\n", t);

    my_results["time"] = t;
    results["build_ghost_trees"] = my_results;
}

void build_epsilon_graph()
{
    auto comm = Comm::world();
    auto timer = comm.get_timer();
    timer.start_timer();

    json my_results;

    Index my_n_edges = 0;

    for (const auto& [p_i, V_i] : mycellids)
    {
        const auto& T_i = mytrees.at(p_i);
        const auto& P_i = mycellpts.at(p_i);

        for (Index j = 0; j < V_i.size(); ++j)
        {
            mygraph.emplace_back();
            T_i.range_query(P_i[j], epsilon, mygraph.back());
            IndexSet tmp(mygraph.back().begin(), mygraph.back().end());
            mygraph.back().assign(tmp.begin(), tmp.end());
            my_n_edges += mygraph.back().size();
        }
    }

    timer.stop_timer();
    double t = timer.get_max_time();
    total_time += t;

    Index n_edges = 0;
    comm.reduce(my_n_edges, n_edges, 0, MPI_SUM);

    if (!comm.rank()) fmt::print("[time={:.3f}] built epsilon graph [density={:.3f},edges={}]\n", t, (n_edges+0.0)/totsize, n_edges);

    my_results["density"] = (n_edges+0.0)/totsize;
    my_results["num_edges"] = n_edges;
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
            if (distance(netpoints[u], query) <= radius)
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
                if (distance(query, netpoints[v]) <= std::pow(covering_factor, l+1) + radius)
                {
                    stack.emplace_back(v, l-1);
                }
            }

            dist_comps += children.size();
        }
    }

    return dist_comps;
}
