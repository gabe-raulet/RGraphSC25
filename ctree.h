#ifndef COVER_TREE_H_
#define COVER_TREE_H_


#include "fmt/core.h"
#include "fmt/ranges.h"
#include "itree.h"
#include "points.h"
#include <assert.h>
#include <unordered_set>
#include <unordered_map>
#include <omp.h>

template <class PointTraits_, class Distance_>
class CoverTree
{
    public:

        using PointTraits = PointTraits_;
        using Distance = Distance_;
        using Point = typename PointTraits::Point;

        using Index = int64_t;
        using Real = float;

        static inline constexpr Distance distance = Distance();

        using RealVector = std::vector<Real>;
        using IndexVector = std::vector<Index>;
        using PointVector = std::vector<Point>;

        CoverTree(const PointVector& points) : points(points) {}
        CoverTree(const PointVector& points, const IndexVector& globids) : points(points), globids(globids) {}

        void build(Real covering_factor, Index leaf_size);
        Index query(const Point& query, Real radius, IndexVector& neighbors) const;

        struct Ball {Index id; Real radius; };
        struct PointBall {Point pt; Index id; Real radius; };

        using BallTree = InsertTree<Ball>;
        using PointBallTree = InsertTree<PointBall>;

    private:

        PointVector points;
        IndexVector globids;
        PointBallTree tree;

        bool has_globids() const { return !globids.empty(); }
};

template <class CoverTree_>
class Hub
{
    public:

        using CoverTree = CoverTree_;
        using PointTraits = typename CoverTree::PointTraits;
        using Distance = typename CoverTree::Distance;
        using Index = typename CoverTree::Index;
        using Real = typename CoverTree::Real;
        using Point = typename CoverTree::Point;
        using Ball = typename CoverTree::Ball;
        using PointBall = typename CoverTree::PointBall;
        using RealVector = typename CoverTree::RealVector;
        using IndexVector = typename CoverTree::IndexVector;
        using PointVector = typename CoverTree::PointVector;
        using BallTree = typename CoverTree::BallTree;
        using PointBallTree = typename CoverTree::PointBallTree;
        /* using IndexSet = typename CoverTree::IndexSet; */

        using BallVector = std::vector<Ball>;
        using PointBallVector = std::vector<PointBall>;

        static inline constexpr Distance distance = Distance();

        struct HubPoint
        {
            Index id, leader;
            Real dist;

            Ball ball() const { return {id, dist}; }
            PointBall point_ball(Point pt) const { return {pt, id, dist}; }
        };

        using HubVector = std::vector<Hub>;
        using HubPointVector = std::vector<HubPoint>;

        Hub(const PointVector& points) : Hub(points, 0, points.front()) {}
        Hub(const HubPointVector& hub_points, Index representative, Index candidate, Point candidate_point, Index parent, Real radius);

        Index size() const { return hub_size; }
        Index repr() const { return representative; }
        Index cand() const { return candidate; }
        Index parent() const { return hub_parent; }
        Real radius() const { return hub_radius; }
        Real sep() const { return hub_sep; }
        bool is_split(Real split_ratio) const { return sep() <= split_ratio*radius(); }
        PointBall get_cand_ball() const { return {candidate_point, candidate, hub_sep}; }
        const HubPointVector& get_hub_points() const { return hub_points; }
        const IndexVector& get_leaves() const { return leaves; }

        void add_new_leader(const PointVector& points);
        void split_leaders(const PointVector& points);
        void find_leaves(Index min_hub_size);

        Index add_hub_vertex(BallTree& tree);
        Index update_tree(BallTree& tree, HubVector& next_hubs, std::vector<bool>& leaf_flags);

    private:

        Hub(const PointVector& points, Index myoffset, Point repr_pt);

        IndexVector leaders; /* point ids of hub leaders */
        HubPointVector hub_points; /* indices, leader pointers and leader distances of hub points */

        Index hub_size; /* number of hub points */
        Index hub_vertex; /* vertex id of hub tree vertex */
        Index hub_parent; /* vertex id of hub parent tree vertex */
        Index representative; /* point id of hub representative (first leader) */
        Index candidate; /* point id of current candidate for the next leader */
        Point candidate_point; /* current candidate point */

        Real hub_radius; /* hub radius */
        Real hub_sep; /* hub separation */

        IndexVector leaves; /* leaf point ids associated with this hub */
        HubVector new_hubs; /* child hubs */

        bool active; /* false if this hub has already contributed hub/leaf vertices to tree */

        Index localsize() const { return hub_size; }
        Index localoffset() const { return 0; }
};

template <class CoverTree_>
Hub<CoverTree_>::Hub(const PointVector& points, Index myoffset, Point repr_pt)
    : leaders({0}),
      hub_size(points.size()),
      representative(0),
      candidate(0),
      hub_parent(-1),
      hub_radius(0.),
      hub_sep(0.),
      active(true)
{
    hub_points.reserve(hub_size);

    for (Index id = 0; id < hub_size; ++id)
    {
        hub_points.emplace_back(myoffset + id, 0, distance(repr_pt, points[id]));

        if (hub_points[id].dist > hub_points[candidate].dist)
            candidate = id;
    }

    hub_radius = hub_sep = hub_points[candidate].dist;
    candidate_point = points[candidate];

    candidate += myoffset;
}

template <class CoverTree_>
Hub<CoverTree_>::Hub(const HubPointVector& hub_points, Index representative, Index candidate, Point candidate_point, Index parent, Real radius)
    : leaders({representative}),
      hub_points(hub_points),
      hub_size(hub_points.size()),
      representative(representative),
      candidate(candidate),
      candidate_point(candidate_point),
      hub_parent(parent),
      hub_radius(radius),
      hub_sep(radius),
      active(true) {}

template <class CoverTree_>
void Hub<CoverTree_>::add_new_leader(const PointVector& points)
{
    Index n = localsize();
    Index a = localoffset();

    Index new_leader = candidate;
    leaders.push_back(new_leader);

    candidate = a;
    hub_sep = 0.;

    for (Index i = 0; i < n; ++i)
    {
        auto& [id, leader, dist] = hub_points[i];
        Real new_leader_dist = distance(candidate_point, points[id-a]);

        if (new_leader_dist < dist)
        {
            dist = new_leader_dist;
            leader = new_leader;
        }

        if (dist > hub_sep)
        {
            candidate = id;
            hub_sep = dist;
        }
    }

    candidate_point = points[candidate-a]; // update candidate point
}

template <class CoverTree_>
void Hub<CoverTree_>::split_leaders(const PointVector& points)
{
    Index n = localsize();
    Index a = localoffset();

    for (Index leader : leaders)
    {
        Index relcand = 0;
        HubPointVector new_hub_points;

        for (Index i = 0; i < n; ++i)
        {
            if (hub_points[i].leader == leader)
            {
                new_hub_points.push_back(hub_points[i]);

                if (new_hub_points.back().dist > new_hub_points[relcand].dist)
                    relcand = new_hub_points.size()-1;
            }
        }

        Index new_hub_size = new_hub_points.size();

        Real new_radius = -1.;
        Index new_candidate = -1;
        Point new_candidate_point;

        if (new_hub_size > 0)
        {
            new_radius = new_hub_points[relcand].dist;
            new_candidate = new_hub_points[relcand].id;
            new_candidate_point = points[new_hub_points[relcand].id-a];
        }

        new_hubs.emplace_back(new_hub_points, leader, new_candidate, new_candidate_point, -1, new_radius);
    }
}

template <class CoverTree_>
void Hub<CoverTree_>::find_leaves(Index min_hub_size)
{
    HubVector updated_new_hubs;

    for (Index i = 0; i < new_hubs.size(); ++i)
    {
        Hub& new_hub = new_hubs[i];

        if (new_hub.size() <= min_hub_size)
            for (const HubPoint& p : new_hub.hub_points)
                leaves.push_back(p.id);
        else
            updated_new_hubs.push_back(new_hub);
    }

    std::swap(updated_new_hubs, new_hubs);
}

template <class CoverTree_>
typename Hub<CoverTree_>::Index
Hub<CoverTree_>::update_tree(BallTree& tree, HubVector& next_hubs, std::vector<bool>& leaf_flags)
{
    assert((active));
    add_hub_vertex(tree);
    active = false;

    for (Hub& new_hub : new_hubs)
    {
        new_hub.hub_parent = hub_vertex;
        next_hubs.push_back(new_hub);
    }

    for (Index leaf : leaves)
    {
        tree.add_vertex({leaf, 0.}, hub_vertex);
        leaf_flags[leaf] = true;
    }

    return leaves.size();
}

template <class CoverTree_>
typename Hub<CoverTree_>::Index
Hub<CoverTree_>::add_hub_vertex(BallTree& tree)
{
    assert((active));
    hub_vertex = tree.add_vertex({repr(), radius()}, parent());
    return hub_vertex;
}


template <class PointTraits_, class Distance_>
void CoverTree<PointTraits_, Distance_>::build(Real covering_factor, Index leaf_size)
{
    using Hub = Hub<CoverTree>;
    using HubVector = typename Hub::HubVector;

    Index size = points.size();

    Index leaf_count, num_hubs;
    HubVector hubs;
    BallTree balltree;

    hubs.emplace_back(points);

    leaf_count = 0;
    std::vector<bool> leaf_flags(size, false);

    do
    {
        HubVector next_hubs;
        num_hubs = hubs.size();

        HubVector my_hubs;

        for (Index i = 0; i < num_hubs; ++i)
        {
            Hub& hub = hubs[i];

            do { hub.add_new_leader(points); } while (!hub.is_split(1./covering_factor));

            hub.split_leaders(points);
            hub.find_leaves(leaf_size);
            my_hubs.push_back(hub);
        }

        for (Hub& hub : my_hubs)
        {
            leaf_count += hub.update_tree(balltree, next_hubs, leaf_flags);
        }

        std::swap(hubs, next_hubs);

    } while (leaf_count < size);

    auto itemizer = [&](const Ball& ball) -> PointBall { return {points[ball.id], ball.id, ball.radius}; };

    balltree.itemize_new_tree(tree, itemizer);
}

template <class PointTraits_, class Distance_>
typename CoverTree<PointTraits_, Distance_>::Index
CoverTree<PointTraits_, Distance_>::query(const Point& query, Real radius, IndexVector& neighbors) const
{
    const auto& [root_pt, root_id, root_radius] = tree[0];

    Index num_dist_comps = 1;

    if (distance(query, root_pt) > root_radius + radius)
        return num_dist_comps;

    IndexVector stack = {0};

    while (!stack.empty())
    {
        Index u = stack.back(); stack.pop_back();
        const auto& [upt, uid, uradius] = tree[u];

        IndexVector children;
        tree.get_children(u, children);

        if (children.empty())
        {
            if (distance(query, upt) <= radius)
            {
                neighbors.push_back(uid);
            }

            num_dist_comps++;
        }
        else
        {
            for (Index v : children)
            {
                const auto& [vpt, vid, vradius] = tree[v];

                if (distance(query, vpt) <= vradius + radius)
                    stack.push_back(v);

                num_dist_comps++;
            }
        }
    }

    if (has_globids()) std::for_each(neighbors.begin(), neighbors.end(), [&](Index& id) { id = globids[id]; });
}

#endif
