* SC25 code

High-level overview:

    1. Build a net P_m = (p_1, ..., p_m) of P using greedy permutations. Along the way
       we collect:

           (a) V_i = {p in P : d(p, p_i) = d(p, P_m)} (Voronoi cell for point p_i)
           (b) r_i = max_{p in V_i}d(p, p_i) (radius of V_i)

    2. Build a cover tree T = T(P_m) using a bottom-up strategy. (this is the replication tree)

    3. For each i=1..m, build the set G_i = {p in P \ V_i : p could be an epsilon neighbor of a point in V_i}. (these are the ghost points for V_i)

    4. For each i=1..m, build a cover tree T_i = T(V_i union G_i) using a top-down strategy (already implemented during IPDPS attempt, the so-called ghost trees)

    5. For each p in V_i, find its epsilon neighbors by querying against T_i.



Details:

    1. Let n be the number of points in P. We then do the following:

           p_1 = arbitrary seed point
           D[p] = d(p, p_1) for all p in P
           V[p] = p_1 for all p in P

           for i=2..m do
               p_i = argmax(D)
               r_{i-1} = D[p_i]
               F[p] = d(p, p_i) for all p in P
               V[p] = p_i for all p such that F[p] < D[p]
               D[p] = min(D[p], F[p]) for all p in P

           r = max(D)

        V[p] then stores the Voronoi cell of a given point p, and D[p] stores
        the distance of each point to its Voronoi cell site. r is the farthest distance
        between a Voronoi site and one of its members, meaning that all points in the net P_m
        are pairwise separated by a distance of r. Every point is with a distance of r of
        its Voronoi site for the same reason, so P_m is an r-net.

2. Let k be the largest integer such that 2^k < r. Then putting all the points in P_m on level
    k fulfills the separation condition. We need to build T from the leaves upward. We use
    the following algorithm to do so:

           i = k
           C = P_m

           while |C| > 1 do:
               C' = empty set
               while |C| != 0:
                   p = pop(C)
                   Add p to C'
                   insert self-parent p^{i+1} on level i+1 with p^i as nested child
                   S = empty set
                   for q in C do:
                       if d(p, q) <= 2^{i+1} then:
                           insert q^k as a child of p^{k+1}
                           Add q to S
                   C = C \ S
               i = i + 1
               C = C'

3. Now that we have built T, we have a data structure that can, given any point, find all epsilon neighbors in  P_m. We need to use T to construct the sets G_i. Recall the definition: G_i is the set of all potential epsilon neighbors of points in V_i, that are NOT in V_i already. Suppose we only cared about V_j. We need to find all points p not in V_j that could be within a distance of epsilon of a point in V_j. Suppose that p is a point in V_i. Then if p is epsilon ghost of V_j, there exists a point q in V_j such that d(p,q) <= epsilon. We then have that

           d(p, p_j) <= d(p, q) + d(q, p_j)
                     <= epsilon + d(q, p_j)
                     <= epsilon + d(q, p_i)
                     <= epsilon + d(q, p) + d(p, p_i)
                     <= 2*epsilon + d(p, p_i).

So, to find epsilon ghosts for V_j, we need to find all points p such that d(p, p_j) <= D[p] + 2*epsilon, where D here comes from Algorithm 1. Recall that D[p] is the distance between p and its Voronoi cell site, which is the same as d(p, p_i) assuming that p is in V_i.  We therefore do the following. We go through each Voronoi cell V_i and, for each p in V_i, find which other Voronoi cells V_j for which p is an epsilon ghost. We then add p to a running set G_j. Once we have gone through all V_i, we will also have finished all G_i.  Let's say we are currently working on V_i. For each p in V_i, we need to find all j such that d(p, p_j) <= d(p, p_i) + 2*epsilon. We query T for all points in the intersection of P_m and the ball B(p, D[p] + 2*epsilon)). For each p_j in this set, we add o G_j.

4. Once this is done, we build a cover tree T_i = T(V_i + G_i) for each site i=1..m using the top down construction algorithm  implemented for IPDPS.

5. For each T_i, we query all points in V_i against T_i to find its epsilon neighbors.

