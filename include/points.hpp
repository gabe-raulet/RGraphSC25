template <int D>
void FloatingPointTraits<D>::unpack_point(const PointRecord& record, Point& p)
{
    int dim;

    const char *dim_src = record.data();
    const char *pt_src = record.data() + sizeof(int);
    char *pt_dest = (char*)p.data();

    std::memcpy(&dim, dim_src, sizeof(int)); assert((dim == D));
    std::memcpy(pt_dest, pt_src, sizeof(Point));
}

template <int D>
void FloatingPointTraits<D>::read_fvecs(PointVector& points, const char *fname)
{
    Point p;
    std::ifstream is;
    PointRecord record;
    size_t filesize, n;
    int dim;

    is.open(fname, std::ios::binary | std::ios::in);

    is.seekg(0, is.end);
    filesize = is.tellg();
    is.seekg(0, is.beg);

    is.read((char*)&dim, sizeof(int)); assert((dim == D));
    is.seekg(0, is.beg);

    assert((filesize % sizeof(PointRecord)) == 0);
    n = filesize / sizeof(PointRecord);
    points.resize(n);

    for (size_t i = 0; i < n; ++i)
    {
        is.read(record.data(), sizeof(PointRecord));
        unpack_point(record, points[i]);
    }

    is.close();
}

template <int D>
void FloatingPointTraits<D>::read_fvecs(PointVector& mypoints, Index& myoffset, Index& totsize, const char *fname)
{
    /*
     * TODO: Parallelize this to fix memory issues for large point clouds
     */

    auto comm = Comm::world();
    auto timer = comm.get_timer();

    timer.start_timer();

    PointVector points;
    std::vector<int> sendcounts;

    if (!comm.rank())
    {
        read_fvecs(points, fname);
        totsize = points.size();

        sendcounts.resize(comm.size());
        get_balanced_counts(sendcounts, (size_t)totsize);
    }

    comm.bcast(totsize, 0);

    comm.scatterv(points, sendcounts, mypoints, 0);

    Index mysize = mypoints.size();
    comm.exscan(mysize, myoffset, MPI_SUM, static_cast<Index>(0));
}

template <int D>
struct L2Distance<FloatingPointTraits<D>>
{
    using PointTraits = FloatingPointTraits<D>;
    using Point = typename PointTraits::Point;

    float operator()(const Point& p, const Point& q) const
    {
        float sum = 0, delta;

        for (int i = 0; i < D; ++i)
        {
            delta = p[i] - q[i];
            sum += delta * delta;
        }

        return std::sqrt(sum);
    }
};
