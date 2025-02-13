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
    auto comm = Comm::world();

    PointRecord record;
    size_t b[2];
    int dim;

    size_t& filesize = b[0], &n = b[1];

    if (!comm.rank())
    {
        std::ifstream is;
        is.open(fname, std::ios::binary | std::ios::in);

        is.seekg(0, is.end);
        filesize = is.tellg();
        is.seekg(0, is.beg);

        is.read((char*)&dim, sizeof(int)); assert((dim == D));
        is.close();

        assert((filesize % sizeof(PointRecord)) == 0);
        n = filesize / sizeof(PointRecord);
    }

    comm.bcast(b, 0);
    dim = D;

    IndexVector counts(comm.size()), displs(comm.size());
    get_balanced_counts(counts, n);

    std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), static_cast<Index>(0));

    Index mysize = counts[comm.rank()];
    myoffset = displs[comm.rank()];
    totsize = n;

    std::vector<char> mybuf(mysize * sizeof(PointRecord));

    MPI_File fh;
    MPI_File_open(comm.getcomm(), fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    MPI_Offset fileoffset = myoffset * sizeof(PointRecord);
    MPI_File_read_at_all(fh, fileoffset, mybuf.data(), static_cast<int>(mybuf.size()), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    mypoints.resize(mysize);
    char *ptr = mybuf.data();

    for (Index i = 0; i < mysize; ++i)
    {
        std::memcpy(record.data(), ptr, sizeof(PointRecord));
        unpack_point(record, mypoints[i]);
        ptr += sizeof(PointRecord);
    }
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
