#ifndef MPI_ENV_H_
#define MPI_ENV_H_

#include <map>
#include <sstream>
#include <numeric>
#include <ostream>
#include <typeinfo>
#include <algorithm>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#include <tuple> // std::tuple
#include <array> // std::array
#include <vector> // std::vector
#include <string> // std::string
#include <concepts> // std::same_as
#include <type_traits> // std::{true,false}_type, std::is_array

template <class T>
struct is_tuple : std::false_type {};

template <class ...Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {};

template <class T>
inline constexpr bool is_tuple_v = is_tuple<T>::value;

template <class T>
struct is_std_array : std::false_type {};

template <class T, size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

template <class T>
inline constexpr bool is_array_type = is_std_array<T>::value || std::is_array<T>::value;

template <class T>
concept array_type = is_array_type<T>;

template <array_type T>
struct array_info;

template <class T, size_t N>
struct array_info<T[N]>
{
    using value_type = T;
    static constexpr size_t size = N;
};

template <class T, size_t N>
struct array_info<std::array<T, N>>
{
    using value_type = T;
    static constexpr size_t size = N;
};

template <class T>
inline constexpr size_t array_size = array_info<T>::size;

template <class T>
concept string_type = std::same_as<T, std::string> || std::same_as<T, std::vector<char>>;

template <class T>
concept real_type = std::same_as<T, float> || std::same_as<T, double>;

template <class T>
concept index_type = std::is_integral_v<T>;

template <index_type Index>
void get_balanced_counts(std::vector<Index>& counts, size_t totsize);

namespace MPIEnv
{
    int initialize(int *argc, char **argv[]);
    int finalize();
    void exit(int err);

    bool is_initialized();
    bool is_finalized();

    bool comms_equal(MPI_Comm lhs, MPI_Comm rhs);

    struct type_info_compare
    {
        bool operator()(const std::type_info *lhs, const std::type_info *rhs) const;
    };

    class TypeCache
    {
        public:

            TypeCache();
            ~TypeCache();

            MPI_Datatype get_type(const std::type_info *t);
            void set_type(const std::type_info *t, MPI_Datatype dtype);
            void free_types();
            void reset();

        private:

            std::map<const std::type_info*, MPI_Datatype, type_info_compare> type_map;
    };

    class CommTimer
    {
        public:

            CommTimer(MPI_Comm world)
            {
                MPI_Comm_dup(world, &comm);
                MPI_Comm_rank(comm, &myrank);
                MPI_Comm_size(comm, &nprocs);
            }

            CommTimer(const CommTimer& rhs)
            {
                MPI_Comm_dup(rhs.comm, &comm);
                MPI_Comm_rank(comm, &myrank);
                MPI_Comm_size(comm, &nprocs);
            }

            ~CommTimer() { MPI_Comm_free(&comm); }

            void start_timer()
            {
                MPI_Barrier(comm);
                t = -MPI_Wtime();
            }

            void stop_timer()
            {
                t += MPI_Wtime();
                MPI_Allreduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, comm);
                MPI_Allreduce(&t, &sumtime, 1, MPI_DOUBLE, MPI_SUM, comm);
            }

            double get_my_time() const { return t; }
            double get_max_time() const { return maxtime; }
            double get_sum_time() const { return sumtime; }
            double get_avg_time() const { return sumtime/nprocs; }

        private:

            int myrank, nprocs;
            MPI_Comm comm;
            double t, maxtime, sumtime;
    };

    class Comm
    {
        public:

            static Comm world() { return Comm(MPI_COMM_WORLD); }
            static Comm self() { return Comm(MPI_COMM_SELF); }
            static Comm null() { return Comm(MPI_COMM_NULL); }

            bool operator==(const Comm& rhs) const;
            bool operator!=(const Comm& rhs) const { return !(*this == rhs); }

            Comm& operator=(const Comm& rhs);

            MPI_Comm getcomm() const { return commbuf[0]; }
            int rank() const { return myrank; }
            int size() const { return nprocs; }
            std::tuple<int, int, MPI_Comm> comminfo() const { return {myrank, nprocs, commbuf[0]}; }

            CommTimer get_timer() const { return CommTimer(getcomm()); }

            bool is_distributed() const { return (nprocs > 1); }

            Comm();
            Comm(MPI_Comm comm);
            Comm(const Comm& rhs);
            ~Comm();

            void swap(Comm& rhs) noexcept;

            template <class T>
            bool is_same_val(T val) const;

            template <class T>
            bool are_same_vals(const std::vector<T>& vals) const;

            int barrier() const;

            template <class T> int reduce(const T* sendbuf, T* recvbuf, int count, int root, MPI_Op op) const;
            template <class T> int reduce(const T& sendbuf, T& recvbuf, int root, MPI_Op op) const;
            template <class T> int reduce(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, int root, MPI_Op op) const;

            template <class T> int bcast(T* buffer, int count, int root) const;
            template <class T> int bcast(T& buffer, int root) const;
            template <class T> int bcast(std::vector<T>& buffer, int root) const;

            template <class T> int exscan(const T* sendbuf, T* recvbuf, int count, MPI_Op op, T identity) const;
            template <class T> int exscan(const T& sendbuf, T& recvbuf, MPI_Op op, T identity) const;

            template <class T> int allreduce(const T* sendbuf, T* recvbuf, int count, MPI_Op op) const;
            template <class T> int allreduce(const T& sendbuf, T& recvbuf, MPI_Op op) const;
            template <class T> int allreduce(T* buffer, int count, MPI_Op op) const;
            template <class T> int allreduce(T& buffer, MPI_Op op) const;
            template <class T> int allreduce(std::vector<T>& buffer, MPI_Op op) const;
            template <class T> int allreduce(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, MPI_Op op) const;

            template <class T> int gather(const T* sendbuf, int count, T* recvbuf, int root) const;
            template <class T> int gather(const T& sendbuf, std::vector<T>& recvbuf, int root) const;
            template <class T> int gather(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, int root) const;
            template <class T> int gatherv(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, int root) const;

            template <class T> int allgather(const T* sendbuf, int count, T* recvbuf) const;
            template <class T> int allgather(const T& sendbuf, std::vector<T>& recvbuf) const;
            template <class T> int allgather(T* buffer, int count) const;
            template <class T> int allgather(std::vector<T>& buffer) const;
            template <class T> int allgatherv(const std::vector<T>& sendbuf, std::vector<T>& recvbuf) const;

            template <class T> int scatter(const T* sendbuf, int count, T* recvbuf, int root) const;
            template <class T> int scatter(const std::vector<T>& sendbuf, T& recvbuf, int root) const;
            template <class T> int scatter(const std::vector<T>& sendbuf, std::vector<T>& recvbuf, int root) const;
            template <class T> int scatterv(const std::vector<T>& sendbuf, const std::vector<int>& sendcounts, std::vector<T>& recvbuf, int root) const;
            template <class T> int scatterv(const std::vector<std::vector<T>>& sendbufs, std::vector<T>& recvbuf, int root) const;

            template <class T> int alltoall(const T* sendbuf, int count, T* recvbuf) const;
            template <class T> int alltoall(const std::vector<T>& sendbuf, std::vector<T>& recvbuf) const;
            template <class T> int alltoallv(const std::vector<T>& sendbuf, const std::vector<int>& sendcounts, std::vector<T>& recvbuf) const;
            template <class T> int alltoallv(const std::vector<std::vector<T>>& sendbufs, std::vector<T>& recvbuf) const;

            void log_strings(const std::string& mystr, std::ostream& os) const;
            void log_string(const std::string& str, std::ostream& os, int root) const;

        private:

            int myrank, nprocs;
            MPI_Comm commbuf[1];

            void init(MPI_Comm comm);
    };

    template <real_type Real, index_type Index>
    struct ArgmaxPair
    {
        Index index;
        Real value;

        ArgmaxPair(Index index, Real value) : index(index), value(value) {}

        static void mpi_argmax(void *_in, void *_inout, int *len, MPI_Datatype *dtype);
        static void create_mpi_handlers(MPI_Datatype& MPI_ARGMAX_PAIR, MPI_Op& MPI_ARGMAX);
    };

    TypeCache cache;

    template <class T> MPI_Datatype mpi_type();
    template <class T> MPI_Datatype mpi_commit_type();
};

#include "mpienv.hpp"

#endif
