"""
    gather(x::PencilArray{T, N}, [root::Integer=0]) -> Array{T, N}

Gather data from all MPI processes into one (big) array.

Data is received by the `root` process.

Returns the full array on the `root` process, and `nothing` on the other
processes.

Note that `gather` always returns a base `Array`, even when the
`PencilArray` wraps a different kind of array (e.g. a `CuArray`).

This function can be useful for testing, but it shouldn't be used with
very large datasets!
"""
function gather(x::PencilArray{T,N}, root::Integer=0) where {T, N}
    timer = Pencils.timer(pencil(x))

    @timeit_debug timer "gather" begin

    # TODO reduce allocations! see `transpose_impl!`
    comm = get_comm(x)
    rank = MPI.Comm_rank(comm)
    mpi_tag = 42
    pen = pencil(x)
    extra_dims = PencilArrays.extra_dims(x)

    # Each process sends its data to the root process.
    # If the local indices are permuted, the permutation is reverted before
    # sending the data.
    data = let perm = permutation(pen)
        if isidentity(perm)
            parent(x)
        else
            # Apply inverse permutation.
            p = append(inv(perm), Val(length(extra_dims)))
            permutedims(parent(x), Tuple(p))  # creates copy!
        end
    end

    # The output is a regular CPU array.
    DestArray = Array{T}

    # For GPU arrays, this transfers data to the CPU (allocating a new Array).
    # If `data` is already an Array{T}, this is non-allocating.
    data_cpu = convert(DestArray, data)

    if rank != root
        # Wait for data to be sent, then return.
        buf = MPI.Buffer(data_cpu)
        send_req = MPI.Isend(buf, root, mpi_tag, comm)
        MPI.Wait!(send_req)
        return nothing
    end

    # Receive data (root only).
    topo = pen.topology
    Nproc = length(topo)
    recv = Vector{Array{T,N}}(undef, Nproc)
    recv_req = Vector{MPI.Request}(undef, Nproc)

    root_index = -1

    for n = 1:Nproc
        # Global data range that I will receive from process n.
        rrange = pen.axes_all[n]
        rdims = length.(rrange)

        src_rank = topo.ranks[n]  # actual rank of sending process
        if src_rank == root
            root_index = n
            recv_req[n] = MPI.REQUEST_NULL
        else
            # TODO avoid allocation?
            recv[n] = Array{T,N}(undef, rdims..., extra_dims...)
            recv_req[n] = MPI.Irecv!(recv[n], src_rank, mpi_tag, comm)
        end
    end

    # Unpack data.
    dest = DestArray(undef, size_global(x))

    # Copy local data.
    colons_extra_dims = ntuple(n -> Colon(), Val(length(extra_dims)))
    dest[pen.axes_local..., colons_extra_dims...] .= data_cpu

    # Copy remote data.
    for m = 2:Nproc
        n, status = MPI.Waitany!(recv_req)
        rrange = pen.axes_all[n]
        dest[rrange..., colons_extra_dims...] .= recv[n]
    end

    end  # @timeit_debug

    dest
end

