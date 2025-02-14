import Pkg; Pkg.activate(@__DIR__) ##Current Project Directory

using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

filepath = joinpath(@__DIR__, "MAT Files", "Pavia.mat") ##HSI Data Path
gt_filepath = joinpath(@__DIR__, "GT Files", "Pavia.mat") ## Ground Truth Data Path

CACHEDIR = joinpath(@__DIR__, "cache_files", "Aerial Datasets")

function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

vars = matread(filepath)
vars_gt = matread(gt_filepath)

data = vars["pavia"]
gt_data = vars_gt["pavia_gt"]


gt_labels = sort(unique(gt_data))
bg_indices = findall(gt_data .== 0)

begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

n_clusters = length(unique(gt_data)) - 1

function affinity(X::Matrix; max_nz=10, chunksize=isqrt(size(X,2)),
    func = c -> exp(-2*acos(clamp(c,-1,1))))

    # Compute normalized spectra (so that inner product = cosine of angle)
    X = mapslices(normalize, X; dims=1)

    # Find nonzero values (in chunks)
    C_buf = similar(X, size(X,2), chunksize)    # pairwise cosine buffer
    s_buf = Vector{Int}(undef, size(X,2))       # sorting buffer
    nz_list = @withprogress mapreduce(vcat, enumerate(Iterators.partition(1:size(X,2), chunksize))) do (chunk_idx, chunk)
        # Compute cosine angles (for chunk) and store in appropriate buffer
        C_chunk = length(chunk) == chunksize ? C_buf : similar(X, size(X,2), length(chunk))
        mul!(C_chunk, X', view(X, :, chunk))

        # Zero out all but `max_nz` largest values
        nzs = map(chunk, eachcol(C_chunk)) do col, c
            idx = partialsortperm!(s_buf, c, 1:max_nz; rev=true)
            collect(idx), fill(col, max_nz), func.(view(c,idx))
        end

        # Log progress and return
        @logprogress chunk_idx/cld(size(X,2),chunksize)
        return nzs
    end

    # Form and return sparse array
    rows = reduce(vcat, getindex.(nz_list, 1))
    cols = reduce(vcat, getindex.(nz_list, 2))
    vals = reduce(vcat, getindex.(nz_list, 3))
    return sparse([rows; cols],[cols; rows],[vals; vals])
end
affinity(cube::Array{<:Real,3}; kwargs...) =
affinity(permutedims(reshape(cube, :, size(cube,3))); kwargs...)

max_nz = 10

A = cachet(joinpath(CACHEDIR, "Affinity_Pavia$max_nz.bson")) do
	affinity(permutedims(data[mask, :]); max_nz)
end

function embedding(A, k; seed=0, kwargs...)
	# Set seed for reproducibility
	Random.seed!(seed)

	# Compute node degrees and form Laplacian
	d = vec(sum(A; dims=2))
	Dsqrinv = sqrt(inv(Diagonal(d)))
	L = Symmetric(I - (Dsqrinv * A) * Dsqrinv)

	# Compute eigenvectors
	decomp, history = partialschur(L; nev=k, which=:SR, kwargs...)
	@info history

	return mapslices(normalize, decomp.Q; dims=2)
end


## Batch K-Means function
function batchkmeans(X, k, args...; nruns=100, kwargs...)
	runs = @withprogress map(1:nruns) do idx
		# Run K-means
		Random.seed!(idx)  # set seed for reproducibility
		result = with_logger(NullLogger()) do
			kmeans(X, k, args...; kwargs...)
		end

		# Log progress and return result
		@logprogress idx/nruns
		return result
	end

	# Print how many converged
	nconverged = count(run -> run.converged, runs)
	@info "$nconverged/$nruns runs converged"

	# Return runs sorted best to worst
	return sort(runs; by=run->run.totalcost)
end

spec_clusterings = cachet(joinpath(CACHEDIR, "spec-clusterings_Pavia-max_nz-$max_nz-n_clusters-$n_clusters.bson")) do
	V = embedding(A, n_clusters; restarts=2000)
	batchkmeans(permutedims(V), n_clusters; maxiter=1000)
end


min_idx = argmin([spec_clusterings[i].totalcost for i in 1:100])

relabel_map = Dict(
	0 => 0,
	1 => 8,
	2 => 3,
	3 => 1,
	4 => 6,
	5 => 7,
	6 => 2,
	7 => 9,
	8 => 5,
	9 => 4,
)

D_relabel = [relabel_map[label] for label in spec_clusterings[min_idx].assignments]

