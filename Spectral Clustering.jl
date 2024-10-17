### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ bd04ff16-a2dd-4ac8-a230-872a81322e44
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 1c6be7eb-be86-479f-9af7-93cef151c8df
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ 294af126-8c97-11ef-172b-51a969cac8db
html"""<style>
main {
    max-width: 66%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 1cbfe5ab-13c2-4926-bc0c-a142339cc8c0
md"""
## Loading and Pre-processing Data
"""

# ╔═╡ 21bd809b-779e-4b55-8039-c863bea04bf1
filepath = joinpath(@__DIR__, "MAT Files", "barbara", "barbara_2013.mat")

# ╔═╡ 9c041d1e-eba9-4ca2-be18-70637b1c9c17
CACHEDIR = joinpath(@__DIR__, "cache_files", "Aerial Datasets")

# ╔═╡ 917ccf7f-43c2-4756-afa3-673e7993c405
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ eec929ed-6280-476e-8d23-ea1c45071151
vars = matread(filepath)

# ╔═╡ 5ee71c45-4165-4dff-999d-fdeaa13e3feb
data = vars["HypeRvieW"]

# ╔═╡ b9262c01-3733-49c8-b4cf-9ce0bb158563
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ ccfee2de-5e2a-49a7-8551-2e3832780fb9
with_theme() do
	fig = Figure(; size=(600, 800))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	image!(ax, permutedims(data[:, :, band]))
	fig
end

# ╔═╡ d7b9d628-6e1f-417f-a058-cea75a9fc8cb
function affinity(cube; max_nz=10, chunksize=minimum(size(cube)[1:2]),
	func = c -> exp(-2*acos(clamp(c,-1,1))))

	# Verify that chunksize divides the total number of pixels
	mod(prod(size(cube)[1:2]), chunksize) == 0 ||
		error("chunksize must divide the total number of pixels")

	# Compute normalized spectra (so that inner product = cosine of angle)
	X = permutedims(reshape(cube, :, size(cube,3)))
	X = mapslices(normalize, X; dims=1)

	# Find nonzero values (in chunks)
	C_buf = similar(X, size(X,2), chunksize)    # pairwise cosine buffer
	s_buf = Vector{Int}(undef, size(X,2))       # sorting buffer
	nz_list = @withprogress mapreduce(vcat, enumerate(Iterators.partition(1:size(X,2), chunksize))) do (chunk_idx, chunk)
		# Compute cosine angles (for chunk) and store in buffer
		mul!(C_buf, X', view(X, :, chunk))

		# Zero out all but `max_nz` largest values
		nzs = map(chunk, eachcol(C_buf)) do col, c
			idx = partialsortperm!(s_buf, c, 1:max_nz; rev=true)
			collect(idx), fill(col, max_nz), func.(view(c,idx))
		end

		# Log progress and return
		@logprogress chunk_idx/(size(X,2) ÷ chunksize)
		return nzs
	end

	# Form and return sparse array
	rows = reduce(vcat, getindex.(nz_list, 1))
	cols = reduce(vcat, getindex.(nz_list, 2))
	vals = reduce(vcat, getindex.(nz_list, 3))
	return sparse([rows; cols],[cols; rows],[vals; vals])
end

# ╔═╡ 7b029e3a-673a-4f64-a24e-6f9d0eca37d8
max_nz = 20

# ╔═╡ 9a3791cf-5d01-4306-822e-74326c1d4450
A = cachet(joinpath(CACHEDIR, "Affinity_$max_nz.bson")) do
	affinity(data; max_nz)
end

# ╔═╡ d1da9840-b207-4e6c-8125-302390e73219
function embedding(A, k; seed=0)
	# Set seed for reproducibility
	Random.seed!(seed)

	# Compute node degrees and form Laplacian
	d = vec(sum(A; dims=2))
	Dsqrinv = sqrt(inv(Diagonal(d)))
	L = Symmetric(I - (Dsqrinv * A) * Dsqrinv)

	# Compute eigenvectors
	decomp, history = partialschur(L; nev=k, which=:SR)
	@info history

	return mapslices(normalize, decomp.Q; dims=2)
end

# ╔═╡ 67b90d06-eca3-4a86-9643-bf24da9a2075
n_clusters = 5

# ╔═╡ f11c78a9-a35a-48a4-9c1e-0c9907651aa2
V = embedding(A, n_clusters)

# ╔═╡ f71b41c7-a265-438c-ab24-0801a786f17b
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

# ╔═╡ 51687fb0-786b-4fc0-b7e6-dc92a36be6f6
spec_clusterings = batchkmeans(permutedims(V), n_clusters; maxiter=1000)

# ╔═╡ 0db500b8-366f-4da5-92fc-dc1c366c6a46
costs = [spec_clusterings[i].totalcost for i in 1:100]

# ╔═╡ 831cd3a3-9826-446b-a6ee-5d6fc66b8cfa
min_index = argmin(costs)

# ╔═╡ 10574db7-b212-44cb-8cba-477da79900bd
clusters = spec_clusterings[min_index].assignments

# ╔═╡ 312f3597-3879-4914-a613-2db966f7e411
mat = reshape(clusters, size(data, 1), size(data, 2))

# ╔═╡ c9c39233-2a02-4b7d-afdb-9ba18b52df02
with_theme() do
	fig = Figure(; size=(600, 700))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	colors = Makie.Colors.distinguishable_colors(n_clusters)
	hm = heatmap!(ax, permutedims(mat); colormap=Makie.Categorical(colors))
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ 2947e9dc-3005-4d1e-8657-a11620cd8405


# ╔═╡ Cell order:
# ╠═294af126-8c97-11ef-172b-51a969cac8db
# ╟─1cbfe5ab-13c2-4926-bc0c-a142339cc8c0
# ╠═bd04ff16-a2dd-4ac8-a230-872a81322e44
# ╠═1c6be7eb-be86-479f-9af7-93cef151c8df
# ╠═21bd809b-779e-4b55-8039-c863bea04bf1
# ╠═9c041d1e-eba9-4ca2-be18-70637b1c9c17
# ╠═917ccf7f-43c2-4756-afa3-673e7993c405
# ╠═eec929ed-6280-476e-8d23-ea1c45071151
# ╠═5ee71c45-4165-4dff-999d-fdeaa13e3feb
# ╠═b9262c01-3733-49c8-b4cf-9ce0bb158563
# ╠═ccfee2de-5e2a-49a7-8551-2e3832780fb9
# ╠═d7b9d628-6e1f-417f-a058-cea75a9fc8cb
# ╠═7b029e3a-673a-4f64-a24e-6f9d0eca37d8
# ╠═9a3791cf-5d01-4306-822e-74326c1d4450
# ╠═d1da9840-b207-4e6c-8125-302390e73219
# ╠═67b90d06-eca3-4a86-9643-bf24da9a2075
# ╠═f11c78a9-a35a-48a4-9c1e-0c9907651aa2
# ╠═f71b41c7-a265-438c-ab24-0801a786f17b
# ╠═51687fb0-786b-4fc0-b7e6-dc92a36be6f6
# ╠═0db500b8-366f-4da5-92fc-dc1c366c6a46
# ╠═831cd3a3-9826-446b-a6ee-5d6fc66b8cfa
# ╠═10574db7-b212-44cb-8cba-477da79900bd
# ╠═312f3597-3879-4914-a613-2db966f7e411
# ╠═c9c39233-2a02-4b7d-afdb-9ba18b52df02
# ╠═2947e9dc-3005-4d1e-8657-a11620cd8405
