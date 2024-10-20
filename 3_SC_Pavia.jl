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

# ╔═╡ ac175796-3e1b-4d8c-ad1b-9c67b521c13c
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 3d5c0527-359c-487c-9346-7eb5021c7c02
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ 7c16bf70-8d7d-11ef-23e6-f9d6b2d61dd3
html"""<style>
main {
    max-width: 66%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 41634c3e-36c5-46a4-a1c2-d7b469939902
@bind Location Select(["Pavia", "PaviaUni",])

# ╔═╡ d5f8609c-7b86-4df9-a543-35a82aa7fe16
filepath = joinpath(@__DIR__, "MAT Files", "$Location.mat")

# ╔═╡ 0469353b-46d0-4110-a5c9-5d8efb543786
gt_filepath = joinpath(@__DIR__, "GT Files", "$Location.mat")

# ╔═╡ 4183d79c-e9fd-44bc-8030-efeb1a63997b
CACHEDIR = joinpath(@__DIR__, "cache_files", "Aerial Datasets")

# ╔═╡ 0cfd16eb-9569-4243-b2c5-b8a951ee363a
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ f83ea28c-4afb-48a2-aa62-1bd101ffe866
vars = matread(filepath)

# ╔═╡ ec4ee7f9-0bfb-4d46-a18f-592f70020f5d
vars_gt = matread(gt_filepath)

# ╔═╡ 90306f8b-e1ec-44f4-809a-a6887348c0c6
data = vars["paviaU"]

# ╔═╡ 629b21f2-5e8c-413c-8518-a0bd3a5dc3d4
gt_data = vars_gt["paviaU_gt"]

# ╔═╡ e4152df8-46ed-453a-bef7-e20212b9a48c
labels = length(unique(gt_data))

# ╔═╡ 32898a09-a236-477a-a946-3debd3931420
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 43fb9331-3469-4849-b029-56404f69cd77
with_theme() do
	fig = Figure(; size=(600, 600))
	labels = length(unique(gt_data))
	cmap = distinguishable_colors(labels)
	colored_gt = cmap[gt_data .+ 1]
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	ax1 = Axis(fig[1, 2], aspect=DataAspect(), yreversed=true)
	image!(ax, permutedims(data[:, :, band]))
	image!(ax1, permutedims(colored_gt))
	fig
end

# ╔═╡ ed2b2f3b-f485-4a97-bf8c-2420cc670a3f
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

# ╔═╡ 388af406-2e65-408d-9b9d-ebd6080172b6
max_nz = 20

# ╔═╡ e26cbff7-556d-4b91-bbfc-506c652073de
A = cachet(joinpath(CACHEDIR, "Affinity_$Location$max_nz.bson")) do
	affinity(data; max_nz)
end

# ╔═╡ 6668fae0-e208-4765-b5ec-efb95add8043
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

# ╔═╡ c65d124a-4ac5-40b0-9f9f-2b01a1d99e76
V = embedding(A, labels)

# ╔═╡ ca73fa62-e6c9-40bc-981a-21f23299ff98
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

# ╔═╡ f8957686-0315-4d7f-8324-d14394545557
spec_clusterings = batchkmeans(permutedims(V), labels; maxiter=1000)

# ╔═╡ d511f698-2328-455b-9ecf-234b2e9ecb51
costs = [spec_clusterings[i].totalcost for i in 1:100]

# ╔═╡ 09f90731-1263-4668-8c72-eb14c94a6d6b
min_index = argmin(costs)

# ╔═╡ 97fb0c17-337d-4150-aed4-7dcbb16b7e3d
clusters = spec_clusterings[min_index].assignments

# ╔═╡ 8636216c-61e9-4529-a38e-e55897ed365b
mat = reshape(clusters, size(data, 1), size(data, 2))

# ╔═╡ 7490b637-22cd-4c26-9ab0-5a31ff60248f
with_theme() do
	fig = Figure(; size=(600, 700))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	cmap = distinguishable_colors(labels)
	colored_gt = cmap[gt_data .+ 1]
	colors = Makie.Colors.distinguishable_colors(labels)
	hm = heatmap!(ax, permutedims(mat); colormap=Makie.Categorical(colors))
	ax1 = Axis(fig[1, 2], aspect=DataAspect(), yreversed=true)
	image!(ax1, permutedims(colored_gt))
	# Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ Cell order:
# ╠═7c16bf70-8d7d-11ef-23e6-f9d6b2d61dd3
# ╠═ac175796-3e1b-4d8c-ad1b-9c67b521c13c
# ╠═3d5c0527-359c-487c-9346-7eb5021c7c02
# ╠═41634c3e-36c5-46a4-a1c2-d7b469939902
# ╠═d5f8609c-7b86-4df9-a543-35a82aa7fe16
# ╠═0469353b-46d0-4110-a5c9-5d8efb543786
# ╠═4183d79c-e9fd-44bc-8030-efeb1a63997b
# ╠═0cfd16eb-9569-4243-b2c5-b8a951ee363a
# ╠═f83ea28c-4afb-48a2-aa62-1bd101ffe866
# ╠═ec4ee7f9-0bfb-4d46-a18f-592f70020f5d
# ╠═90306f8b-e1ec-44f4-809a-a6887348c0c6
# ╠═629b21f2-5e8c-413c-8518-a0bd3a5dc3d4
# ╠═e4152df8-46ed-453a-bef7-e20212b9a48c
# ╠═32898a09-a236-477a-a946-3debd3931420
# ╠═43fb9331-3469-4849-b029-56404f69cd77
# ╠═ed2b2f3b-f485-4a97-bf8c-2420cc670a3f
# ╠═388af406-2e65-408d-9b9d-ebd6080172b6
# ╠═e26cbff7-556d-4b91-bbfc-506c652073de
# ╠═6668fae0-e208-4765-b5ec-efb95add8043
# ╠═c65d124a-4ac5-40b0-9f9f-2b01a1d99e76
# ╠═ca73fa62-e6c9-40bc-981a-21f23299ff98
# ╠═f8957686-0315-4d7f-8324-d14394545557
# ╠═d511f698-2328-455b-9ecf-234b2e9ecb51
# ╠═09f90731-1263-4668-8c72-eb14c94a6d6b
# ╠═97fb0c17-337d-4150-aed4-7dcbb16b7e3d
# ╠═8636216c-61e9-4529-a38e-e55897ed365b
# ╠═7490b637-22cd-4c26-9ab0-5a31ff60248f
