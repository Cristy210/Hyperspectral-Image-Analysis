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

# ╔═╡ fdc7db80-04e9-4d6d-81c7-f6fa15d25279


# ╔═╡ f83ea28c-4afb-48a2-aa62-1bd101ffe866
vars = matread(filepath)

# ╔═╡ ec4ee7f9-0bfb-4d46-a18f-592f70020f5d
vars_gt = matread(gt_filepath)

# ╔═╡ eb0578da-69a3-4957-981e-1a628f340a90
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ 369f49bf-6379-4525-aa50-99f60d8e734f
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ 31e0d145-cfd2-4a3e-9e88-9b36043fbce3
data = vars[data_key]

# ╔═╡ 629b21f2-5e8c-413c-8518-a0bd3a5dc3d4
gt_data = vars_gt[gt_key]

# ╔═╡ 22645d58-fb95-4acf-a3df-95626d5e3e75
gt_labels = unique(gt_data)

# ╔═╡ 33fb2f5f-ba50-4448-a435-6c20231d704a
bg_indices = findall(gt_data .== 0)

# ╔═╡ 69bc85f1-93a3-4dd6-8169-75b2b72c253b
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ e4152df8-46ed-453a-bef7-e20212b9a48c
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ 32898a09-a236-477a-a946-3debd3931420
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 43fb9331-3469-4849-b029-56404f69cd77
with_theme() do
	fig = Figure(; size=(600, 600))
	labels = length(unique(gt_data))
	colors = Makie.Colors.distinguishable_colors(n_clusters)
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	ax1 = Axis(fig[1, 2], aspect=DataAspect(), yreversed=true)
	image!(ax, permutedims(data[:, :, band]))
	hm = heatmap!(ax1, permutedims(gt_data); colormap=Makie.Categorical(colors))
	fig
end

# ╔═╡ ed2b2f3b-f485-4a97-bf8c-2420cc670a3f
begin
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
end

# ╔═╡ 06d247a7-9d7f-4ed6-adc7-5922aa0a865e
permutedims(data[mask, :])

# ╔═╡ 388af406-2e65-408d-9b9d-ebd6080172b6
max_nz = 10

# ╔═╡ e26cbff7-556d-4b91-bbfc-506c652073de
A = cachet(joinpath(CACHEDIR, "Affinity_$Location$max_nz.bson")) do
	affinity(permutedims(data[mask, :]); max_nz)
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
V = embedding(A, n_clusters)

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
spec_clusterings = batchkmeans(permutedims(V), n_clusters; maxiter=1000)

# ╔═╡ d511f698-2328-455b-9ecf-234b2e9ecb51
costs = [spec_clusterings[i].totalcost for i in 1:100]

# ╔═╡ 09f90731-1263-4668-8c72-eb14c94a6d6b
min_index = argmin(costs)

# ╔═╡ 3ca576aa-3ed3-491e-b53d-349cec5e0690
aligned_assignments(clusterings, baseperm=1:maximum(first(clusterings).assignments)) = map(clusterings) do clustering
	# New labels determined by simple heuristic that aims to align different clusterings
	thresh! = a -> (a[a .< 0.2*sum(a)] .= 0; a)
	alignments = [thresh!(a) for a in eachrow(counts(clusterings[1], clustering))]
	new_labels = sortperm(alignments[baseperm]; rev=true)

	# Return assignments with new labels
	return [new_labels[l] for l in clustering.assignments]
end

# ╔═╡ 0661eea8-3c09-49cd-aa3e-8a50c2864f08
spec_aligned = aligned_assignments(spec_clusterings)

# ╔═╡ 0ffd41f7-bc9c-4d45-8c41-d3914f5d5131
clu_map = fill(NaN32, size(data)[1:2])

# ╔═╡ 6686efae-71e8-4d65-a871-bea33454e556
clu_map[mask] .= spec_aligned[1]

# ╔═╡ 70547615-7bbf-44d4-a62d-60996b1005e7
clu_map

# ╔═╡ 7d0c3354-09fe-4adb-9595-e6727e12cab9
spec_aligned[1]

# ╔═╡ c38adb1c-3f7b-4741-aa3c-193c82d851c3
@bind spec_clustering_idx PlutoUI.Slider(1:length(spec_clusterings); show_value=true)

# ╔═╡ 3353e446-d2d8-46c1-a909-c819626db269
with_theme() do
	assignments, idx = spec_aligned, spec_clustering_idx

	# Create figure
	fig = Figure(; size=(600, 450))
	colors = Makie.Colors.distinguishable_colors(n_clusters)

	# Show data
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth")
	
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors))
	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false)

	# Show cluster map
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="Clustering Results")
	clustermap = fill(NaN32, size(data)[1:2])
	clustermap[mask] .= assignments[idx]
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors))
	# Colorbar(fig[1,3], hm)

	fig
end

# ╔═╡ fe5dfa09-ffd3-4512-89a3-4834b790dabe
md"""
### Confusion Matrix
"""

# ╔═╡ 355e84c9-3fcb-4634-9436-815fe65680d5


# ╔═╡ Cell order:
# ╠═7c16bf70-8d7d-11ef-23e6-f9d6b2d61dd3
# ╠═ac175796-3e1b-4d8c-ad1b-9c67b521c13c
# ╠═3d5c0527-359c-487c-9346-7eb5021c7c02
# ╠═41634c3e-36c5-46a4-a1c2-d7b469939902
# ╠═d5f8609c-7b86-4df9-a543-35a82aa7fe16
# ╠═0469353b-46d0-4110-a5c9-5d8efb543786
# ╠═4183d79c-e9fd-44bc-8030-efeb1a63997b
# ╠═0cfd16eb-9569-4243-b2c5-b8a951ee363a
# ╠═fdc7db80-04e9-4d6d-81c7-f6fa15d25279
# ╠═f83ea28c-4afb-48a2-aa62-1bd101ffe866
# ╠═ec4ee7f9-0bfb-4d46-a18f-592f70020f5d
# ╠═eb0578da-69a3-4957-981e-1a628f340a90
# ╠═69bc85f1-93a3-4dd6-8169-75b2b72c253b
# ╠═369f49bf-6379-4525-aa50-99f60d8e734f
# ╠═31e0d145-cfd2-4a3e-9e88-9b36043fbce3
# ╠═629b21f2-5e8c-413c-8518-a0bd3a5dc3d4
# ╠═22645d58-fb95-4acf-a3df-95626d5e3e75
# ╠═33fb2f5f-ba50-4448-a435-6c20231d704a
# ╠═e4152df8-46ed-453a-bef7-e20212b9a48c
# ╠═32898a09-a236-477a-a946-3debd3931420
# ╠═43fb9331-3469-4849-b029-56404f69cd77
# ╠═ed2b2f3b-f485-4a97-bf8c-2420cc670a3f
# ╠═06d247a7-9d7f-4ed6-adc7-5922aa0a865e
# ╠═388af406-2e65-408d-9b9d-ebd6080172b6
# ╠═e26cbff7-556d-4b91-bbfc-506c652073de
# ╠═6668fae0-e208-4765-b5ec-efb95add8043
# ╠═c65d124a-4ac5-40b0-9f9f-2b01a1d99e76
# ╠═ca73fa62-e6c9-40bc-981a-21f23299ff98
# ╠═f8957686-0315-4d7f-8324-d14394545557
# ╠═d511f698-2328-455b-9ecf-234b2e9ecb51
# ╠═09f90731-1263-4668-8c72-eb14c94a6d6b
# ╠═3ca576aa-3ed3-491e-b53d-349cec5e0690
# ╠═0661eea8-3c09-49cd-aa3e-8a50c2864f08
# ╠═0ffd41f7-bc9c-4d45-8c41-d3914f5d5131
# ╠═6686efae-71e8-4d65-a871-bea33454e556
# ╠═70547615-7bbf-44d4-a62d-60996b1005e7
# ╠═7d0c3354-09fe-4adb-9595-e6727e12cab9
# ╠═c38adb1c-3f7b-4741-aa3c-193c82d851c3
# ╠═3353e446-d2d8-46c1-a909-c819626db269
# ╟─fe5dfa09-ffd3-4512-89a3-4834b790dabe
# ╠═355e84c9-3fcb-4634-9436-815fe65680d5
