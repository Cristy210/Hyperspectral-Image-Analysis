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

# ╔═╡ 3c38e381-851e-48c7-b014-c525a3872e3f
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ cd948fc6-cb61-4f73-8d75-f62cdf37e8ff
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ c164ba95-9cad-4911-a192-f9fd9fdaa7be
html"""<style>
main {
    max-width: 66%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ d52bb10c-2419-476d-b956-1960eea3394a
@bind Location Select(["Pavia", "PaviaUni",])

# ╔═╡ 5e30f0ce-c571-46a3-b6ea-3dfad3893342
filepath = joinpath(@__DIR__, "MAT Files", "$Location.mat")

# ╔═╡ 347846b9-7493-4f59-aa9f-55dcb17ea52f
gt_filepath = joinpath(@__DIR__, "GT Files", "$Location.mat")

# ╔═╡ 70b67f9a-8ffd-42ab-ac1f-54c50d87a23d
CACHEDIR = joinpath(@__DIR__, "cache_files", "Aerial Datasets")

# ╔═╡ cda2a989-9fae-4aa0-91de-75e4eb6cce96
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ f9e40b76-dee4-4b17-81b1-9c6b390a9d38
vars = matread(filepath)

# ╔═╡ 6fdde689-8792-40a4-ada3-36928e1ffc46
vars_gt = matread(gt_filepath)

# ╔═╡ 3c1752ff-c3a4-4ec2-8c91-576bb3f802d4
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ 3f8b8de0-a881-42f9-881a-1a8a733e5d81
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ d8e94813-e6c3-48cb-ba5e-c538083ef838
data = vars[data_key]

# ╔═╡ 90b5daa8-6bea-43ff-841f-5ebeb957742b
gt_data = vars_gt[gt_key]

# ╔═╡ fa21bddc-f86e-457d-8324-a3b7c2b954bc
gt_labels = sort(unique(gt_data))

# ╔═╡ d573d06a-1e2c-4077-9c84-aaeaadc1e226
bg_indices = findall(gt_data .== 0)

# ╔═╡ 75ad169b-eb41-4e8f-a57a-008c3244cc8c
md"""
### Defining mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ 4189289d-b705-4bef-8889-b996341ebd77
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ 5e95594e-e563-4914-9463-052dbfa6db09
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ 01765d65-5ecf-400d-9adb-1bb392440e97
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ b7f316bf-cc77-4882-b52f-3e4eaff19c2d
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

# ╔═╡ 947caec9-c4cc-4cc6-8585-389d167911a3
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

# ╔═╡ e7060ec5-1ef8-457b-ab9c-8b5eb22521ac
max_nz = 10

# ╔═╡ ca1e3c68-ae82-4830-bc96-41cff8846dd5
A = cachet(joinpath(CACHEDIR, "Affinity_$Location$max_nz.bson")) do
	affinity(permutedims(data[mask, :]); max_nz)
end

# ╔═╡ 0bafb957-e51a-4566-959f-efec2ed82915
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

# ╔═╡ 431c43b1-d50f-404b-86eb-99f6ced57ff9
V = embedding(A, n_clusters)

# ╔═╡ b72b2f04-8e50-4a1d-b669-0dcc5d00e5a2
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

# ╔═╡ 89fe0783-c753-4aab-84b0-56ade7085e2c
spec_clusterings = batchkmeans(permutedims(V), n_clusters; maxiter=1000)

# ╔═╡ db787ee0-d6d4-47e5-849c-7ef43513c719
costs = [spec_clusterings[i].totalcost for i in 1:100]

# ╔═╡ d1a2cf7e-0f76-47ff-8337-c69da0a81bf8
min_index = argmin(costs)

# ╔═╡ 0caf7b9a-1a6f-4b52-a2c0-b5b20d522514
aligned_assignments(clusterings, baseperm=1:maximum(first(clusterings).assignments)) = map(clusterings) do clustering
	# New labels determined by simple heuristic that aims to align different clusterings
	thresh! = a -> (a[a .< 0.2*sum(a)] .= 0; a)
	alignments = [thresh!(a) for a in eachrow(counts(clusterings[1], clustering))]
	new_labels = sortperm(alignments[baseperm]; rev=true)

	# Return assignments with new labels
	return [new_labels[l] for l in clustering.assignments]
end

# ╔═╡ c30a96ff-dc38-4d25-9d01-160228d7b3f8
spec_aligned = aligned_assignments(spec_clusterings)

# ╔═╡ d6401fb4-c383-4152-884d-27fdea1d289e
@bind spec_clustering_idx PlutoUI.Slider(1:length(spec_clusterings); show_value=true)

# ╔═╡ cbe67967-ecee-44cc-83fa-2eba9f6e3d70


# ╔═╡ 9180827c-6c81-4860-aa3b-6e0a92098ad2
with_theme() do
	assignments, idx = spec_aligned, spec_clustering_idx

	# Create figure
	fig = Figure(; size=(600, 750))
	colors = Makie.Colors.distinguishable_colors(n_clusters)

	# Show data
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth")
	
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors))
	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false)

	# Show cluster map
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="Clustering Results")
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= assignments[idx]
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors))
	Colorbar(fig[2,2], hm, tellwidth=false, vertical=false)

	# classes = ["1 - Water", "2 - Trees", "3 - Asphalt", "4 - Self-Blocking Bricks", "5 - Bitumen", "6 - Tiles", "7 - Shadows", "8 - Meadows", "9 - Bare Soil"]
	# ax1 = Axis(fig[1, 3], title="Label Classes")
	# hidedecorations!(ax1)

	# for (i, class_label) in enumerate(classes)
 #    	text!(ax1, class_label, position=(0.1, 1 - i * 0.1), align=:left, color=:black, fontsize=14)
	# end

	fig
end

# ╔═╡ a80982dc-fe35-4e98-b951-93665b49cede
md"""
### Confusion Matrix -- Clustering Results
"""

# ╔═╡ 61ba04a7-c8f1-4aad-8e03-4b919d96046b
begin
	ground_labels = filter(x -> x != 0, gt_labels) #Filter out the background pixel label
	true_labels = length(ground_labels)
	predicted_labels = n_clusters

	confusion_matrix = zeros(Float64, true_labels, predicted_labels) #Initialize a confusion matrix filled with zeros
	cluster_results = fill(NaN32, size(data)[1:2]) #Clustering algorithm results

	clu_assign, idx = spec_aligned, spec_clustering_idx

	cluster_results[mask] .= clu_assign[idx]

	for (label_idx, label) in enumerate(ground_labels)
	
		label_indices = findall(gt_data .== label)
	
		cluster_values = [cluster_results[idx] for idx in label_indices]
		t_pixels = length(cluster_values)
		cluster_counts = [count(==(cluster), cluster_values) for cluster in 1:n_clusters]
		confusion_matrix[label_idx, :] .= [count / t_pixels * 100 for count in cluster_counts]
	end
end

# ╔═╡ 58acd8d7-9f54-4546-b415-4c327f97b86f
with_theme() do
	fig = Figure(; size=(600, 700))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels, yticks = 1:true_labels)
	hm = heatmap!(ax, permutedims(confusion_matrix), colormap=:viridis)
	pm = permutedims(confusion_matrix)

	for i in 1:true_labels, j in 1:predicted_labels
        value = round(pm[i, j], digits=1)
        text!(ax, i - 0.02, j - 0.1, text = "$value", color=:black, align = (:center, :center), fontsize=14)
    end
	Colorbar(fig[1, 2], hm)
	fig
end;

# ╔═╡ f4db5598-5ed4-4c3a-b8c9-811e1778ecab
relabel_map_Pavia = Dict(
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

# ╔═╡ 8f2ce860-19c0-4569-b912-3593702449db
relabel_map_PaviaUni = Dict(
	0 => 0,
	1 => 3,
	2 => 4,
	3 => 1,
	4 => 9,
	5 => 7,
	6 => 6,
	7 => 5,
	8 => 8,
	9 => 2,
)

# ╔═╡ f414abc4-9909-46e6-995b-f76f5d86cdfd
relabel_maps = Dict(
	"Pavia" => Dict(
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
),
	"PaviaUni" => Dict(
	0 => 0,
	1 => 3,
	2 => 4,
	3 => 1,
	4 => 9,
	5 => 7,
	6 => 6,
	7 => 5,
	8 => 8,
	9 => 2,
)
)

# ╔═╡ 0b5d8e16-7ee0-4f71-8f17-89bb5f3363cd
relabel_keys = relabel_maps[Location]

# ╔═╡ 20bc5e97-3e49-4974-bfbf-9eff9bc97d05
D_relabel = [relabel_keys[label] for label in spec_aligned[1]]

# ╔═╡ bad3f503-961b-487b-872f-216aaa174b2f
with_theme() do
	assignments, idx = spec_aligned, spec_clustering_idx

	# Create figure
	fig = Figure(; size=(700, 750))
	colors = Makie.Colors.distinguishable_colors(n_clusters)
	# colors_re = Makie.Colors.distinguishable_colors(length(re_labels))

	# Show data
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=20)
	
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, 9))
	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false)

	# Show cluster map
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="Clustering Results", titlesize=20)
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= D_relabel
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, 9))
	Colorbar(fig[2,2], hm, tellwidth=false, vertical=false)
	
	fig
end

# ╔═╡ Cell order:
# ╠═c164ba95-9cad-4911-a192-f9fd9fdaa7be
# ╠═3c38e381-851e-48c7-b014-c525a3872e3f
# ╠═cd948fc6-cb61-4f73-8d75-f62cdf37e8ff
# ╠═d52bb10c-2419-476d-b956-1960eea3394a
# ╠═5e30f0ce-c571-46a3-b6ea-3dfad3893342
# ╠═347846b9-7493-4f59-aa9f-55dcb17ea52f
# ╠═70b67f9a-8ffd-42ab-ac1f-54c50d87a23d
# ╠═cda2a989-9fae-4aa0-91de-75e4eb6cce96
# ╠═f9e40b76-dee4-4b17-81b1-9c6b390a9d38
# ╠═6fdde689-8792-40a4-ada3-36928e1ffc46
# ╠═3c1752ff-c3a4-4ec2-8c91-576bb3f802d4
# ╠═3f8b8de0-a881-42f9-881a-1a8a733e5d81
# ╠═d8e94813-e6c3-48cb-ba5e-c538083ef838
# ╠═90b5daa8-6bea-43ff-841f-5ebeb957742b
# ╠═fa21bddc-f86e-457d-8324-a3b7c2b954bc
# ╠═d573d06a-1e2c-4077-9c84-aaeaadc1e226
# ╠═75ad169b-eb41-4e8f-a57a-008c3244cc8c
# ╠═4189289d-b705-4bef-8889-b996341ebd77
# ╠═5e95594e-e563-4914-9463-052dbfa6db09
# ╠═01765d65-5ecf-400d-9adb-1bb392440e97
# ╠═b7f316bf-cc77-4882-b52f-3e4eaff19c2d
# ╠═947caec9-c4cc-4cc6-8585-389d167911a3
# ╠═e7060ec5-1ef8-457b-ab9c-8b5eb22521ac
# ╠═ca1e3c68-ae82-4830-bc96-41cff8846dd5
# ╠═0bafb957-e51a-4566-959f-efec2ed82915
# ╠═431c43b1-d50f-404b-86eb-99f6ced57ff9
# ╠═b72b2f04-8e50-4a1d-b669-0dcc5d00e5a2
# ╠═89fe0783-c753-4aab-84b0-56ade7085e2c
# ╠═db787ee0-d6d4-47e5-849c-7ef43513c719
# ╠═d1a2cf7e-0f76-47ff-8337-c69da0a81bf8
# ╠═0caf7b9a-1a6f-4b52-a2c0-b5b20d522514
# ╠═c30a96ff-dc38-4d25-9d01-160228d7b3f8
# ╠═d6401fb4-c383-4152-884d-27fdea1d289e
# ╠═cbe67967-ecee-44cc-83fa-2eba9f6e3d70
# ╟─9180827c-6c81-4860-aa3b-6e0a92098ad2
# ╟─a80982dc-fe35-4e98-b951-93665b49cede
# ╠═61ba04a7-c8f1-4aad-8e03-4b919d96046b
# ╠═58acd8d7-9f54-4546-b415-4c327f97b86f
# ╠═f4db5598-5ed4-4c3a-b8c9-811e1778ecab
# ╠═8f2ce860-19c0-4569-b912-3593702449db
# ╠═f414abc4-9909-46e6-995b-f76f5d86cdfd
# ╠═0b5d8e16-7ee0-4f71-8f17-89bb5f3363cd
# ╠═20bc5e97-3e49-4974-bfbf-9eff9bc97d05
# ╠═bad3f503-961b-487b-872f-216aaa174b2f
