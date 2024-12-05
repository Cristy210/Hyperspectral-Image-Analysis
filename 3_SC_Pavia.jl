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
gt_labels = sort(unique(gt_data))

# ╔═╡ 33fb2f5f-ba50-4448-a435-6c20231d704a
bg_indices = findall(gt_data .== 0)

# ╔═╡ 51eaa0fd-8875-40af-b855-d8ef3d303b15
md"""
### Defining mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ 1ac794f2-0a43-4de5-b181-5869cb285a16
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

# ╔═╡ 8b5a2010-748e-475e-8176-0230cc57a1a9


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

# ╔═╡ c38adb1c-3f7b-4741-aa3c-193c82d851c3
@bind spec_clustering_idx PlutoUI.Slider(1:length(spec_clusterings); show_value=true)

# ╔═╡ 3353e446-d2d8-46c1-a909-c819626db269
with_theme() do
	assignments, idx = spec_aligned, spec_clustering_idx

	# Create figure
	fig = Figure(; size=(600, 750))
	colors = Makie.Colors.distinguishable_colors(n_clusters+1)

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

# ╔═╡ fe5dfa09-ffd3-4512-89a3-4834b790dabe
md"""
### Confusion Matrix -- Clustering Results
"""

# ╔═╡ 355e84c9-3fcb-4634-9436-815fe65680d5
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

# ╔═╡ 58084ada-1773-463c-bacd-0478cbfbd5a1
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

# ╔═╡ 81b128d2-b7cc-4493-bc19-4035f79d5935
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

# ╔═╡ ea9c42bd-218c-4d6a-962d-1fc13f82408b
relabel_keys = relabel_maps[Location]

# ╔═╡ 3ca21e15-d7e5-4582-8da3-5aba521fb937
D_relabel = [relabel_keys[label] for label in spec_aligned[1]]

# ╔═╡ dba9ea3e-d226-4b68-84e4-54fdcb76de62
md"""
### Confusion Matrix -- Best Clustering Result
"""

# ╔═╡ 0cd5fa6d-fa11-4279-a71b-ae414b353aed
with_theme() do
	assignments, idx = spec_aligned, spec_clustering_idx

	# Create figure
	fig = Figure(; size=(700, 750))
	colors = Makie.Colors.distinguishable_colors(n_clusters + 1)
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

# ╔═╡ ce41f7df-d3fe-4c5f-8194-84c33bc36619
begin
	ground_labels_re = filter(x -> x != 0, gt_labels) #Filter out the background pixel label
	true_labels_re = length(ground_labels_re)
	predicted_labels_re = n_clusters

	confusion_matrix_re = zeros(Float64, true_labels_re, predicted_labels_re) #Initialize a confusion matrix filled with zeros
	cluster_results_re = fill(NaN32, size(data)[1:2]) #Clustering algorithm results

	# clu_assign, idx = spec_aligned, spec_clustering_idx

	cluster_results_re[mask] .= D_relabel

	for (label_idx, label) in enumerate(ground_labels_re)
	
		label_indices = findall(gt_data .== label)
	
		cluster_values = [cluster_results_re[idx] for idx in label_indices]
		t_pixels = length(cluster_values)
		cluster_counts = [count(==(cluster), cluster_values) for cluster in 1:n_clusters]
		confusion_matrix_re[label_idx, :] .= [count / t_pixels * 100 for count in cluster_counts]
	end
end

# ╔═╡ a062718a-2f67-4864-aa7e-0129cb5f6f58
with_theme() do
	fig = Figure(; size=(600, 700))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels_re, yticks = 1:true_labels_re, title="Confusion Matrix", titlesize=20)
	hm = heatmap!(ax, permutedims(confusion_matrix_re), colormap=:viridis)
	pm = permutedims(confusion_matrix_re)

	for i in 1:true_labels_re, j in 1:predicted_labels_re
        value = round(pm[i, j], digits=1)
        text!(ax, i - 0.02, j - 0.1, text = "$value", color=:black, align = (:center, :center), fontsize=14)
    end
	Colorbar(fig[1, 2], hm; height=Relative(0.8))
	fig
end

# ╔═╡ b3ce2cbd-471f-4bef-899d-6d85511a2539
md"""
## Plot Clustering Results vs Spectrum
"""

# ╔═╡ 57eeec35-8bb0-4879-929b-b5599c733f84
masked_2darray = permutedims(data[mask, :]);

# ╔═╡ 85db2fc6-525a-490c-8356-1b9bee82e37c
masked_gt = dropdims(gt_data[mask, :], dims=2)

# ╔═╡ beb37be6-fea9-4a21-a7ed-81ba87d4e5b7
D_relabel

# ╔═╡ 473c1586-df91-4ccb-992c-b98aad3f61e6
with_theme() do
    fig = Figure(; size=(1300, 700))
	supertitle = Label(fig[0, 1:3], "Spectrum Analysis of Clustering Results with Corresponding Ground Truth Label", fontsize=20, halign=:center, valign=:top)
	# Label(main_grid[1, 1:2], text="Spectrum Analysis of Clustering Results with Corresponding Ground Truth Label", fontsize=20, halign=:center, valign=:top, padding=(10, 10, 10, 10))
	
    grid_1 = GridLayout(fig[1, 1]; nrow=2, ncol=1)
	grid_2 = GridLayout(fig[1, 2]; nrow=5, ncol=2)
	grid_3 = GridLayout(fig[1, 3]; nrow=2, ncol=1)

	

    # Define Colors
    colors = Makie.Colors.distinguishable_colors(n_clusters + 1)
    colors_spec = Makie.Colors.distinguishable_colors(n_clusters + 1)[2:end]

    # Heatmaps
    ax_hm = Axis(grid_1[1, 1], aspect=DataAspect(), yreversed=true, title="Clustering Results")
    clustermap = fill(0, size(data)[1:2])
    clustermap[mask] .= D_relabel
    hm = heatmap!(ax_hm, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
    Colorbar(grid_1[2, 1], hm, vertical=false)

    ax_hm1 = Axis(grid_3[1, 1], aspect=DataAspect(), yreversed=true, title="Ground Truth")
    hm1 = heatmap!(ax_hm1, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
    Colorbar(grid_3[2, 1], hm1, vertical=false)

    # Spectrum Plots
    for label in 1:n_clusters
        row = div(label - 1, 2) + 1   
        col = mod(label - 1, 2) + 1   

        ax = Axis(grid_2[row, col], title="Cluster $label")
		hidedecorations!(ax)
        cluster_indices = findall(D_relabel .== label)
        selected_indices = cluster_indices[randperm(length(cluster_indices))[1:200]]

        selected_spectra = masked_2darray[:, selected_indices]
        selected_colors = [colors_spec[masked_gt[idx]] for idx in selected_indices]

        for i in 1:length(selected_indices)
            lines!(ax, selected_spectra[:, i], color=selected_colors[i])
        end
    end

    fig
end


# ╔═╡ 6c4c0af5-5820-46c9-aa53-0eb74fdf8038
with_theme() do
    fig = Figure(; size=(1300, 700))
	supertitle = Label(fig[0, 1:3], "Spectrum Analysis of Clustering Results with Corresponding Clustering Result Label", fontsize=20, halign=:center, valign=:top)
    grid_1 = GridLayout(fig[1, 1]; nrow=2, ncol=1)
	grid_2 = GridLayout(fig[1, 2]; nrow=5, ncol=2)
	grid_3 = GridLayout(fig[1, 3]; nrow=2, ncol=1)

	

    # Define Colors
    colors = Makie.Colors.distinguishable_colors(n_clusters + 1)
    colors_spec = Makie.Colors.distinguishable_colors(n_clusters + 1)[2:end]

    # Heatmap for Clustering Results
    ax_hm = Axis(grid_1[1, 1], aspect=DataAspect(), yreversed=true, title="Clustering Results")
    clustermap = fill(0, size(data)[1:2])
    clustermap[mask] .= D_relabel
    hm = heatmap!(ax_hm, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
    Colorbar(grid_1[2, 1], hm, vertical=false)

    # Heatmap for Ground Truth
	ax_hm1 = Axis(grid_3[1, 1], aspect=DataAspect(), yreversed=true, title="Ground Truth")
    hm1 = heatmap!(ax_hm1, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
    Colorbar(grid_3[2, 1], hm1, vertical=false)

    # Spectrum Plots
    for label in 1:n_clusters
        row = div(label - 1, 2) + 1   
        col = mod(label - 1, 2) + 1   

        ax = Axis(grid_2[row, col], title="Cluster $label")
		hidedecorations!(ax)
        cluster_indices = findall(D_relabel .== label)
        selected_indices = cluster_indices[randperm(length(cluster_indices))[1:200]]

        selected_spectra = masked_2darray[:, selected_indices]
        selected_colors = [colors_spec[D_relabel[idx]] for idx in selected_indices]

        for i in 1:length(selected_indices)
            lines!(ax, selected_spectra[:, i], color=selected_colors[i])
        end
    end

    fig
end


# ╔═╡ 1f96c203-8a00-4fef-bfcb-554c08ed09ab


# ╔═╡ 79f5e435-e43b-4dd2-81a0-0ac1d611b218
md"""
### Rough Work
"""

# ╔═╡ 38a49d7b-d975-4ea3-9901-44f789c202e7


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
# ╠═eb0578da-69a3-4957-981e-1a628f340a90
# ╠═369f49bf-6379-4525-aa50-99f60d8e734f
# ╠═31e0d145-cfd2-4a3e-9e88-9b36043fbce3
# ╠═629b21f2-5e8c-413c-8518-a0bd3a5dc3d4
# ╠═22645d58-fb95-4acf-a3df-95626d5e3e75
# ╠═33fb2f5f-ba50-4448-a435-6c20231d704a
# ╟─51eaa0fd-8875-40af-b855-d8ef3d303b15
# ╠═1ac794f2-0a43-4de5-b181-5869cb285a16
# ╠═e4152df8-46ed-453a-bef7-e20212b9a48c
# ╠═32898a09-a236-477a-a946-3debd3931420
# ╠═43fb9331-3469-4849-b029-56404f69cd77
# ╠═8b5a2010-748e-475e-8176-0230cc57a1a9
# ╠═ed2b2f3b-f485-4a97-bf8c-2420cc670a3f
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
# ╠═c38adb1c-3f7b-4741-aa3c-193c82d851c3
# ╠═3353e446-d2d8-46c1-a909-c819626db269
# ╟─fe5dfa09-ffd3-4512-89a3-4834b790dabe
# ╠═355e84c9-3fcb-4634-9436-815fe65680d5
# ╠═58084ada-1773-463c-bacd-0478cbfbd5a1
# ╠═81b128d2-b7cc-4493-bc19-4035f79d5935
# ╠═ea9c42bd-218c-4d6a-962d-1fc13f82408b
# ╠═3ca21e15-d7e5-4582-8da3-5aba521fb937
# ╟─dba9ea3e-d226-4b68-84e4-54fdcb76de62
# ╠═0cd5fa6d-fa11-4279-a71b-ae414b353aed
# ╟─ce41f7df-d3fe-4c5f-8194-84c33bc36619
# ╠═a062718a-2f67-4864-aa7e-0129cb5f6f58
# ╟─b3ce2cbd-471f-4bef-899d-6d85511a2539
# ╠═57eeec35-8bb0-4879-929b-b5599c733f84
# ╠═85db2fc6-525a-490c-8356-1b9bee82e37c
# ╠═beb37be6-fea9-4a21-a7ed-81ba87d4e5b7
# ╠═473c1586-df91-4ccb-992c-b98aad3f61e6
# ╠═6c4c0af5-5820-46c9-aa53-0eb74fdf8038
# ╠═1f96c203-8a00-4fef-bfcb-554c08ed09ab
# ╟─79f5e435-e43b-4dd2-81a0-0ac1d611b218
# ╠═38a49d7b-d975-4ea3-9901-44f789c202e7
