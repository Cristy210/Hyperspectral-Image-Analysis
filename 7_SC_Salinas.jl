### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 63de9b5b-5e33-4b2c-99ac-9bfea2900f88
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 020ebf16-6751-42c2-b438-fbc53e4b768e
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ 86b2daa8-4ed6-4235-bbd0-a9d88a1a207e
html"""<style>
input[type*="range"] {
	width: calc(100% - 4rem);
}
main {
    max-width: 96%;
    margin-left: 0%;
    margin-right: 2% !important;
}
"""

# ╔═╡ eab1b825-6441-4274-bb46-806c936d42a7
filepath = joinpath(@__DIR__, "MAT Files", "Salinas_corrected.mat")

# ╔═╡ bb44222b-68c3-4a8d-af9a-9aef2c0823e3
gt_filepath = joinpath(@__DIR__, "GT Files", "Salinas_gt.mat")

# ╔═╡ 431b7f3f-a3a0-4ea0-9df1-b80e1d7cc384
CACHEDIR = joinpath(@__DIR__, "cache_files", "Aerial Datasets")

# ╔═╡ 706e869c-e85b-420e-bb1e-6aa3f427cf1b
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ bf77d63c-09b5-4b80-aa8f-6a8a901a989a
vars = matread(filepath)

# ╔═╡ 7efc774a-eb75-44c5-95ee-76cb9b06f17a
vars_gt = matread(gt_filepath)

# ╔═╡ f60b11ba-3f17-4525-808c-82dd49fce5fe
data = vars["salinas_corrected"]

# ╔═╡ 278820b5-1037-4479-b79b-4e1d90c59f4d
gt_data = vars_gt["salinas_gt"]

# ╔═╡ f104e513-6bf3-43fd-bd87-a6085cf7eb21
gt_labels = sort(unique(gt_data))

# ╔═╡ c4456bce-09b5-4e11-8d1d-b16b50855281
bg_indices = findall(gt_data .== 0)

# ╔═╡ 48dc661e-86ca-4e65-8273-c34f518d0cc8
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ 6819ce1c-12c8-4d93-8a95-f409b78a9ec6
with_theme() do
	
	# Create figure
	fig = Figure(; size=(600, 600))
	colors = Makie.Colors.distinguishable_colors(n_clusters+1)

	# Show data
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="Salinas")
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors))
	Colorbar(fig[1,1], hm, flipaxis=false, flip_vertical_label=true)
	colgap!(fig.layout, 1, -220)

	fig
end

# ╔═╡ 1000381a-3f79-46be-ab85-7ab94176d693
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 5763e16d-4f72-4302-99b7-c52b10269161
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

# ╔═╡ 31eb6d55-a398-4715-a8fc-5780b0377e0d
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ c316e306-4260-48d5-b514-27bdc5509ae7
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

# ╔═╡ f22664a1-bef4-4da4-9762-004aa54ed31d
permutedims(data[mask, :])

# ╔═╡ 56852fa7-8a0b-454b-ba70-ca12a86d551e
max_nz = 150

# ╔═╡ 5ced229a-cfd1-47e9-815e-69eb32b935bc
A = cachet(joinpath(CACHEDIR, "Affinity_Salinas_$max_nz.bson")) do
	affinity(permutedims(data[mask, :]); max_nz)
end

# ╔═╡ e87c036e-65b8-433e-8cdd-e2d119d8d458
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

# ╔═╡ 76e02728-ffa5-4214-9eb1-81e4e4779aca
V = embedding(A, n_clusters)

# ╔═╡ e7c650dd-8a82-44ac-a7f2-c14f0af3e1c7
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

# ╔═╡ b0956a49-0ccc-43f5-a970-e1093b5930ce
spec_clusterings = batchkmeans(permutedims(V), n_clusters; maxiter=100)

# ╔═╡ 01d90d35-fe22-45a5-9d34-48c55d16374e
aligned_assignments(clusterings, baseperm=1:maximum(first(clusterings).assignments)) = map(clusterings) do clustering
	# New labels determined by simple heuristic that aims to align different clusterings
	thresh! = a -> (a[a .< 0.2*sum(a)] .= 0; a)
	alignments = [thresh!(a) for a in eachrow(counts(clusterings[1], clustering))]
	new_labels = sortperm(alignments[baseperm]; rev=true)

	# Return assignments with new labels
	return [new_labels[l] for l in clustering.assignments]
end

# ╔═╡ 35d1546a-730c-4701-85b6-08d06adb68a4
spec_aligned = aligned_assignments(spec_clusterings)

# ╔═╡ 883cf099-8b07-4dac-8cde-ab0e8cd3a97f
min_index = argmin(spec_clusterings[i].totalcost for i in 1:100)

# ╔═╡ 6cc95f84-a545-40f3-8ade-ecc3432c41c0
@bind spec_clustering_idx PlutoUI.Slider(1:length(spec_clusterings); show_value=true)

# ╔═╡ 3894966f-662e-4296-8c89-87cfe06eebab
# clu_map = fill(NaN32, size(data)[1:2])

# ╔═╡ 23f2afbf-7635-4827-9a8f-2dc1c98e2d8e
# with_theme() do
# 	assignments, idx = spec_aligned, spec_clustering_idx

# 	# Create figure
# 	fig = Figure(; size=(800, 650))
# 	colors = Makie.Colors.distinguishable_colors(n_clusters)

# 	# Show data
# 	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth")
	
# 	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors))
# 	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false, ticklabelsize=:8)

# 	# Show cluster map
# 	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="Clustering Results")
# 	clustermap = fill(NaN32, size(data)[1:2])
# 	clustermap[mask] .= assignments[idx]
# 	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors))
# 	Colorbar(fig[2,2], hm, tellwidth=false, vertical=false, ticklabelsize=:8)

# 	fig
# end

# ╔═╡ 4ce4c122-2d70-46a8-a4f7-c9c730548a77
spec_clusterings[min_index].assignments

# ╔═╡ 47a2d47c-0b8d-48c3-800e-ac119c084dbc
with_theme() do
	assignments, idx = spec_aligned, spec_clustering_idx

	# Create figure
	fig = Figure(; size=(800, 650))
	colors = Makie.Colors.distinguishable_colors(n_clusters)

	# Show data
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth")
	
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors))
	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false, ticklabelsize=:8)

	# Show cluster map
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="Clustering Results")
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= assignments[idx]
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors))
	Colorbar(fig[2,2], hm, tellwidth=false, vertical=false, ticklabelsize=:8)

	fig
end

# ╔═╡ aad82e02-9466-48a9-b314-a6561be75a16
md"""
### Confusion Matrix
"""

# ╔═╡ e134cc3a-54db-4a74-babf-78fe70e9a0cc
filter(x -> x != 0, gt_labels)

# ╔═╡ 9f2dd747-767b-4bb1-8ad6-2e1037687fc2
begin
	ground_labels = filter(x -> x != 0, gt_labels) #Filter out the background pixel label
	true_labels = length(ground_labels)
	predicted_labels = n_clusters

	confusion_matrix = zeros(Float64, true_labels, predicted_labels) #Initialize a confusion matrix filled with zeros
	cluster_results = fill(NaN32, size(data)[1:2]) #Clusteirng algorithm results

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

# ╔═╡ 841ee5e1-278b-4ebe-be33-744ce6bd7abc
with_theme() do
	fig = Figure(; size=(900, 800))
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

# ╔═╡ d5c7b00a-9156-44f3-a908-24cea4f121f3
md"""
#### Cluster Relabel
"""

# ╔═╡ 6c6bcbb6-469c-4860-95e7-78ec97ac230c
relabel_map = Dict(
	0 => 0,
	1 => 5,
	2 => 12,
	3 => 3,
	4 => 8,
	5 => 11,
	6 => 9,
	7 => 14,
	8 => 7,
	9 => 16,
	10 => 6,
	11 => 2,
	12 => 10,
	13 => 13,
	14 => 15,
	15 => 1,
	16 => 4,
)

# ╔═╡ 2a6fc313-05a6-45dd-84dc-9cf8fecc6f26
spec_aligned[min_index]

# ╔═╡ 9e49c81a-777f-4f5d-8ea9-ff4d178894c9
D_relabel = [relabel_map[label] for label in spec_aligned[min_index]]

# ╔═╡ cbf023b5-0a95-4bb1-98bb-328430b50aec
with_theme() do
	# assignments, idx = spec_aligned, spec_clustering_idx

	# Create figure
	fig = Figure(; size=(800, 800))
	colors = Makie.Colors.distinguishable_colors(n_clusters+1)

	# Show data
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth")
	
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors))
	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false, ticklabelsize=:8)

	# Show cluster map
	
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="Threshold Subspace Clustering - Salinas")
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= D_relabel
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors))
	Colorbar(fig[2,2], hm, tellwidth=false, vertical=false, ticklabelsize=:8)

	fig
end

# ╔═╡ 8ece88c9-477c-4275-8df3-d7b4b7d3d953
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

# ╔═╡ ef60113d-fa13-4e9f-8e77-4545ca6d4f36
with_theme() do
	fig = Figure(; size=(800, 800))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels_re, yticks = 1:true_labels_re, title="Threshold Subspace Clustering - Salinas - Confusion Matrix")
	hm = heatmap!(ax, permutedims(confusion_matrix_re), colormap=:viridis)
	pm = permutedims(confusion_matrix_re)

	for i in 1:true_labels_re, j in 1:predicted_labels_re
        value = round(pm[i, j], digits=1)
        text!(ax, i - 0.02, j - 0.1, text = "$value", color=:black, align = (:center, :center), fontsize=14)
    end
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ c5bbfe13-116a-4c0e-9791-22b68174f03e
md"""
## Plot Clustering Results vs Spectrum
"""

# ╔═╡ 60a6af2f-d24a-4ff3-9f98-f08924fac72d
masked_2darray = permutedims(data[mask, :]);

# ╔═╡ 2a68f8a6-bd67-4b4b-9c8b-ff9a82cc8a14
masked_gt = dropdims(gt_data[mask, :], dims=2)

# ╔═╡ 6d15aab3-7dc5-4e37-a812-d3bcf28ea1dc


# ╔═╡ 7181e5b5-4cd2-48e1-bab9-cca281333688
with_theme() do
    fig = Figure(; size=(1500, 700))
	supertitle = Label(fig[0, 1:3], "Spectrum Analysis of Clustering Results with Corresponding Ground Truth Label Color", fontsize=20, halign=:center, valign=:top)
	# Label(main_grid[1, 1:2], text="Spectrum Analysis of Clustering Results with Corresponding Ground Truth Label", fontsize=20, halign=:center, valign=:top, padding=(10, 10, 10, 10))
	
    grid_1 = GridLayout(fig[1, 1]; nrow=1, ncol=2)
	grid_2 = GridLayout(fig[1, 2]; nrow=4, ncol=4)
	grid_3 = GridLayout(fig[1, 3]; nrow=1, ncol=2)
	

    # Define Colors
    colors = Makie.Colors.distinguishable_colors(n_clusters + 1) 
    colors_spec = Makie.Colors.distinguishable_colors(n_clusters + 1)[2:end]

    # Heatmaps
    ax_hm = Axis(grid_1[1, 2], aspect=DataAspect(), yreversed=true, title="Clustering Results")
    clustermap = fill(0, size(data)[1:2])
    clustermap[mask] .= D_relabel
    hm = heatmap!(ax_hm, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
    Colorbar(grid_1[1, 1], hm)

    ax_hm1 = Axis(grid_3[1, 1], aspect=DataAspect(), yreversed=true, title="Ground Truth")
    hm1 = heatmap!(ax_hm1, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
    Colorbar(grid_3[1, 2], hm1)

    # Spectrum Plots
    for label in 1:n_clusters
        row = div(label - 1, 4) + 1   
        col = mod(label - 1, 4) + 1   

        ax = Axis(grid_2[row, col], title="Cluster $label")
		hidedecorations!(ax)
        cluster_indices = findall(D_relabel .== label)
        selected_indices = cluster_indices[randperm(length(cluster_indices))[1:200]]

        selected_spectra = masked_2darray[:, selected_indices]
        selected_colors = [colors_spec[Int(round(masked_gt[idx]))] for idx in selected_indices]

        for i in 1:length(selected_indices)
            lines!(ax, selected_spectra[:, i], color=selected_colors[i])
        end
    end

    fig
end

# ╔═╡ 7109e412-98b1-4acc-9617-e8399169a065
with_theme() do
    fig = Figure(; size=(1500, 700))
	supertitle = Label(fig[0, 1:3], "Spectrum Analysis of Clustering Results with Corresponding Clustering Result Label", fontsize=20, halign=:center, valign=:top)
    grid_1 = GridLayout(fig[1, 1]; nrow=1, ncol=2)
	grid_2 = GridLayout(fig[1, 2]; nrow=4, ncol=4)
	grid_3 = GridLayout(fig[1, 3]; nrow=1, ncol=2)

	

    # Define Colors
    colors = Makie.Colors.distinguishable_colors(n_clusters + 1)
    colors_spec = Makie.Colors.distinguishable_colors(n_clusters + 1)[2:end]

    # Heatmap for Clustering Results
    ax_hm = Axis(grid_1[1, 2], aspect=DataAspect(), yreversed=true, title="Clustering Results")
    clustermap = fill(0, size(data)[1:2])
    clustermap[mask] .= D_relabel
    hm = heatmap!(ax_hm, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
    Colorbar(grid_1[1, 1], hm, flipaxis=false)

    # Heatmap for Ground Truth
	ax_hm1 = Axis(grid_3[1, 1], aspect=DataAspect(), yreversed=true, title="Ground Truth")
    hm1 = heatmap!(ax_hm1, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
    Colorbar(grid_3[1, 2], hm1)

    # Spectrum Plots
    for label in 1:n_clusters
        row = div(label - 1, 4) + 1   
        col = mod(label - 1, 4) + 1   

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

# ╔═╡ Cell order:
# ╟─86b2daa8-4ed6-4235-bbd0-a9d88a1a207e
# ╠═63de9b5b-5e33-4b2c-99ac-9bfea2900f88
# ╠═020ebf16-6751-42c2-b438-fbc53e4b768e
# ╠═eab1b825-6441-4274-bb46-806c936d42a7
# ╠═bb44222b-68c3-4a8d-af9a-9aef2c0823e3
# ╠═431b7f3f-a3a0-4ea0-9df1-b80e1d7cc384
# ╠═706e869c-e85b-420e-bb1e-6aa3f427cf1b
# ╠═bf77d63c-09b5-4b80-aa8f-6a8a901a989a
# ╠═7efc774a-eb75-44c5-95ee-76cb9b06f17a
# ╠═f60b11ba-3f17-4525-808c-82dd49fce5fe
# ╠═278820b5-1037-4479-b79b-4e1d90c59f4d
# ╠═f104e513-6bf3-43fd-bd87-a6085cf7eb21
# ╠═c4456bce-09b5-4e11-8d1d-b16b50855281
# ╠═6819ce1c-12c8-4d93-8a95-f409b78a9ec6
# ╠═48dc661e-86ca-4e65-8273-c34f518d0cc8
# ╠═1000381a-3f79-46be-ab85-7ab94176d693
# ╠═5763e16d-4f72-4302-99b7-c52b10269161
# ╠═31eb6d55-a398-4715-a8fc-5780b0377e0d
# ╠═c316e306-4260-48d5-b514-27bdc5509ae7
# ╠═f22664a1-bef4-4da4-9762-004aa54ed31d
# ╠═56852fa7-8a0b-454b-ba70-ca12a86d551e
# ╠═5ced229a-cfd1-47e9-815e-69eb32b935bc
# ╠═e87c036e-65b8-433e-8cdd-e2d119d8d458
# ╠═76e02728-ffa5-4214-9eb1-81e4e4779aca
# ╠═e7c650dd-8a82-44ac-a7f2-c14f0af3e1c7
# ╠═b0956a49-0ccc-43f5-a970-e1093b5930ce
# ╠═01d90d35-fe22-45a5-9d34-48c55d16374e
# ╠═35d1546a-730c-4701-85b6-08d06adb68a4
# ╠═883cf099-8b07-4dac-8cde-ab0e8cd3a97f
# ╠═6cc95f84-a545-40f3-8ade-ecc3432c41c0
# ╠═3894966f-662e-4296-8c89-87cfe06eebab
# ╟─23f2afbf-7635-4827-9a8f-2dc1c98e2d8e
# ╠═4ce4c122-2d70-46a8-a4f7-c9c730548a77
# ╠═47a2d47c-0b8d-48c3-800e-ac119c084dbc
# ╟─aad82e02-9466-48a9-b314-a6561be75a16
# ╠═e134cc3a-54db-4a74-babf-78fe70e9a0cc
# ╠═9f2dd747-767b-4bb1-8ad6-2e1037687fc2
# ╠═841ee5e1-278b-4ebe-be33-744ce6bd7abc
# ╟─d5c7b00a-9156-44f3-a908-24cea4f121f3
# ╠═6c6bcbb6-469c-4860-95e7-78ec97ac230c
# ╠═2a6fc313-05a6-45dd-84dc-9cf8fecc6f26
# ╠═9e49c81a-777f-4f5d-8ea9-ff4d178894c9
# ╠═cbf023b5-0a95-4bb1-98bb-328430b50aec
# ╠═8ece88c9-477c-4275-8df3-d7b4b7d3d953
# ╠═ef60113d-fa13-4e9f-8e77-4545ca6d4f36
# ╟─c5bbfe13-116a-4c0e-9791-22b68174f03e
# ╠═60a6af2f-d24a-4ff3-9f98-f08924fac72d
# ╠═2a68f8a6-bd67-4b4b-9c8b-ff9a82cc8a14
# ╠═6d15aab3-7dc5-4e37-a812-d3bcf28ea1dc
# ╠═7181e5b5-4cd2-48e1-bab9-cca281333688
# ╠═7109e412-98b1-4acc-9617-e8399169a065
