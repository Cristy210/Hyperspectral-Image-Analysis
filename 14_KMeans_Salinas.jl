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

# ╔═╡ e37dc536-1779-4866-8c25-d6565dfe2fc3
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ f8771dbd-7530-48fc-ad2b-2408ce8e1a75
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ 8f503704-4381-4b21-a7bf-eb6363cf64d6
html"""<style>
main {
    max-width: 66%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 22565233-5f57-4f1b-9970-90ecde316b54
filepath = joinpath(@__DIR__, "MAT Files", "Salinas_corrected.mat")

# ╔═╡ 146bf1d5-dd24-423a-97ca-1b04c97fe887
gt_filepath = joinpath(@__DIR__, "GT Files", "Salinas_gt.mat")

# ╔═╡ be0dc82b-1cab-4953-8085-90f52c697460
CACHEDIR = joinpath(@__DIR__, "cache_files", "Aerial Datasets")

# ╔═╡ 449c434e-faba-4a43-91e9-1b0ec27420de
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 85dcfe34-89ac-4c65-8af2-b04d37105a73
vars = matread(filepath)

# ╔═╡ 72da186a-c9c0-4b6d-a7df-f61d79d18a8a
vars_gt = matread(gt_filepath)

# ╔═╡ b7233f31-69c4-42c0-8074-a9cf90929d37
data = vars["salinas_corrected"]

# ╔═╡ 8235923b-ce83-427a-9650-f22467d86b2e
gt_data = vars_gt["salinas_gt"]

# ╔═╡ 3eba04a4-1345-4ec8-8da1-d454998c817b
gt_labels = sort(unique(gt_data))

# ╔═╡ 50942895-27ed-412c-b9f2-4b99d6349e08
bg_indices = findall(gt_data .== 0)

# ╔═╡ a7818f9f-b3bd-40ad-b725-ae3db3647d4a
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ 91b9005d-3f32-4dbf-82d6-45cc4b28fc56
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

# ╔═╡ 5f6737ab-0046-40a8-8687-02b2c847c297
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

# ╔═╡ 4ab25c9b-5e0e-4604-ac0f-f931830fb702
md"""
### K-Means Clustering
"""

# ╔═╡ c0751d99-e085-4edf-9cee-702a6aad96dc
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

# ╔═╡ ec231899-7d3d-404c-8643-7c4ead732c4d
kmeans_runs = 100

# ╔═╡ b946f434-e65b-4b37-97da-400d8cb61bbb
permutedims(reshape(data, :, size(data,3)))

# ╔═╡ 72e09a96-cb3a-4648-ba8b-ed56d4af8664
# permutedims(data[mask, :])

# ╔═╡ 9f7875d0-8601-4b36-8f4c-79b220365238
kmeans_clusterings = cachet(joinpath(splitext(basename(@__FILE__))[1], "kmeans_Salinas_($n_clusters).bson")) do
	batchkmeans(permutedims(data[mask, :]), n_clusters; nruns=kmeans_runs, maxiter=1000)
end

# ╔═╡ ff11d646-76de-407a-b256-eb7759dd4572
KM_Results = kmeans_clusterings[1].assignments

# ╔═╡ 100137ae-e0ca-4fe1-9041-9ab210d134c4
relabel_map = Dict(
	0 => 0,
	1 => 12,
	2 => 1,
	3 => 9,
	4 => 8,
	5 => 3,
	6 => 6,
	7 => 7,
	8 => 4,
	9 => 13,
	10 => 10,
	11 => 11,
	12 => 14,
	13 => 5,
	14 => 2,
	15 => 15,
	16 => 16,
)

# ╔═╡ 24b48d7e-986c-4a5a-9b83-66dcabf9faf5
D_relabel = [relabel_map[label] for label in KM_Results]

# ╔═╡ 2ceed3bd-90c2-46e0-b017-59d25b5ddbcf
with_theme() do
	

	# Create figure
	fig = Figure(; size=(1200, 650))
	colors = Makie.Colors.distinguishable_colors(n_clusters + 1)
	# colors_re = Makie.Colors.distinguishable_colors(length(re_labels))

	# Show data
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=20)
	
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false)

	# Show cluster map
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="K-Means Clustering - Salinas", titlesize=20)
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= D_relabel
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
	Colorbar(fig[2,2], hm, tellwidth=false, vertical=false)
	
	fig
end

# ╔═╡ a3bc6dd9-17ea-46c7-a4be-505bc38e06ae
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

# ╔═╡ 4e7b4e49-0123-42b8-9351-7d0590704654
with_theme() do
	fig = Figure(; size=(800, 700))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels_re, yticks = 1:true_labels_re, title="Confusion Matrix - Salinas")
	hm = heatmap!(ax, permutedims(confusion_matrix_re), colormap=:viridis)
	pm = permutedims(confusion_matrix_re)

	for i in 1:true_labels_re, j in 1:predicted_labels_re
        value = round(pm[i, j], digits=1)
        text!(ax, i - 0.02, j - 0.1, text = "$value", color=:black, align = (:center, :center), fontsize=14)
    end
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ 66dba672-fe4d-42d0-9495-387fcd4e13ab
# ╠═╡ disabled = true
#=╠═╡
n_clusters = length(unique(gt_data)) - 1
  ╠═╡ =#

# ╔═╡ 19f99593-4192-42d6-906a-dd19c4254315
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ 0809c6ef-438e-4bec-9e3c-d58309c76b30
# ╠═╡ disabled = true
#=╠═╡
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)
  ╠═╡ =#

# ╔═╡ 946e68fe-3f68-464a-b2de-46075a3a6462
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ Cell order:
# ╟─8f503704-4381-4b21-a7bf-eb6363cf64d6
# ╠═e37dc536-1779-4866-8c25-d6565dfe2fc3
# ╠═f8771dbd-7530-48fc-ad2b-2408ce8e1a75
# ╠═22565233-5f57-4f1b-9970-90ecde316b54
# ╠═146bf1d5-dd24-423a-97ca-1b04c97fe887
# ╠═be0dc82b-1cab-4953-8085-90f52c697460
# ╠═449c434e-faba-4a43-91e9-1b0ec27420de
# ╠═85dcfe34-89ac-4c65-8af2-b04d37105a73
# ╠═72da186a-c9c0-4b6d-a7df-f61d79d18a8a
# ╠═b7233f31-69c4-42c0-8074-a9cf90929d37
# ╠═8235923b-ce83-427a-9650-f22467d86b2e
# ╠═3eba04a4-1345-4ec8-8da1-d454998c817b
# ╠═50942895-27ed-412c-b9f2-4b99d6349e08
# ╠═66dba672-fe4d-42d0-9495-387fcd4e13ab
# ╠═0809c6ef-438e-4bec-9e3c-d58309c76b30
# ╠═91b9005d-3f32-4dbf-82d6-45cc4b28fc56
# ╠═a7818f9f-b3bd-40ad-b725-ae3db3647d4a
# ╠═19f99593-4192-42d6-906a-dd19c4254315
# ╠═946e68fe-3f68-464a-b2de-46075a3a6462
# ╠═5f6737ab-0046-40a8-8687-02b2c847c297
# ╠═4ab25c9b-5e0e-4604-ac0f-f931830fb702
# ╠═c0751d99-e085-4edf-9cee-702a6aad96dc
# ╠═ec231899-7d3d-404c-8643-7c4ead732c4d
# ╠═b946f434-e65b-4b37-97da-400d8cb61bbb
# ╠═72e09a96-cb3a-4648-ba8b-ed56d4af8664
# ╠═9f7875d0-8601-4b36-8f4c-79b220365238
# ╠═ff11d646-76de-407a-b256-eb7759dd4572
# ╠═100137ae-e0ca-4fe1-9041-9ab210d134c4
# ╠═24b48d7e-986c-4a5a-9b83-66dcabf9faf5
# ╠═2ceed3bd-90c2-46e0-b017-59d25b5ddbcf
# ╠═a3bc6dd9-17ea-46c7-a4be-505bc38e06ae
# ╠═4e7b4e49-0123-42b8-9351-7d0590704654
