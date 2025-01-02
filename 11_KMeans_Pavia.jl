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

# ╔═╡ 8904648f-0bd5-44e0-92c5-4134aefeab45
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 9b6ecd09-a64a-4018-846d-c06c278b7f89
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ dca00a0f-50c3-476b-8670-917fc9f91d90
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

# ╔═╡ fd9c906d-9a02-40fe-819b-65a1aa23a839
@bind Location Select(["Pavia", "PaviaUni",])

# ╔═╡ 505b9084-f3e3-4954-9596-903e5858017c
filepath = joinpath(@__DIR__, "MAT Files", "$Location.mat")

# ╔═╡ bf4b29f5-b455-480f-808c-cec2ccd1d4e7
gt_filepath = joinpath(@__DIR__, "GT Files", "$Location.mat")

# ╔═╡ 2935fba9-5317-4834-8863-9e595a08d2f8
CACHEDIR = joinpath(@__DIR__, "cache_files", "Aerial Datasets")

# ╔═╡ 897d90e9-0963-44e1-8b23-a92a27a627ae
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 333ed4f2-c91f-474c-9670-2edb69366c90
vars = matread(filepath)

# ╔═╡ 949be59b-0bc1-4901-b51e-9bb0a7311457
vars_gt = matread(gt_filepath)

# ╔═╡ 31862280-de9c-495b-bce8-d54d647e9e6e
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ f9d0cd6e-e37b-462b-84a5-1b89857db041
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ fa9eecb7-6362-4fed-932f-558bcbfef470
data = vars[data_key]

# ╔═╡ eaab2283-d003-4ef4-a9e7-cd3843ff0721
gt_data = vars_gt[gt_key]

# ╔═╡ 582067b6-7199-43ad-bcf9-563b2e197024
gt_labels = sort(unique(gt_data))

# ╔═╡ 765bef7c-e75f-4405-ba54-3260ada101cd
bg_indices = findall(gt_data .== 0)

# ╔═╡ 0468fc3b-6e72-41a0-81ea-a99d2c2de53c
md"""
### Defining mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ 7f2cfd75-8296-4c09-ae91-c275fe889373
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ f5412fb0-d068-4b1b-be57-f2c9615da844
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ 4ff54803-8b63-4bb4-bb53-1e5da3e98283
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 121e1954-5a2d-42aa-8510-6ed5a3aafa1a
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

# ╔═╡ 70a641f4-cc5a-4475-93e3-aa005db8665a
md"""
### K-Means Clustering
"""

# ╔═╡ 502a2d4b-87f1-4e4a-a6bb-484ad74755b7
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

# ╔═╡ 5c711144-9744-4780-b95d-8eef32c35558
kmeans_runs = 100

# ╔═╡ 86ed8710-7a37-4266-a6d7-89272b2dfde2
permutedims(reshape(data, :, size(data,3)))

# ╔═╡ 360ee108-52a6-433d-8a61-18f3477b1704
# permutedims(data[mask, :])

# ╔═╡ f4b00d98-36b6-4666-be51-f03b8aabb15b
kmeans_clusterings = cachet(joinpath(splitext(basename(@__FILE__))[1], "kmeans_$Location($n_clusters).bson")) do
	batchkmeans(permutedims(data[mask, :]), n_clusters; nruns=kmeans_runs, maxiter=1000)
end

# ╔═╡ cc393367-dda3-4b2a-8354-4f95202a35ef
KM_Results = kmeans_clusterings[1].assignments

# ╔═╡ b5d514b9-e1e2-4fcb-8687-8ba4ad3ae302


# ╔═╡ e4a23449-3b73-44a5-9e91-d27cd58748b7
relabel_maps = Dict(
	"Pavia" => Dict(
	0 => 0,
	1 => 3,
	2 => 8,
	3 => 1,
	4 => 4,
	5 => 5,
	6 => 9,
	7 => 7,
	8 => 2,
	9 => 6,
),
	"PaviaUni" => Dict(
	0 => 0,
	1 => 8,
	2 => 3,
	3 => 1,
	4 => 9,
	5 => 7,
	6 => 6,
	7 => 2,
	8 => 4,
	9 => 5,
)
)

# ╔═╡ ee527c06-1162-4421-ac49-3731cd18dec4
relabel_keys = relabel_maps[Location]

# ╔═╡ e59f9947-bce9-4cb6-8419-e1878bd7ea4c
D_relabel = [relabel_keys[label] for label in KM_Results]

# ╔═╡ a64ff4d5-82b4-45af-b914-463d7e7cb849
with_theme() do
	

	# Create figure
	fig = Figure(; size=(800, 650))
	colors = Makie.Colors.distinguishable_colors(n_clusters + 1)
	# colors_re = Makie.Colors.distinguishable_colors(length(re_labels))

	# Show data
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=15)
	
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false)

	# Show cluster map
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="K-Means Clustering - $Location", titlesize=15)
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= D_relabel
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
	Colorbar(fig[2,2], hm, tellwidth=false, vertical=false)
	
	fig
end

# ╔═╡ a24675dd-54ae-4c42-84b4-aa3222e58775
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

# ╔═╡ ba6fb26a-ba15-4473-b982-9d7bc829dddc
with_theme() do
	fig = Figure(; size=(800, 650))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels_re, yticks = 1:true_labels_re, title="K-Means - $Location - Confusion Matrix", titlesize=15)
	hm = heatmap!(ax, permutedims(confusion_matrix_re), colormap=:viridis)
	pm = permutedims(confusion_matrix_re)

	for i in 1:true_labels_re, j in 1:predicted_labels_re
        value = round(pm[i, j], digits=1)
        text!(ax, i - 0.02, j - 0.1, text = "$value", color=:black, align = (:center, :center), fontsize=14)
    end
	Colorbar(fig[1, 2], hm; height=Relative(1.0))
	fig
end

# ╔═╡ Cell order:
# ╠═dca00a0f-50c3-476b-8670-917fc9f91d90
# ╠═8904648f-0bd5-44e0-92c5-4134aefeab45
# ╠═9b6ecd09-a64a-4018-846d-c06c278b7f89
# ╠═fd9c906d-9a02-40fe-819b-65a1aa23a839
# ╠═505b9084-f3e3-4954-9596-903e5858017c
# ╠═bf4b29f5-b455-480f-808c-cec2ccd1d4e7
# ╠═2935fba9-5317-4834-8863-9e595a08d2f8
# ╠═897d90e9-0963-44e1-8b23-a92a27a627ae
# ╠═333ed4f2-c91f-474c-9670-2edb69366c90
# ╠═949be59b-0bc1-4901-b51e-9bb0a7311457
# ╠═31862280-de9c-495b-bce8-d54d647e9e6e
# ╠═f9d0cd6e-e37b-462b-84a5-1b89857db041
# ╠═fa9eecb7-6362-4fed-932f-558bcbfef470
# ╠═eaab2283-d003-4ef4-a9e7-cd3843ff0721
# ╠═582067b6-7199-43ad-bcf9-563b2e197024
# ╠═765bef7c-e75f-4405-ba54-3260ada101cd
# ╟─0468fc3b-6e72-41a0-81ea-a99d2c2de53c
# ╠═7f2cfd75-8296-4c09-ae91-c275fe889373
# ╠═f5412fb0-d068-4b1b-be57-f2c9615da844
# ╠═4ff54803-8b63-4bb4-bb53-1e5da3e98283
# ╠═121e1954-5a2d-42aa-8510-6ed5a3aafa1a
# ╟─70a641f4-cc5a-4475-93e3-aa005db8665a
# ╠═502a2d4b-87f1-4e4a-a6bb-484ad74755b7
# ╠═5c711144-9744-4780-b95d-8eef32c35558
# ╠═86ed8710-7a37-4266-a6d7-89272b2dfde2
# ╠═360ee108-52a6-433d-8a61-18f3477b1704
# ╠═f4b00d98-36b6-4666-be51-f03b8aabb15b
# ╠═cc393367-dda3-4b2a-8354-4f95202a35ef
# ╠═b5d514b9-e1e2-4fcb-8687-8ba4ad3ae302
# ╠═e4a23449-3b73-44a5-9e91-d27cd58748b7
# ╠═ee527c06-1162-4421-ac49-3731cd18dec4
# ╠═e59f9947-bce9-4cb6-8419-e1878bd7ea4c
# ╠═a64ff4d5-82b4-45af-b914-463d7e7cb849
# ╠═a24675dd-54ae-4c42-84b4-aa3222e58775
# ╠═ba6fb26a-ba15-4473-b982-9d7bc829dddc
