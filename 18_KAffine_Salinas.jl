### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 3a71bb67-4d96-4abd-a653-6f1fd6bced19
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 6d73fbd7-71fa-491f-b7ea-3ce148c6815c
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT, Statistics

# ╔═╡ df7891b5-2404-4d9b-bf8d-0ab70b90b7c5
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

# ╔═╡ 447d6b14-134b-4f42-8d71-93e63cb60891
filepath = joinpath(@__DIR__, "MAT Files", "Salinas_corrected.mat")

# ╔═╡ b9f86828-a12e-46c0-b94e-458af1933f85
gt_filepath = joinpath(@__DIR__, "GT Files", "Salinas_gt.mat")

# ╔═╡ 5496f925-61ac-41f5-b5d8-28898e96a271
CACHEDIR = joinpath(@__DIR__, "cache_files", "KAffine_Salinas")

# ╔═╡ c6ffa0a0-c0c3-41c8-bce8-dd6915ce491d
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 0003539c-dd56-4b6b-864e-26db1c8b0e3f
vars = matread(filepath)

# ╔═╡ 0e744f2a-e08c-494c-8dbf-d1c0f9116919
vars_gt = matread(gt_filepath)

# ╔═╡ e70b456c-2fbf-49fb-a6af-b6898431895b
data = vars["salinas_corrected"]

# ╔═╡ 331ad16a-14b4-4289-82c1-61af3718fe36
gt_data = vars_gt["salinas_gt"]

# ╔═╡ a8e4c051-dab4-423c-8981-c122ed13526c
gt_labels = sort(unique(gt_data))

# ╔═╡ 012c3264-24b2-4ef9-b339-6555c70de9b8
bg_indices = findall(gt_data .== 0)

# ╔═╡ 7177a99d-2846-4bb7-ac8a-d54073490c07
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ e5ae0522-f203-4a71-9f7c-2e86164e3e92
md"""
### Defining mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ 207703a0-40a5-4516-9eba-d3ff2378a57c
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ a91b7363-47e6-4282-a1c9-ba8508114660
data[mask, :]

# ╔═╡ 673ccd2a-e734-46cc-9357-52bafd18bd03
md"""
### affine approximation
"""

# ╔═╡ 2b248a37-3fd3-44d9-9ef5-10ad1988955a
function affine_approx(X, k)
	bhat = mean(eachcol(X))
	Uhat = svd(X - bhat*ones(size(X,2))').U[:,1:k]
	return bhat, Uhat
end

# ╔═╡ 05c8124a-c8ab-436d-87ed-87c974f76f8b
function dist(x, (bhat, Uhat))
	return norm(x - (Uhat*Uhat'*(x-bhat) + bhat))
end

# ╔═╡ 73de4258-fdda-4662-8e79-913d1334cd69
function K_Affine(X, d; niters=100)

	K = length(d)
	N = size(X, 2)
	D = size(X, 1)
	
	#Random Initialization - Affine space bases and base vectors
	U = randn.(size(X, 1), collect(d))
	b = [randn(size(X, 1)) for _ in 1:K]

	#Random Initial Cluster assignment
	c = rand(1:K, N)
	c_prev = copy(c)

	#Iterations
	@progress for t in 1:niters
		#Update Affine space basis
		for k in 1:K
			ilist = findall(==(k), c)
			if isempty(ilist)
				U[1] = randn(D, d[k])
				b[1] = randn(D)
			else
				X_k = X[:, ilist]
				b[k], U[k] = affine_approx(X_k, d[k])
			end
		end

		#Update Clusters
		for i in 1:N
			distances = [dist(X[:, i], (b[k], U[k])) for k in 1:K]
			c[i] = argmin(distances)
		end

		# Break if clusters did not change, update otherwise
		if c == c_prev
			
			@info "Terminated early at iteration $t"
			break
		end
		c_prev .= c
	end

	return U, b, c
end

# ╔═╡ 859415ba-eeb2-479a-acab-c4902ab5ed88
function batch_KAffine(X, d; niters=100, nruns=10)

	D, N = size(X)
	runs = Vector{Tuple{Vector{Matrix{Float64}}, Vector{Vector{Float64}}, Vector{Int}, Float64}}(undef, nruns)
	@progress for idx in 1:nruns
		U, b, c = cachet(joinpath(CACHEDIR, "run_1-$idx.bson")) do
			Random.seed!(idx)
			K_Affine(X, d; niters=niters)
		end

		total_cost = 0
		for i in 1:N
			cost = norm(view(X, :, i) - (U[c[i]]*U[c[i]]'*(view(X, :, i)-b[c[i]]) + b[c[i]]))
			total_cost += cost
		end

		runs[idx] = (U, b, c, total_cost)
	end

	return runs
end

# ╔═╡ 5e6d28e7-58e5-4a67-9da5-50efedbf9b15
KAffine_runs = 50

# ╔═╡ b9deb150-9038-4e97-b7bf-8a6f71c61210
KAffine_Clustering = batch_KAffine(permutedims(data[mask, :]), fill(1, n_clusters); niters=100, nruns=KAffine_runs)

# ╔═╡ f7da5ded-5cd0-4937-a5be-8b4abff133a6
min_idx_KAffine = argmax(KAffine_Clustering[i][4] for i in 1:KAffine_runs)

# ╔═╡ a87984e3-034b-4475-9ded-7d082a88f2a6
KAffine_Results = KAffine_Clustering[min_idx_KAffine][3]

# ╔═╡ 68890796-2b30-4014-b825-05f6f4ebb47b
relabel_map = Dict(
	0 => 0,
	1 => 1,
	2 => 6,
	3 => 3,
	4 => 4,
	5 => 11,
	6 => 2,
	7 => 7,
	8 => 8,
	9 => 12,
	10 => 10,
	11 => 5,
	12 => 9,
	13 => 13,
	14 => 14,
	15 => 15,
	16 => 16,
)

# ╔═╡ 54e6c3c9-dc85-44a4-b7f7-4263adc481d1
D_relabel = [relabel_map[label] for label in KAffine_Results]

# ╔═╡ 10c1c8ad-dc67-43dc-96a3-d71f51d2a704
with_theme() do
	# assignments, idx = spec_aligned, spec_clustering_idx

	# Create figure
	fig = Figure(; size=(800, 800))
	colors = Makie.Colors.distinguishable_colors(n_clusters + 1)
	# colors_re = Makie.Colors.distinguishable_colors(length(re_labels))

	# Show data
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=15)
	
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false, ticklabelsize=:8)

	# Show cluster map
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="K-Affinespaces Clustering Results", titlesize=15)
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= D_relabel
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
	Colorbar(fig[2,2], hm, tellwidth=false, vertical=false, ticklabelsize=:8)
	
	fig
end

# ╔═╡ f04c789b-a126-43fc-8bb5-ae324dada034
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

# ╔═╡ be85fbd8-d251-497b-bb8d-1de180d5d4ca
with_theme() do
	fig = Figure(; size=(800, 800))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels_re, yticks = 1:true_labels_re, title="K-Affinespaces - Salinas - Confusion Matrix", titlesize=15)
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
# ╟─df7891b5-2404-4d9b-bf8d-0ab70b90b7c5
# ╠═3a71bb67-4d96-4abd-a653-6f1fd6bced19
# ╠═6d73fbd7-71fa-491f-b7ea-3ce148c6815c
# ╠═447d6b14-134b-4f42-8d71-93e63cb60891
# ╠═b9f86828-a12e-46c0-b94e-458af1933f85
# ╠═5496f925-61ac-41f5-b5d8-28898e96a271
# ╠═c6ffa0a0-c0c3-41c8-bce8-dd6915ce491d
# ╠═0003539c-dd56-4b6b-864e-26db1c8b0e3f
# ╠═0e744f2a-e08c-494c-8dbf-d1c0f9116919
# ╠═e70b456c-2fbf-49fb-a6af-b6898431895b
# ╠═331ad16a-14b4-4289-82c1-61af3718fe36
# ╠═a8e4c051-dab4-423c-8981-c122ed13526c
# ╠═012c3264-24b2-4ef9-b339-6555c70de9b8
# ╠═7177a99d-2846-4bb7-ac8a-d54073490c07
# ╟─e5ae0522-f203-4a71-9f7c-2e86164e3e92
# ╠═207703a0-40a5-4516-9eba-d3ff2378a57c
# ╠═a91b7363-47e6-4282-a1c9-ba8508114660
# ╟─673ccd2a-e734-46cc-9357-52bafd18bd03
# ╠═2b248a37-3fd3-44d9-9ef5-10ad1988955a
# ╠═05c8124a-c8ab-436d-87ed-87c974f76f8b
# ╠═73de4258-fdda-4662-8e79-913d1334cd69
# ╠═859415ba-eeb2-479a-acab-c4902ab5ed88
# ╠═5e6d28e7-58e5-4a67-9da5-50efedbf9b15
# ╠═b9deb150-9038-4e97-b7bf-8a6f71c61210
# ╠═f7da5ded-5cd0-4937-a5be-8b4abff133a6
# ╠═a87984e3-034b-4475-9ded-7d082a88f2a6
# ╠═68890796-2b30-4014-b825-05f6f4ebb47b
# ╠═54e6c3c9-dc85-44a4-b7f7-4263adc481d1
# ╠═10c1c8ad-dc67-43dc-96a3-d71f51d2a704
# ╠═f04c789b-a126-43fc-8bb5-ae324dada034
# ╠═be85fbd8-d251-497b-bb8d-1de180d5d4ca
