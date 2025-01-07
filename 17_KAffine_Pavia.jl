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

# ╔═╡ 3ff5fa73-4af0-4f51-b119-27807de8f95e
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 39cde77c-5647-4262-aa38-ce373415c2ac
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT, Statistics

# ╔═╡ e510d702-e7f0-4c01-a8bc-12bdb1b80cab
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

# ╔═╡ 60351d80-ef17-4568-a0eb-5f8f5552159c
@bind Location Select(["Pavia", "PaviaUni",])

# ╔═╡ 1e2d8cad-f61b-49b3-a9bd-685e05108c1a
filepath = joinpath(@__DIR__, "MAT Files", "$Location.mat")

# ╔═╡ acd2e6f9-122c-45dc-9dfe-779d11c99e89
gt_filepath = joinpath(@__DIR__, "GT Files", "$Location.mat")

# ╔═╡ 1acb66b3-8e0b-42b9-a15a-f07e47222aa9
CACHEDIR = joinpath(@__DIR__, "cache_files", "KAffine_Pavia")

# ╔═╡ ef6de7b6-02ea-4b87-951b-66042ea42c09
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 51bed330-5d55-4104-b890-2e121b6e75b6
vars = matread(filepath)

# ╔═╡ 574bc7e0-547d-4a76-b4cd-8dd40ffb101e
vars_gt = matread(gt_filepath)

# ╔═╡ 9d91845f-c36c-41eb-a5e8-17cf16a40386
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ e6dc372e-85cd-4134-9f53-1f64ca1e6da2
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ 307818a8-54b9-4798-8967-45d18f724176
data = vars[data_key]

# ╔═╡ 043e454a-2dc3-4c11-86bc-de1944761e51
gt_data = vars_gt[gt_key]

# ╔═╡ 6651c2fa-b448-4221-9274-7bd8d6684dd4
gt_labels = sort(unique(gt_data))

# ╔═╡ a1f3cf5a-8657-4067-96d9-1a8412042bf5
bg_indices = findall(gt_data .== 0)

# ╔═╡ 71723dd4-e832-4c2e-93b8-06a38ec4fb68
md"""
### Defining mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ 46cbeaad-ec9b-4dd0-924d-11141363b7b8
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ f60a370c-142d-40f0-9398-93842a3bb965
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ d22347d8-0846-4214-b57e-41cb93f9d82f
md"""
### affine approximation
"""

# ╔═╡ 2f4b6696-2595-4220-88a0-a9a49479e109
function affine_approx(X, k)
	bhat = mean(eachcol(X))
	Uhat = svd(X - bhat*ones(size(X,2))').U[:,1:k]
	return bhat, Uhat
end

# ╔═╡ 150c06c3-bc8c-46be-8988-716678780329
function dist(x, (bhat, Uhat))
	return norm(x - (Uhat*Uhat'*(x-bhat) + bhat))
end

# ╔═╡ 07d5092f-03c4-4d2e-97a7-de5885e278c4
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

# ╔═╡ 58035f5a-52c7-4006-bcc2-c41de0a68aa4
function batch_KAffine(X, d; niters=100, nruns=10)

	D, N = size(X)
	runs = Vector{Tuple{Vector{Matrix{Float64}}, Vector{Vector{Float64}}, Vector{Int}, Float64}}(undef, nruns)
	@progress for idx in 1:nruns
		U, b, c = cachet(joinpath(CACHEDIR, "run_1_$Location-$idx.bson")) do
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

# ╔═╡ bc382656-a63f-42a5-aa00-540a8fb37132
KAffine_runs = 50

# ╔═╡ 31ef0f16-1e80-48be-b2d0-158aa01fb117
KAffine_Clustering = batch_KAffine(permutedims(data[mask, :]), fill(1, n_clusters); niters=100, nruns=KAffine_runs)

# ╔═╡ 6bbe5785-b592-4872-9a35-355473b4d19a
min_idx_KAffine = argmax(KAffine_Clustering[i][4] for i in 1:KAffine_runs)

# ╔═╡ 0e380268-51a5-4eb4-a05e-4a99412b8509


# ╔═╡ 018a4a3a-7533-45b3-83f6-31a02e4d50a9
KAffine_Results = KAffine_Clustering[min_idx_KAffine][3]

# ╔═╡ 9a429935-ec53-4f18-8229-cf15e8aabbc1
relabel_maps = Dict(
	"Pavia" => Dict(
	0 => 0,
	1 => 1,
	2 => 8,
	3 => 6,
	4 => 9,
	5 => 5,
	6 => 3,
	7 => 7,
	8 => 4,
	9 => 2
),
	"PaviaUni" => Dict(
	0 => 0,
	1 => 1,
	2 => 3,
	3 => 8,
	4 => 4,
	5 => 5,
	6 => 9,
	7 => 7,
	8 => 2,
	9 => 6,
)
)

# ╔═╡ d10767ce-0944-4fd1-80dc-7eabf89d1388
relabel_keys = relabel_maps[Location]

# ╔═╡ bd7a78da-80c2-407d-920f-32e65251e0f7
D_relabel = [relabel_keys[label] for label in KAffine_Results]

# ╔═╡ 150a2548-7cb7-48c6-a889-27fbb459e84c
with_theme() do
	

	# Create figure
	fig = Figure(; size=(800, 700))
	colors = Makie.Colors.distinguishable_colors(n_clusters + 1)
	

	# Show data
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=15)
	
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false, ticklabelsize=:8)

	# Show cluster map
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="KAffine Clustering Results - $Location", titlesize=15)
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= D_relabel
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
	Colorbar(fig[2,2], hm, tellwidth=false, vertical=false, ticklabelsize=:8)
	
	fig
end

# ╔═╡ dad3de1b-9cde-44c6-beb2-d56f0083f24d
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

# ╔═╡ 32d1f93f-a711-467f-997f-4f5de2eb3711
with_theme() do
	fig = Figure(; size=(800, 650))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels_re, yticks = 1:true_labels_re, title="K-Affinespaces - $Location - Confusion Matrix", titlesize=15)
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
# ╟─e510d702-e7f0-4c01-a8bc-12bdb1b80cab
# ╠═3ff5fa73-4af0-4f51-b119-27807de8f95e
# ╠═39cde77c-5647-4262-aa38-ce373415c2ac
# ╠═60351d80-ef17-4568-a0eb-5f8f5552159c
# ╠═1e2d8cad-f61b-49b3-a9bd-685e05108c1a
# ╠═acd2e6f9-122c-45dc-9dfe-779d11c99e89
# ╠═1acb66b3-8e0b-42b9-a15a-f07e47222aa9
# ╠═ef6de7b6-02ea-4b87-951b-66042ea42c09
# ╠═51bed330-5d55-4104-b890-2e121b6e75b6
# ╠═574bc7e0-547d-4a76-b4cd-8dd40ffb101e
# ╠═9d91845f-c36c-41eb-a5e8-17cf16a40386
# ╠═e6dc372e-85cd-4134-9f53-1f64ca1e6da2
# ╠═307818a8-54b9-4798-8967-45d18f724176
# ╠═043e454a-2dc3-4c11-86bc-de1944761e51
# ╠═6651c2fa-b448-4221-9274-7bd8d6684dd4
# ╠═a1f3cf5a-8657-4067-96d9-1a8412042bf5
# ╟─71723dd4-e832-4c2e-93b8-06a38ec4fb68
# ╠═46cbeaad-ec9b-4dd0-924d-11141363b7b8
# ╠═f60a370c-142d-40f0-9398-93842a3bb965
# ╟─d22347d8-0846-4214-b57e-41cb93f9d82f
# ╠═2f4b6696-2595-4220-88a0-a9a49479e109
# ╠═150c06c3-bc8c-46be-8988-716678780329
# ╠═07d5092f-03c4-4d2e-97a7-de5885e278c4
# ╠═58035f5a-52c7-4006-bcc2-c41de0a68aa4
# ╠═bc382656-a63f-42a5-aa00-540a8fb37132
# ╠═31ef0f16-1e80-48be-b2d0-158aa01fb117
# ╠═6bbe5785-b592-4872-9a35-355473b4d19a
# ╠═0e380268-51a5-4eb4-a05e-4a99412b8509
# ╠═018a4a3a-7533-45b3-83f6-31a02e4d50a9
# ╠═9a429935-ec53-4f18-8229-cf15e8aabbc1
# ╠═d10767ce-0944-4fd1-80dc-7eabf89d1388
# ╠═bd7a78da-80c2-407d-920f-32e65251e0f7
# ╠═150a2548-7cb7-48c6-a889-27fbb459e84c
# ╠═dad3de1b-9cde-44c6-beb2-d56f0083f24d
# ╠═32d1f93f-a711-467f-997f-4f5de2eb3711
