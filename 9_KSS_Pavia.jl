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

# ╔═╡ b17ad74a-e088-4a72-aa0e-eef3fa7f5ca3
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 5685a44c-81db-4b63-b9f1-60eb86009a9c
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ cec84524-bceb-4a8d-a30b-e82faca68cc7
html"""<style>
main {
    max-width: 66%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 1a652c20-e339-4df3-9ccb-2ccd128ac56d
@bind Location Select(["Pavia", "PaviaUni",])

# ╔═╡ aa573616-092e-42d0-a7af-570f6d1dd6e5
filepath = joinpath(@__DIR__, "MAT Files", "$Location.mat")

# ╔═╡ 347bd60b-b1f2-44ed-b158-f439369ae38a
gt_filepath = joinpath(@__DIR__, "GT Files", "$Location.mat")

# ╔═╡ bb23b272-abef-481f-a7cc-9f11f9eedc2a
CACHEDIR = joinpath(@__DIR__, "cache_files", "KSS_Pavia")

# ╔═╡ 62a2322b-a16b-40c6-8f18-2dd0c4b25981
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 05c90322-b232-4112-81be-23971472be29
vars = matread(filepath)

# ╔═╡ 679dd062-dfc0-4cbf-bcb2-b5b755b4348f
vars_gt = matread(gt_filepath)

# ╔═╡ 5ce6d6c2-13ab-4ed8-a8c8-ca05beb63455
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ 30ba8e64-f32a-42f2-80e8-1f6ad92a94a6
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ 2770834a-eb75-4892-974f-7598191aaf55
data = vars[data_key]

# ╔═╡ 9c48fa8b-cd33-4b0c-9cc9-fc64f08b7739
gt_data = vars_gt[gt_key]

# ╔═╡ 8b339a3b-3c1f-4d85-bb25-7c079765b79c
gt_labels = sort(unique(gt_data))

# ╔═╡ 4f93390d-bfd9-4097-ae8a-b27389af8c62
bg_indices = findall(gt_data .== 0)

# ╔═╡ 2b260d09-8853-42db-9c84-8902a5c0f7bc
md"""
### Defining mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ f8fe700a-9bfc-42d2-979d-60a500da4d6b
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ 2d9f470e-314e-4d97-b415-1b86d654df7e
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ 6d6d0675-bf03-4a80-b69f-963da7f3b1a3
md"""
### K-Subspaces
"""

# ╔═╡ b8991416-ba95-46f2-907d-7a43edf6c890
function polar(X)
	U, _, V = svd(X)
	U*V'
end

# ╔═╡ 62a83180-5565-4f6e-9210-89ee42fc6e6b
"""
	KSS(X, d; niters=100)

Run K-subspaces on the data matrix `X`
with subspace dimensions `d[1], ..., d[K]`,
treating the columns of `X` as the datapoints.
"""
function KSS(X, d; niters=100, Uinit=polar.(randn.(size(X, 1), collect(d))))
	K = length(d)
	D, N = size(X)

	# Initialize
	U = deepcopy(Uinit)
	c = [argmax(norm(U[k]' * view(X, :, i)) for k in 1:K) for i in 1:N]
	c_prev = copy(c)

	# Iterations
	@progress for t in 1:niters
		# Update subspaces
		for k in 1:K
			ilist = findall(==(k), c)
			# println("Cluster $k, ilist size: ", length(ilist))
			if isempty(ilist)
				# println("Initializing $k subspace")
				U[k] = polar(randn(D, d[k]))
			else
				A = view(X, :, ilist) * transpose(view(X, :, ilist))
				decomp, history = partialschur(A; nev=d[k], which=:LR)
				@show history
				# U[k] = tsvd(view(X, :, ilist), d[k])[1]
				U[k] = decomp.Q
			end
		end

		# Update clusters
		for i in 1:N
			c[i] = argmax(norm(U[k]' * view(X, :, i)) for k in 1:K)
		end

		# Break if clusters did not change, update otherwise
		if c == c_prev
			@info "Terminated early at iteration $t"
			break
		end
		c_prev .= c
	end

	return U, c
end

# ╔═╡ 3f0bc139-107e-4dcd-a58f-e41bd6e980f6
function batch_KSS(X, d; niters=100, nruns=10)
	D, N = size(X)
	runs = Vector{Tuple{Vector{Matrix{Float64}}, Vector{Int}, Float64}}(undef, nruns)
	@progress for idx in 1:nruns
		U, c = cachet(joinpath(CACHEDIR, "run_$Location-$idx.bson")) do
			Random.seed!(idx)
			KSS(X, d; niters=niters)
		end

		total_cost = 0
		for i in 1:N
			cost = norm(U[c[i]]' * view(X, :, i))
			total_cost += cost
		end

		runs[idx] = (U, c, total_cost)

		
	end

	 return runs
end

# ╔═╡ 015883f0-2cbe-4699-9971-552b6d174688
fill(2, n_clusters)

# ╔═╡ 02732d8e-171c-4b15-b5e7-b0e79941c999
KSS_Clustering = batch_KSS(permutedims(data[mask, :]), fill(1, n_clusters); niters=100, nruns=100)

# ╔═╡ 77f09f83-eb68-4359-a1bf-5f30d002746c
min_idx_KSS = argmax(KSS_Clustering[i][3] for i in 1:100)

# ╔═╡ aa903bca-1405-4b12-80b8-809113f9dc27


# ╔═╡ 71584f7f-0477-4b3e-96db-4f1a8b2cf8b4
KSS_Results = KSS_Clustering[min_idx_KSS][2]

# ╔═╡ 8c6174a3-0782-4d44-a8b6-b6ec92668871
relabel_maps = Dict(
	"Pavia" => Dict(
	0 => 0,
	1 => 1,
	2 => 4,
	3 => 8,
	4 => 2,
	5 => 7,
	6 => 3,
	7 => 5,
	8 => 6,
	9 => 9
),
	"PaviaUni" => Dict(
	0 => 0,
	1 => 5,
	2 => 8,
	3 => 3,
	4 => 9,
	5 => 1,
	6 => 6,
	7 => 4,
	8 => 2,
	9 => 7,
)
)

# ╔═╡ f0dfbe41-44dd-4c4c-b544-edc2faa7d3ff
relabel_keys = relabel_maps[Location]

# ╔═╡ 8012832c-42b0-48a4-97b0-bd1f61ed5e3b
D_relabel = [relabel_keys[label] for label in KSS_Results]

# ╔═╡ 574b1660-cea0-4626-aa47-7eeff7d93e61
with_theme() do
	# assignments, idx = spec_aligned, spec_clustering_idx

	# Create figure
	fig = Figure(; size=(700, 650))
	colors = Makie.Colors.distinguishable_colors(n_clusters + 1)
	# colors_re = Makie.Colors.distinguishable_colors(length(re_labels))

	# Show data
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=20)
	
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, 9))
	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false)

	# Show cluster map
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="KSS Clustering Results", titlesize=20)
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= D_relabel
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, 9))
	Colorbar(fig[2,2], hm, tellwidth=false, vertical=false)
	
	fig
end

# ╔═╡ bb21a2c0-6011-4631-ac1c-3df0832def1d
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

# ╔═╡ 106b6d99-27a5-4a0e-8367-39c69d338a9f
with_theme() do
	fig = Figure(; size=(800, 700))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels_re, yticks = 1:true_labels_re, title="Confusion Matrix - KSS Clustering - Pavia")
	hm = heatmap!(ax, permutedims(confusion_matrix_re), colormap=:viridis)
	pm = permutedims(confusion_matrix_re)

	for i in 1:true_labels_re, j in 1:predicted_labels_re
        value = round(pm[i, j], digits=1)
        text!(ax, i - 0.02, j - 0.1, text = "$value", color=:black, align = (:center, :center), fontsize=14)
    end
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ 19298cbb-5ef1-462b-8624-47ab63d14ca4
with_theme() do
    fig = Figure(; size=(1300, 700))
	supertitle = Label(fig[0, 1:3], "Spectrum Analysis of Clustering Results with Corresponding Ground Truth Label", fontsize=20, halign=:center, valign=:top)
	
    grid_1 = GridLayout(fig[1, 1]; nrow=2, ncol=1)
	grid_2 = GridLayout(fig[1, 2]; nrow=5, ncol=2)
	grid_3 = GridLayout(fig[1, 3]; nrow=2, ncol=1)
	masked_gt = dropdims(gt_data[mask, :], dims=2)
	masked_2darray = permutedims(data[mask, :])

	

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


# ╔═╡ Cell order:
# ╠═cec84524-bceb-4a8d-a30b-e82faca68cc7
# ╠═b17ad74a-e088-4a72-aa0e-eef3fa7f5ca3
# ╠═5685a44c-81db-4b63-b9f1-60eb86009a9c
# ╠═1a652c20-e339-4df3-9ccb-2ccd128ac56d
# ╠═aa573616-092e-42d0-a7af-570f6d1dd6e5
# ╠═347bd60b-b1f2-44ed-b158-f439369ae38a
# ╠═bb23b272-abef-481f-a7cc-9f11f9eedc2a
# ╠═62a2322b-a16b-40c6-8f18-2dd0c4b25981
# ╠═05c90322-b232-4112-81be-23971472be29
# ╠═679dd062-dfc0-4cbf-bcb2-b5b755b4348f
# ╠═5ce6d6c2-13ab-4ed8-a8c8-ca05beb63455
# ╠═30ba8e64-f32a-42f2-80e8-1f6ad92a94a6
# ╠═2770834a-eb75-4892-974f-7598191aaf55
# ╠═9c48fa8b-cd33-4b0c-9cc9-fc64f08b7739
# ╠═8b339a3b-3c1f-4d85-bb25-7c079765b79c
# ╠═4f93390d-bfd9-4097-ae8a-b27389af8c62
# ╟─2b260d09-8853-42db-9c84-8902a5c0f7bc
# ╠═f8fe700a-9bfc-42d2-979d-60a500da4d6b
# ╠═2d9f470e-314e-4d97-b415-1b86d654df7e
# ╟─6d6d0675-bf03-4a80-b69f-963da7f3b1a3
# ╠═b8991416-ba95-46f2-907d-7a43edf6c890
# ╠═62a83180-5565-4f6e-9210-89ee42fc6e6b
# ╠═3f0bc139-107e-4dcd-a58f-e41bd6e980f6
# ╠═015883f0-2cbe-4699-9971-552b6d174688
# ╠═02732d8e-171c-4b15-b5e7-b0e79941c999
# ╠═77f09f83-eb68-4359-a1bf-5f30d002746c
# ╠═aa903bca-1405-4b12-80b8-809113f9dc27
# ╠═71584f7f-0477-4b3e-96db-4f1a8b2cf8b4
# ╠═8c6174a3-0782-4d44-a8b6-b6ec92668871
# ╠═f0dfbe41-44dd-4c4c-b544-edc2faa7d3ff
# ╠═8012832c-42b0-48a4-97b0-bd1f61ed5e3b
# ╠═574b1660-cea0-4626-aa47-7eeff7d93e61
# ╠═bb21a2c0-6011-4631-ac1c-3df0832def1d
# ╠═106b6d99-27a5-4a0e-8367-39c69d338a9f
# ╠═19298cbb-5ef1-462b-8624-47ab63d14ca4
