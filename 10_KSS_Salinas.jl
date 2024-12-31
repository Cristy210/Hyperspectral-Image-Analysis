### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 51938f91-d03b-4d21-9308-627c308a3b1a
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 8ba89f39-057c-469e-b8da-961fc3b01709
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ e5e87648-8a71-4759-ac7a-209b91122916
html"""<style>
main {
    max-width: 66%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 0c2c46d7-cdb6-42e1-b15a-2093512d7082
filepath = joinpath(@__DIR__, "MAT Files", "Salinas_corrected.mat")

# ╔═╡ 07a5ee92-8b4e-4c80-a44f-12d3554a8525
gt_filepath = joinpath(@__DIR__, "GT Files", "Salinas_gt.mat")

# ╔═╡ a60a2575-062a-4e9b-90ce-da8fa082a94d
CACHEDIR = joinpath(@__DIR__, "cache_files", "KSS_Salinas")

# ╔═╡ 0fffb162-c684-436e-ad63-3a96f7e2cbf2
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 7883496f-7deb-4a59-b538-069ecd0511ba
vars = matread(filepath)

# ╔═╡ 40d2057e-2ccf-4749-bdc7-f56b165fe8cb
vars_gt = matread(gt_filepath)

# ╔═╡ 6877f3fd-760e-4cef-9123-30573874fbbc
data = vars["salinas_corrected"]

# ╔═╡ 8eb86990-2b96-4689-b271-4cb9d11d8007
gt_data = vars_gt["salinas_gt"]

# ╔═╡ 2145ccfd-749a-4d6e-a5fd-0b88b8f60912
gt_labels = sort(unique(gt_data))

# ╔═╡ c1643ed0-4592-45c5-ba1c-1808b404a1f4
bg_indices = findall(gt_data .== 0)

# ╔═╡ bb09c5df-49bf-41af-be2e-41b677afd05b
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ 5f04dddc-8553-4f10-aacf-5ebe48090741
md"""
### Defining mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ b987c1c6-7975-4257-a0e9-b2f353811440
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ 3d21c3f9-037f-4ef5-b595-fdf95b500fa8
md"""
### K-Subspaces
"""

# ╔═╡ 52acc824-7384-4e49-b25e-da43ccb75464
function polar(X)
	U, _, V = svd(X)
	U*V'
end

# ╔═╡ f30f7bfe-3b13-476f-947b-42a608f80f97
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

# ╔═╡ a7186736-a8e0-42fb-83e9-b55d9c0980be
function batch_KSS(X, d; niters=100, nruns=10)
	D, N = size(X)
	runs = Vector{Tuple{Vector{Matrix{Float64}}, Vector{Int}, Float64}}(undef, nruns)
	@progress for idx in 1:nruns
		U, c = cachet(joinpath(CACHEDIR, "run_Salinas-$idx.bson")) do
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

# ╔═╡ 2017fd19-83b3-4ea5-9276-b916eff67769
fill(2, n_clusters)

# ╔═╡ 4e464579-1c70-4a4d-94ad-033d59f01d80
KSS_Clustering = batch_KSS(permutedims(data[mask, :]), fill(1, n_clusters); niters=100, nruns=100)

# ╔═╡ 07d23d51-50c3-47fd-bcd5-7c6a967818aa
min_idx_KSS = argmax(KSS_Clustering[i][3] for i in 1:n_clusters)

# ╔═╡ d897e145-3153-4f2b-95d6-5a6f7f2d3086
KSS_Results = KSS_Clustering[min_idx_KSS][2]

# ╔═╡ 76966901-77ba-4b90-b9df-61a1f857038f
unique(KSS_Results)

# ╔═╡ 9cfcfeab-58c1-4c13-ac19-524ecd0ecba8
relabel_map = Dict(
	0 => 0,
	1 => 1,
	2 => 2,
	3 => 3,
	4 => 4,
	5 => 5,
	6 => 6,
	7 => 7,
	8 => 8,
	9 => 9,
	10 => 10,
	11 => 11,
	12 => 12,
	13 => 13,
	14 => 14,
	15 => 15,
	16 => 16,
)

# ╔═╡ 79101292-84d7-4be7-8c4a-3a46f5dd1062
D_relabel = [relabel_map[label] for label in KSS_Results]

# ╔═╡ a40a1f05-852a-464e-b207-efee9b1d2fb3
with_theme() do
	# assignments, idx = spec_aligned, spec_clustering_idx

	# Create figure
	fig = Figure(; size=(1200, 750))
	colors = Makie.Colors.distinguishable_colors(n_clusters + 1)
	# colors_re = Makie.Colors.distinguishable_colors(length(re_labels))

	# Show data
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=20)
	
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
	Colorbar(fig[2,1], hm, tellwidth=false, vertical=false)

	# Show cluster map
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="KSS Clustering Results", titlesize=20)
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= KSS_Results
	hm = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, n_clusters))
	Colorbar(fig[2,2], hm, tellwidth=false, vertical=false)
	
	fig
end

# ╔═╡ Cell order:
# ╟─e5e87648-8a71-4759-ac7a-209b91122916
# ╠═51938f91-d03b-4d21-9308-627c308a3b1a
# ╠═8ba89f39-057c-469e-b8da-961fc3b01709
# ╠═0c2c46d7-cdb6-42e1-b15a-2093512d7082
# ╠═07a5ee92-8b4e-4c80-a44f-12d3554a8525
# ╠═a60a2575-062a-4e9b-90ce-da8fa082a94d
# ╠═0fffb162-c684-436e-ad63-3a96f7e2cbf2
# ╠═7883496f-7deb-4a59-b538-069ecd0511ba
# ╠═40d2057e-2ccf-4749-bdc7-f56b165fe8cb
# ╠═6877f3fd-760e-4cef-9123-30573874fbbc
# ╠═8eb86990-2b96-4689-b271-4cb9d11d8007
# ╠═2145ccfd-749a-4d6e-a5fd-0b88b8f60912
# ╠═c1643ed0-4592-45c5-ba1c-1808b404a1f4
# ╠═bb09c5df-49bf-41af-be2e-41b677afd05b
# ╟─5f04dddc-8553-4f10-aacf-5ebe48090741
# ╠═b987c1c6-7975-4257-a0e9-b2f353811440
# ╟─3d21c3f9-037f-4ef5-b595-fdf95b500fa8
# ╠═52acc824-7384-4e49-b25e-da43ccb75464
# ╠═f30f7bfe-3b13-476f-947b-42a608f80f97
# ╠═a7186736-a8e0-42fb-83e9-b55d9c0980be
# ╠═2017fd19-83b3-4ea5-9276-b916eff67769
# ╠═4e464579-1c70-4a4d-94ad-033d59f01d80
# ╠═07d23d51-50c3-47fd-bcd5-7c6a967818aa
# ╠═d897e145-3153-4f2b-95d6-5a6f7f2d3086
# ╠═76966901-77ba-4b90-b9df-61a1f857038f
# ╠═9cfcfeab-58c1-4c13-ac19-524ecd0ecba8
# ╠═79101292-84d7-4be7-8c4a-3a46f5dd1062
# ╠═a40a1f05-852a-464e-b207-efee9b1d2fb3
