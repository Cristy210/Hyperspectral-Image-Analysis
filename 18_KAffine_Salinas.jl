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
KAffine_runs = 30

# ╔═╡ b9deb150-9038-4e97-b7bf-8a6f71c61210
KAffine_Clustering = batch_KAffine(permutedims(reshape(data, :, size(data, 3))), fill(1, 12); niters=100, nruns=KAffine_runs)

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
# ╠═673ccd2a-e734-46cc-9357-52bafd18bd03
# ╠═2b248a37-3fd3-44d9-9ef5-10ad1988955a
# ╠═05c8124a-c8ab-436d-87ed-87c974f76f8b
# ╠═73de4258-fdda-4662-8e79-913d1334cd69
# ╠═859415ba-eeb2-479a-acab-c4902ab5ed88
# ╠═5e6d28e7-58e5-4a67-9da5-50efedbf9b15
# ╠═b9deb150-9038-4e97-b7bf-8a6f71c61210
