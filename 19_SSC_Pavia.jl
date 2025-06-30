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

# ╔═╡ 2c184464-068a-4f96-b910-0da43a386055
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ c606166d-c02f-4d77-9107-8736889f2ab8
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ b4b55c7f-93f7-42ff-b0c3-8ddaac58262b
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

# ╔═╡ d3b8ac62-2f70-4c6f-9d5f-157c432d25d9
@bind Location Select(["Pavia", "PaviaUni",])

# ╔═╡ ab355b03-fc25-421b-9368-6ab268d3a04c
filepath = joinpath(@__DIR__, "MAT Files", "$Location.mat")

# ╔═╡ 348e2ae0-3aaa-40e0-af60-d5aee5c07782
gt_filepath = joinpath(@__DIR__, "GT Files", "$Location.mat")

# ╔═╡ dcf0b67f-dc73-4c8e-ad68-b4faa91df73c
vars = matread(filepath)

# ╔═╡ bdedd650-ffeb-4e4e-ad84-5fac56a06921
vars_gt = matread(gt_filepath)

# ╔═╡ 64da7998-1af4-4070-aa5e-238cb90f1314
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ d9a198ac-0d02-4ddf-aebb-3ee919be6ce0
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ fd91a607-5d4b-46ed-b981-3c1bb699fa7e
md"
#### Original Data matrix
"

# ╔═╡ ef448820-659e-4576-8971-53a0b945dc22
data = vars[data_key]

# ╔═╡ 7bfe296c-c848-443e-a4fb-119b99e3352c
md"
#### Ground Truth Labels
"

# ╔═╡ 79a33702-9479-410a-a651-92c8f9b9da3b
gt_data = vars_gt[gt_key]

# ╔═╡ a87eef17-9c92-4c3a-93af-88e7fb1e7652
gt_labels = sort(unique(gt_data))

# ╔═╡ 5e113ae0-c528-47a4-97dd-e1ad65c532a0
bg_indices = findall(gt_data .== 0)

# ╔═╡ d0674ece-1b26-4547-9d17-f7883e01059a
md"""
### Defining mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ dea1bc8a-eb2b-4279-bbb8-349bafc39d5e
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ f7d03a7f-5f6b-4919-af5e-3ea2fa014680
n_classes = length(unique(gt_data)) - 1

# ╔═╡ 19eff4cc-834b-4c12-b0b3-6ca9dc220b96
md"""
### Sparse Subspace Clustering
"""

# ╔═╡ ce1bfc8e-62e5-4c61-a9f2-a7bb9a17a9d7
function ssc(X)
	N = size(X, 2) 		# Number of data points
	C = zeros(N, N)		# Sparse-Subspace matrix
end

# ╔═╡ Cell order:
# ╟─b4b55c7f-93f7-42ff-b0c3-8ddaac58262b
# ╠═2c184464-068a-4f96-b910-0da43a386055
# ╠═c606166d-c02f-4d77-9107-8736889f2ab8
# ╠═d3b8ac62-2f70-4c6f-9d5f-157c432d25d9
# ╠═ab355b03-fc25-421b-9368-6ab268d3a04c
# ╠═348e2ae0-3aaa-40e0-af60-d5aee5c07782
# ╠═dcf0b67f-dc73-4c8e-ad68-b4faa91df73c
# ╠═bdedd650-ffeb-4e4e-ad84-5fac56a06921
# ╠═64da7998-1af4-4070-aa5e-238cb90f1314
# ╠═d9a198ac-0d02-4ddf-aebb-3ee919be6ce0
# ╟─fd91a607-5d4b-46ed-b981-3c1bb699fa7e
# ╠═ef448820-659e-4576-8971-53a0b945dc22
# ╟─7bfe296c-c848-443e-a4fb-119b99e3352c
# ╠═79a33702-9479-410a-a651-92c8f9b9da3b
# ╠═a87eef17-9c92-4c3a-93af-88e7fb1e7652
# ╠═5e113ae0-c528-47a4-97dd-e1ad65c532a0
# ╟─d0674ece-1b26-4547-9d17-f7883e01059a
# ╠═dea1bc8a-eb2b-4279-bbb8-349bafc39d5e
# ╠═f7d03a7f-5f6b-4919-af5e-3ea2fa014680
# ╟─19eff4cc-834b-4c12-b0b3-6ca9dc220b96
# ╠═ce1bfc8e-62e5-4c61-a9f2-a7bb9a17a9d7
